import torch

from .distributions import sample_gaussian


def _sample_balanced_dpo_pairs(
    mu_ref: float,
    sigma_ref: float,
    target: float,
    zone_half_width: float,
    n: int,
    good_ratio: float,
    device: str,
):
    zone_min = target - zone_half_width
    zone_max = target + zone_half_width

    good_needed = int(round(n * good_ratio))
    bad_needed = n - good_needed
    good_w = []
    good_l = []
    bad_w = []
    bad_l = []

    attempts = 0
    max_attempts = 200
    batch = max(256, n)

    while (
        sum(t.numel() for t in good_w) < good_needed or sum(t.numel() for t in bad_w) < bad_needed
    ) and attempts < max_attempts:
        attempts += 1
        y1 = sample_gaussian(mu_ref, sigma_ref, batch, device)
        y2 = sample_gaussian(mu_ref, sigma_ref, batch, device)

        dist1 = torch.abs(y1 - target)
        dist2 = torch.abs(y2 - target)
        winner_is_y1 = dist1 <= dist2

        y_w = torch.where(winner_is_y1, y1, y2)
        y_l = torch.where(winner_is_y1, y2, y1)

        winner_in_zone = (y_w >= zone_min) & (y_w <= zone_max)

        if sum(t.numel() for t in good_w) < good_needed:
            take_w = y_w[winner_in_zone]
            take_l = y_l[winner_in_zone]
            if take_w.numel() > 0:
                remaining = good_needed - sum(t.numel() for t in good_w)
                good_w.append(take_w[:remaining])
                good_l.append(take_l[:remaining])

        if sum(t.numel() for t in bad_w) < bad_needed:
            take_w = y_w[~winner_in_zone]
            take_l = y_l[~winner_in_zone]
            if take_w.numel() > 0:
                remaining = bad_needed - sum(t.numel() for t in bad_w)
                bad_w.append(take_w[:remaining])
                bad_l.append(take_l[:remaining])

    good_w = torch.cat(good_w) if good_w else torch.tensor([], device=device)
    good_l = torch.cat(good_l) if good_l else torch.tensor([], device=device)
    bad_w = torch.cat(bad_w) if bad_w else torch.tensor([], device=device)
    bad_l = torch.cat(bad_l) if bad_l else torch.tensor([], device=device)

    if good_w.numel() < good_needed or bad_w.numel() < bad_needed:
        raise RuntimeError("Unable to construct DPO pairs with the requested good_ratio")

    y_w = torch.cat([good_w, bad_w], dim=0)
    y_l = torch.cat([good_l, bad_l], dim=0)
    perm = torch.randperm(y_w.numel(), device=device)
    return y_w[perm], y_l[perm]


def make_dpo_pairs(
    mu_ref: float,
    sigma_ref: float,
    target: float,
    n: int,
    device: str,
    good_ratio: float = None,
    zone_half_width: float = 1.5,
):
    y1 = sample_gaussian(mu_ref, sigma_ref, n, device)
    y2 = sample_gaussian(mu_ref, sigma_ref, n, device)

    dist1 = torch.abs(y1 - target)
    dist2 = torch.abs(y2 - target)
    winner_is_y1 = dist1 <= dist2

    y_w = torch.where(winner_is_y1, y1, y2)
    y_l = torch.where(winner_is_y1, y2, y1)

    if good_ratio is not None:
        return _sample_balanced_dpo_pairs(
            mu_ref,
            sigma_ref,
            target,
            zone_half_width,
            n,
            good_ratio,
            device,
        )

    return y_w, y_l


def make_kto_samples(
    mu_ref: float,
    sigma_ref: float,
    target: float,
    zone_half_width: float,
    n: int,
    good_fraction: float,
    device: str,
    delta: float = 0.0,
    good_ratio: float = None,
):
    zone_half_width = zone_half_width + delta
    zone_min = target - zone_half_width
    zone_max = target + zone_half_width

    if good_ratio is not None:
        good_fraction = good_ratio

    good_needed = int(round(n * good_fraction))
    bad_needed = n - good_needed

    good_list = []
    bad_list = []

    attempts = 0
    max_attempts = 200
    batch = max(256, n)

    while (len(good_list) < good_needed or len(bad_list) < bad_needed) and attempts < max_attempts:
        attempts += 1
        y = sample_gaussian(mu_ref, sigma_ref, batch, device)
        in_zone = (y >= zone_min) & (y <= zone_max)

        if len(good_list) < good_needed:
            y_good = y[in_zone]
            if y_good.numel() > 0:
                take = min(good_needed - len(good_list), y_good.numel())
                good_list.append(y_good[:take])

        if len(bad_list) < bad_needed:
            y_bad = y[~in_zone]
            if y_bad.numel() > 0:
                take = min(bad_needed - len(bad_list), y_bad.numel())
                bad_list.append(y_bad[:take])

    if len(good_list) == 0:
        good = torch.tensor([], device=device)
    else:
        good = torch.cat(good_list)

    if len(bad_list) == 0:
        bad = torch.tensor([], device=device)
    else:
        bad = torch.cat(bad_list)

    if good.numel() < good_needed or bad.numel() < bad_needed:
        raise RuntimeError("Unable to sample enough good/bad points; adjust zone or attempts")

    y = torch.cat([good, bad])
    labels = torch.cat(
        [torch.ones_like(good, dtype=torch.float), torch.zeros_like(bad, dtype=torch.float)]
    )

    perm = torch.randperm(y.numel(), device=device)
    return y[perm], labels[perm]
