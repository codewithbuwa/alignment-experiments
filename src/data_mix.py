"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Build mixture-based DPO pair data and KTO labeled data.
"""

import torch


def _sample_outside_band(ref_policy, target: float, radius: float, n: int, device: str):
    kept = []
    attempts = 0
    max_attempts = 200
    batch = max(256, n)

    # The mixture DPO dataset uses the same exclusion idea as the single case:
    # reject points too close to the optimum before forming preference pairs.
    while sum(t.numel() for t in kept) < n and attempts < max_attempts:
        attempts += 1
        y = ref_policy.sample(batch).to(device)
        outside = y[torch.abs(y - target) >= radius]
        if outside.numel() > 0:
            remaining = n - sum(t.numel() for t in kept)
            kept.append(outside[:remaining])

    if not kept:
        raise RuntimeError("Unable to sample mixture DPO points outside the excluded high-reward band")

    y = torch.cat(kept)
    if y.numel() < n:
        raise RuntimeError("Unable to sample enough mixture DPO points outside the excluded high-reward band")
    return y[:n]


def make_mixture_dpo_pairs(
    ref_policy,
    target: float,
    n: int,
    device: str,
    good_ratio: float = None,
    zone_half_width: float = 1.5,
    reference_sigma: float = 2.0,
):
    # For mixture DPO, use the same kappa = 1 - good_ratio exclusion rule as
    # the single setup, scaled by the reference sigma.
    if good_ratio is None:
        y1 = ref_policy.sample(n).to(device)
        y2 = ref_policy.sample(n).to(device)
    else:
        kappa = 1.0 - good_ratio
        radius = kappa * reference_sigma
        y1 = _sample_outside_band(ref_policy, target, radius, n, device)
        y2 = _sample_outside_band(ref_policy, target, radius, n, device)

    # Winners are whichever points stay closest to the target after exclusion.
    dist1 = torch.abs(y1 - target)
    dist2 = torch.abs(y2 - target)
    winner_is_y1 = dist1 <= dist2

    y_w = torch.where(winner_is_y1, y1, y2)
    y_l = torch.where(winner_is_y1, y2, y1)

    return y_w, y_l


def make_mixture_kto_samples(
    ref_policy,
    target: float,
    zone_half_width: float,
    n: int,
    device: str,
    delta: float = 0.0,
    good_ratio: float = None,
):
    zone_half_width = zone_half_width + delta
    zone_min = target - zone_half_width
    zone_max = target + zone_half_width

    if good_ratio is None:
        y = ref_policy.sample(n).to(device)
        labels = ((y >= zone_min) & (y <= zone_max)).float()
        return y, labels

    good_needed = int(round(n * good_ratio))
    bad_needed = n - good_needed

    good_list = []
    bad_list = []

    attempts = 0
    max_attempts = 200
    batch = max(256, n)

    # Mixture KTO reuses the same good/bad count matching pattern as the
    # single-Gaussian sampler.
    while (len(good_list) < good_needed or len(bad_list) < bad_needed) and attempts < max_attempts:
        attempts += 1
        y = ref_policy.sample(batch).to(device)
        in_zone = (y >= zone_min) & (y <= zone_max)

        # As in the single-Gaussian KTO setup, we build the labeled dataset by
        # explicitly collecting the requested number of positive and negative samples.
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
        raise RuntimeError("Unable to sample enough good/bad points from mixture reference.")

    # Shuffle after concatenation so the final labeled dataset is not ordered by class.
    y = torch.cat([good, bad])
    labels = torch.cat(
        [torch.ones_like(good, dtype=torch.float), torch.zeros_like(bad, dtype=torch.float)]
    )

    perm = torch.randperm(y.numel(), device=device)
    return y[perm], labels[perm]
