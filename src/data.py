"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Build single-Gaussian DPO pair data and KTO labeled data.
"""

import torch

from .distributions import sample_gaussian


def _sample_outside_band(mu_ref: float, sigma_ref: float, target: float, radius: float, n: int, device: str):
    kept = []
    attempts = 0
    max_attempts = 200
    batch = max(256, n)

    # DPO hardness is controlled by excluding points near the optimum. We keep
    # sampling until we have enough points outside the forbidden band.
    while sum(t.numel() for t in kept) < n and attempts < max_attempts:
        attempts += 1
        y = sample_gaussian(mu_ref, sigma_ref, batch, device)
        outside = y[torch.abs(y - target) >= radius]
        if outside.numel() > 0:
            remaining = n - sum(t.numel() for t in kept)
            kept.append(outside[:remaining])

    if not kept:
        raise RuntimeError("Unable to sample DPO points outside the excluded high-reward band")

    y = torch.cat(kept)
    if y.numel() < n:
        raise RuntimeError("Unable to sample enough DPO points outside the excluded high-reward band")
    return y[:n]


def make_dpo_pairs(
    mu_ref: float,
    sigma_ref: float,
    target: float,
    n: int,
    device: str,
    good_ratio: float = None,
    zone_half_width: float = 1.5,
):
    # For DPO, good_ratio now controls the width of the excluded high-reward
    # region through kappa = 1 - good_ratio and radius = kappa * sigma_ref.
    if good_ratio is None:
        y1 = sample_gaussian(mu_ref, sigma_ref, n, device)
        y2 = sample_gaussian(mu_ref, sigma_ref, n, device)
    else:
        kappa = 1.0 - good_ratio
        radius = kappa * sigma_ref
        y1 = _sample_outside_band(mu_ref, sigma_ref, target, radius, n, device)
        y2 = _sample_outside_band(mu_ref, sigma_ref, target, radius, n, device)

    # Preference winners are still whichever point is closer to the target.
    dist1 = torch.abs(y1 - target)
    dist2 = torch.abs(y2 - target)
    winner_is_y1 = dist1 <= dist2

    y_w = torch.where(winner_is_y1, y1, y2)
    y_l = torch.where(winner_is_y1, y2, y1)

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

    # KTO directly samples positive and negative examples until the requested
    # label counts are matched.
    while (len(good_list) < good_needed or len(bad_list) < bad_needed) and attempts < max_attempts:
        attempts += 1
        y = sample_gaussian(mu_ref, sigma_ref, batch, device)
        in_zone = (y >= zone_min) & (y <= zone_max)

        # For KTO, the data-generation step explicitly controls the positive and
        # negative label counts before shuffling the final labeled dataset.
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

    # Shuffle the concatenated labeled examples so the training loader sees a
    # mixed sequence of positive and negative feedback.
    y = torch.cat([good, bad])
    labels = torch.cat(
        [torch.ones_like(good, dtype=torch.float), torch.zeros_like(bad, dtype=torch.float)]
    )

    perm = torch.randperm(y.numel(), device=device)
    return y[perm], labels[perm]
