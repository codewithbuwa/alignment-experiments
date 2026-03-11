"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Construct reference mixture policies used by mixture experiments.
"""

import torch

from .policies import GaussianMixturePolicy


def make_reference_mixture(n_components: int, mu_ref: float, sigma_ref: float, device: str, mu_init=None):
    if mu_init is None:
        mu_init = torch.linspace(mu_ref - 4.0, mu_ref + 4.0, n_components)
    log_sigma_init = torch.log(torch.tensor(sigma_ref)) * torch.ones(n_components)
    logits_init = torch.zeros(n_components)
    return GaussianMixturePolicy(
        n_components=n_components,
        mu_init=mu_init,
        log_sigma_init=log_sigma_init,
        logits_init=logits_init,
        device=device,
    )
