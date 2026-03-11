import torch

from .distributions import gaussian_log_prob


def implicit_reward(y, mu, rho, mu_ref: float, sigma_ref: float, beta: float):
    sigma = torch.exp(rho)
    logp = gaussian_log_prob(y, mu, sigma)
    logp_ref = gaussian_log_prob(y, y.new_tensor(mu_ref), y.new_tensor(sigma_ref))
    return beta * (logp - logp_ref)


def dpo_loss(y_w, y_l, mu, rho, mu_ref: float, sigma_ref: float, beta: float) -> torch.Tensor:
    h_w = implicit_reward(y_w, mu, rho, mu_ref, sigma_ref, beta)
    h_l = implicit_reward(y_l, mu, rho, mu_ref, sigma_ref, beta)
    return -torch.mean(torch.log(torch.sigmoid(h_w - h_l)))


def kto_loss(
    y,
    labels,
    mu,
    rho,
    mu_ref: float,
    sigma_ref: float,
    gamma: float,
    kl_value: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    h = implicit_reward(y, mu, rho, mu_ref, sigma_ref, beta)
    z = h - kl_value

    v = torch.zeros_like(z)
    v[labels == 1.0] = torch.sigmoid(z[labels == 1.0])
    v[labels == 0.0] = torch.sigmoid(-gamma * z[labels == 0.0])
    return torch.mean(1 - v)
