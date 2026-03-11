import math
import torch


def gaussian_pdf(y: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((y - mu) / sigma) ** 2)


def gaussian_log_prob(y: torch.Tensor, mu, sigma) -> torch.Tensor:
    return -0.5 * (((y - mu) / sigma) ** 2 + 2 * torch.log(sigma) + math.log(2 * math.pi))


def gaussian_entropy(sigma: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.log(2 * math.pi * math.e * sigma ** 2)


def kl_gaussian(mu, sigma, mu_ref: float, sigma_ref: float) -> torch.Tensor:
    return torch.log(torch.tensor(sigma_ref, device=sigma.device) / sigma) + \
        (sigma ** 2 + (mu - mu_ref) ** 2) / (2 * sigma_ref ** 2) - 0.5


def sample_gaussian(mu: float, sigma: float, n: int, device: str):
    return torch.normal(mu, sigma, size=(n,), device=device)
