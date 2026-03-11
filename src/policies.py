import math
import torch
import torch.nn as nn


class GaussianPolicy(nn.Module):
    def __init__(self, mu_init: float, sigma_init: float, device: str):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu_init, device=device))
        self.rho = nn.Parameter(torch.tensor(math.log(sigma_init), device=device))

    @property
    def sigma(self):
        return torch.exp(self.rho)

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma
        return -0.5 * (((y - self.mu) / sigma) ** 2 + 2 * torch.log(sigma) + math.log(2 * math.pi))

    def sample(self, n: int) -> torch.Tensor:
        return self.mu + self.sigma * torch.randn(n, device=self.mu.device)

    def kl_to_ref(self, mu_ref: float, sigma_ref: float) -> torch.Tensor:
        return torch.log(torch.tensor(sigma_ref, device=self.mu.device) / self.sigma) + \
            (self.sigma ** 2 + (self.mu - mu_ref) ** 2) / (2 * sigma_ref ** 2) - 0.5


class GaussianMixturePolicy(nn.Module):
    def __init__(
        self,
        n_components: int,
        mu_init=None,
        log_sigma_init=None,
        logits_init=None,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_components = n_components
        if mu_init is None:
            mu_init = torch.linspace(3.0, 8.0, n_components)
        if log_sigma_init is None:
            log_sigma_init = torch.zeros(n_components)
        if logits_init is None:
            logits_init = torch.zeros(n_components)

        self.mus = nn.Parameter(mu_init.clone().detach().float().to(device))
        self.log_sigmas = nn.Parameter(log_sigma_init.clone().detach().float().to(device))
        self.logits = nn.Parameter(logits_init.clone().detach().float().to(device))

    def sigmas(self):
        return torch.exp(self.log_sigmas)

    def probs(self):
        return torch.softmax(self.logits, dim=-1)

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        y = y.unsqueeze(-1)
        mus = self.mus.unsqueeze(0)
        sigmas = self.sigmas().unsqueeze(0)
        log_probs = -0.5 * (((y - mus) / sigmas) ** 2 + 2 * self.log_sigmas + math.log(2 * math.pi))
        log_weights = torch.log_softmax(self.logits, dim=-1).unsqueeze(0)
        return torch.logsumexp(log_probs + log_weights, dim=-1)

    def sample(self, n: int) -> torch.Tensor:
        with torch.no_grad():
            probs = self.probs()
            comp = torch.multinomial(probs, n, replacement=True)
            mus_comp = self.mus[comp]
            sigmas_comp = self.sigmas()[comp]
            return mus_comp + sigmas_comp * torch.randn(n, device=self.mus.device)

    def kl_to_ref(self, ref_policy, n_samples: int = 1000) -> torch.Tensor:
        with torch.no_grad():
            y = self.sample(n_samples)
            log_p = self.log_prob(y)
            log_ref = ref_policy.log_prob(y)
            return torch.mean(log_p - log_ref)
