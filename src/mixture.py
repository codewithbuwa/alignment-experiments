from dataclasses import dataclass
from typing import Dict, List

import torch

from .distributions import gaussian_log_prob


@dataclass
class MixtureConfig:
    seed: int = 123
    n_components: int = 3
    dataset_size: int = 4000
    val_fraction: float = 0.2
    steps: int = 1500
    batch_size: int = 256
    lr: float = 3e-3

    # Target mixture (fixed)
    target_weights: List[float] = None
    target_means: List[float] = None
    target_sigmas: List[float] = None

    # Initialization
    init_means: List[float] = None
    init_sigmas: List[float] = None
    init_logits: List[float] = None

    y_min: float = -5.0
    y_max: float = 15.0

    device: str = "cpu"

    def with_defaults(self):
        if self.target_weights is None:
            self.target_weights = [0.5, 0.3, 0.2]
        if self.target_means is None:
            self.target_means = [3.0, 7.0, 10.0]
        if self.target_sigmas is None:
            self.target_sigmas = [1.0, 0.8, 1.2]
        if self.init_means is None:
            self.init_means = [2.0, 6.0, 9.0]
        if self.init_sigmas is None:
            self.init_sigmas = [1.5, 1.5, 1.5]
        if self.init_logits is None:
            self.init_logits = [0.0, 0.0, 0.0]
        return self


class GaussianMixture(torch.nn.Module):
    def __init__(self, means, sigmas, logits, device: str):
        super().__init__()
        self.means = torch.nn.Parameter(torch.tensor(means, device=device))
        self.log_sigmas = torch.nn.Parameter(torch.log(torch.tensor(sigmas, device=device)))
        self.logits = torch.nn.Parameter(torch.tensor(logits, device=device))

    def weights(self):
        return torch.softmax(self.logits, dim=0)

    def sigmas(self):
        return torch.exp(self.log_sigmas)

    def log_prob(self, y: torch.Tensor):
        y = y.unsqueeze(1)
        logps = gaussian_log_prob(y, self.means, self.sigmas())
        logw = torch.log(self.weights())
        return torch.logsumexp(logps + logw, dim=1)


def sample_mixture(weights, means, sigmas, n: int, device: str):
    weights_t = torch.tensor(weights, device=device)
    means_t = torch.tensor(means, device=device)
    sigmas_t = torch.tensor(sigmas, device=device)

    comp = torch.multinomial(weights_t, num_samples=n, replacement=True)
    eps = torch.randn(n, device=device)
    y = means_t[comp] + sigmas_t[comp] * eps
    return y


def fit_mixture_mle(y_train, y_val, cfg: MixtureConfig) -> Dict[str, List[float]]:
    model = GaussianMixture(cfg.init_means, cfg.init_sigmas, cfg.init_logits, cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n = y_train.numel()
    history = {
        "train_nll": [],
        "val_nll": [],
        "weights": [],
        "means": [],
        "sigmas": [],
    }

    for _ in range(cfg.steps):
        idx = torch.randint(0, n, (cfg.batch_size,), device=cfg.device)
        batch = y_train[idx]

        nll = -model.log_prob(batch).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()

        with torch.no_grad():
            train_nll = -model.log_prob(y_train).mean().item()
            val_nll = -model.log_prob(y_val).mean().item()

            history["train_nll"].append(train_nll)
            history["val_nll"].append(val_nll)
            history["weights"].append(model.weights().detach().cpu().tolist())
            history["means"].append(model.means.detach().cpu().tolist())
            history["sigmas"].append(model.sigmas().detach().cpu().tolist())

    return {"model": model, "history": history}
