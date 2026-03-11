from dataclasses import asdict
from typing import Dict, List

import torch

from .config import ExperimentConfig
from .distributions import gaussian_entropy, gaussian_log_prob, kl_gaussian
from .losses import dpo_loss, kto_loss
from .policies import GaussianPolicy
from .utils import train_val_split


class KLTracker:
    def __init__(self, mode: str, fixed_value: float, ema_decay: float):
        self.mode = mode
        self.fixed_value = fixed_value
        self.ema_decay = ema_decay
        self.ema_value = None

    def update(self, kl_value: torch.Tensor) -> torch.Tensor:
        if self.mode == "fixed":
            return torch.tensor(self.fixed_value, device=kl_value.device)
        if self.mode == "running":
            kl_scalar = kl_value.detach()
            if self.ema_value is None:
                self.ema_value = kl_scalar
            else:
                self.ema_value = self.ema_decay * self.ema_value + (1.0 - self.ema_decay) * kl_scalar
            return self.ema_value
        return kl_value


def _batch_kl_estimate(y: torch.Tensor, mu, sigma, mu_ref: float, sigma_ref: float) -> torch.Tensor:
    logp = gaussian_log_prob(y, mu, sigma)
    logp_ref = gaussian_log_prob(y, y.new_tensor(mu_ref), y.new_tensor(sigma_ref))
    return torch.mean(logp - logp_ref)


def _policy_init(cfg: ExperimentConfig):
    mu_init = cfg.mu_ref if cfg.init_mu is None else cfg.init_mu
    sigma_init = cfg.sigma_ref if cfg.init_sigma is None else cfg.init_sigma
    return mu_init, sigma_init


def _split_tensors(cfg: ExperimentConfig, *tensors):
    n = tensors[0].numel()
    if cfg.eval_fraction <= 0.0:
        return tensors, None
    train_idx, eval_idx = train_val_split(n, cfg.eval_fraction, cfg.seed)
    train_tensors = tuple(t[train_idx.to(t.device)] for t in tensors)
    eval_tensors = tuple(t[eval_idx.to(t.device)] for t in tensors)
    return train_tensors, eval_tensors


def train_dpo(y_w: torch.Tensor, y_l: torch.Tensor, cfg: ExperimentConfig) -> Dict[str, List[float]]:
    mu_init, sigma_init = _policy_init(cfg)
    policy = GaussianPolicy(mu_init, sigma_init, cfg.device)
    opt = torch.optim.Adam([policy.mu, policy.rho], lr=cfg.lr)

    (y_w_train, y_l_train), eval_split = _split_tensors(cfg, y_w, y_l)
    n = y_w_train.numel()
    history = {"mu": [], "sigma": [], "entropy": [], "loss": [], "eval_loss": []}

    for _ in range(cfg.steps):
        idx = torch.randint(0, n, (cfg.batch_size,), device=cfg.device)
        batch_w = y_w_train[idx]
        batch_l = y_l_train[idx]

        loss = dpo_loss(batch_w, batch_l, policy.mu, policy.rho, cfg.mu_ref, cfg.sigma_ref, cfg.beta)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            if eval_split is not None:
                eval_loss = dpo_loss(
                    eval_split[0],
                    eval_split[1],
                    policy.mu,
                    policy.rho,
                    cfg.mu_ref,
                    cfg.sigma_ref,
                    cfg.beta,
                ).item()
            else:
                eval_loss = loss.item()
            history["mu"].append(policy.mu.item())
            history["sigma"].append(policy.sigma.item())
            history["entropy"].append(gaussian_entropy(policy.sigma).item())
            history["loss"].append(loss.item())
            history["eval_loss"].append(eval_loss)

    return {
        "policy": policy,
        "history": history,
        "splits": {
            "train_size": int(y_w_train.numel()),
            "eval_size": int(0 if eval_split is None else eval_split[0].numel()),
        },
    }


def train_kto(y: torch.Tensor, labels: torch.Tensor, cfg: ExperimentConfig) -> Dict[str, List[float]]:
    mu_init, sigma_init = _policy_init(cfg)
    policy = GaussianPolicy(mu_init, sigma_init, cfg.device)
    opt = torch.optim.Adam([policy.mu, policy.rho], lr=cfg.lr)

    (y_train, labels_train), eval_split = _split_tensors(cfg, y, labels)
    n = y_train.numel()
    history = {"mu": [], "sigma": [], "entropy": [], "loss": [], "kl": [], "eval_loss": []}

    tracker = KLTracker(cfg.kl_mode, cfg.kl_fixed, cfg.kl_ema_decay)

    for _ in range(cfg.steps):
        idx = torch.randint(0, n, (cfg.batch_size,), device=cfg.device)
        batch_y = y_train[idx]
        batch_labels = labels_train[idx]

        if cfg.kl_mode == "batch":
            kl_value = _batch_kl_estimate(batch_y, policy.mu, policy.sigma, cfg.mu_ref, cfg.sigma_ref)
        else:
            kl_value = kl_gaussian(policy.mu, policy.sigma, cfg.mu_ref, cfg.sigma_ref)

        if not cfg.kl_grad:
            kl_value = kl_value.detach()

        kl_value = tracker.update(kl_value)

        loss = kto_loss(
            batch_y,
            batch_labels,
            policy.mu,
            policy.rho,
            cfg.mu_ref,
            cfg.sigma_ref,
            cfg.kto_gamma,
            kl_value,
            cfg.beta,
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            if eval_split is not None:
                eval_y, eval_labels = eval_split
                if cfg.kl_mode == "batch":
                    eval_kl = _batch_kl_estimate(eval_y, policy.mu, policy.sigma, cfg.mu_ref, cfg.sigma_ref).detach()
                else:
                    eval_kl = kl_gaussian(policy.mu, policy.sigma, cfg.mu_ref, cfg.sigma_ref).detach()
                eval_loss = kto_loss(
                    eval_y,
                    eval_labels,
                    policy.mu,
                    policy.rho,
                    cfg.mu_ref,
                    cfg.sigma_ref,
                    cfg.kto_gamma,
                    eval_kl,
                    cfg.beta,
                ).item()
            else:
                eval_loss = loss.item()
            history["mu"].append(policy.mu.item())
            history["sigma"].append(policy.sigma.item())
            history["entropy"].append(gaussian_entropy(policy.sigma).item())
            history["loss"].append(loss.item())
            history["kl"].append(kl_value.item())
            history["eval_loss"].append(eval_loss)

    return {
        "policy": policy,
        "history": history,
        "splits": {
            "train_size": int(y_train.numel()),
            "eval_size": int(0 if eval_split is None else eval_split[0].numel()),
        },
    }


def config_to_dict(cfg: ExperimentConfig) -> dict:
    return asdict(cfg)
