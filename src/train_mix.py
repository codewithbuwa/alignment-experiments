import torch

from .config import MixtureDPOKTOConfig
from .data_mix import make_mixture_dpo_pairs, make_mixture_kto_samples
from .policies import GaussianMixturePolicy
from .utils import set_seed, train_val_split


def _mixture_entropy(policy: GaussianMixturePolicy, n_samples: int = 512) -> torch.Tensor:
    with torch.no_grad():
        y = policy.sample(n_samples)
        return -policy.log_prob(y).mean()


def _init_mix(cfg: MixtureDPOKTOConfig):
    mu_init = None
    log_sigma_init = None
    logits_init = None

    if cfg.init_means is not None:
        mu_init = torch.tensor(cfg.init_means)
    if cfg.init_sigmas is not None:
        log_sigma_init = torch.log(torch.tensor(cfg.init_sigmas))
    if cfg.init_logits is not None:
        logits_init = torch.tensor(cfg.init_logits)

    return mu_init, log_sigma_init, logits_init


def _split_tensors(cfg: MixtureDPOKTOConfig, *tensors):
    n = tensors[0].numel()
    if cfg.eval_fraction <= 0.0:
        return tensors, None
    train_idx, eval_idx = train_val_split(n, cfg.eval_fraction, cfg.seed)
    train_tensors = tuple(t[train_idx.to(t.device)] for t in tensors)
    eval_tensors = tuple(t[eval_idx.to(t.device)] for t in tensors)
    return train_tensors, eval_tensors


def train_dpo_mixture(ref_policy, cfg: MixtureDPOKTOConfig, good_ratio: float = None):
    set_seed(cfg.seed)
    mu_init, log_sigma_init, logits_init = _init_mix(cfg)
    policy = GaussianMixturePolicy(
        n_components=cfg.n_components,
        mu_init=mu_init,
        log_sigma_init=log_sigma_init,
        logits_init=logits_init,
        device=cfg.device,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    y_w, y_l = make_mixture_dpo_pairs(
        ref_policy,
        target=cfg.target,
        n=cfg.dataset_size,
        device=cfg.device,
        good_ratio=good_ratio,
        zone_half_width=cfg.zone_half_width,
    )
    (y_w_train, y_l_train), eval_split = _split_tensors(cfg, y_w, y_l)

    history = {"mus": [], "sigmas": [], "logits": [], "loss": [], "eval_loss": [], "entropy": []}
    sigmas = []

    for _ in range(cfg.steps):
        optimizer.zero_grad()
        h_w = cfg.beta * (policy.log_prob(y_w_train) - ref_policy.log_prob(y_w_train))
        h_l = cfg.beta * (policy.log_prob(y_l_train) - ref_policy.log_prob(y_l_train))
        loss = -torch.mean(torch.log(torch.sigmoid(h_w - h_l)))
        loss.backward()
        optimizer.step()

        step_sigmas = policy.sigmas().detach().cpu().tolist()
        sigmas.append(step_sigmas)
        history["mus"].append(policy.mus.detach().cpu().tolist())
        history["sigmas"].append(step_sigmas)
        history["logits"].append(policy.logits.detach().cpu().tolist())
        history["loss"].append(loss.item())
        history["entropy"].append(_mixture_entropy(policy).item())
        if eval_split is not None:
            with torch.no_grad():
                eval_w, eval_l = eval_split
                eval_h_w = cfg.beta * (policy.log_prob(eval_w) - ref_policy.log_prob(eval_w))
                eval_h_l = cfg.beta * (policy.log_prob(eval_l) - ref_policy.log_prob(eval_l))
                eval_loss = -torch.mean(torch.log(torch.sigmoid(eval_h_w - eval_h_l))).item()
        else:
            eval_loss = loss.item()
        history["eval_loss"].append(eval_loss)

    return policy, sigmas, history, {
        "train_size": int(y_w_train.numel()),
        "eval_size": int(0 if eval_split is None else eval_split[0].numel()),
    }


def train_kto_mixture(ref_policy, cfg: MixtureDPOKTOConfig, good_ratio: float = None):
    set_seed(cfg.seed)
    mu_init, log_sigma_init, logits_init = _init_mix(cfg)
    policy = GaussianMixturePolicy(
        n_components=cfg.n_components,
        mu_init=mu_init,
        log_sigma_init=log_sigma_init,
        logits_init=logits_init,
        device=cfg.device,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    y_fixed, labels_fixed = make_mixture_kto_samples(
        ref_policy,
        target=cfg.target,
        zone_half_width=cfg.zone_half_width,
        n=cfg.dataset_size,
        device=cfg.device,
        delta=cfg.delta,
        good_ratio=good_ratio,
    )
    (y_train, labels_train), eval_split = _split_tensors(cfg, y_fixed, labels_fixed)

    history = {"mus": [], "sigmas": [], "logits": [], "loss": [], "eval_loss": [], "kl": [], "entropy": []}
    sigmas = []
    running_kl = torch.tensor(0.0, device=cfg.device)

    for _ in range(cfg.steps):
        optimizer.zero_grad()

        h = cfg.beta * (policy.log_prob(y_train) - ref_policy.log_prob(y_train))

        if cfg.kl_mode == "batch":
            kl = policy.kl_to_ref(ref_policy, n_samples=cfg.dataset_size)
        elif cfg.kl_mode == "running":
            with torch.no_grad():
                kl_batch = policy.kl_to_ref(ref_policy, n_samples=cfg.dataset_size)
                running_kl = (1 - cfg.kl_ema_decay) * running_kl + cfg.kl_ema_decay * kl_batch
                kl = running_kl
        elif cfg.kl_mode == "fixed":
            kl = torch.tensor(cfg.kl_fixed, device=cfg.device)
        else:
            raise ValueError("Unknown kl_mode")

        z = h - kl
        v = torch.zeros_like(z)
        v[labels_train == 1.0] = torch.sigmoid(z[labels_train == 1.0])
        v[labels_train == 0.0] = torch.sigmoid(-cfg.kto_gamma * z[labels_train == 0.0])

        loss = torch.mean(1 - v)
        loss.backward()
        optimizer.step()

        step_sigmas = policy.sigmas().detach().cpu().tolist()
        sigmas.append(step_sigmas)
        history["mus"].append(policy.mus.detach().cpu().tolist())
        history["sigmas"].append(step_sigmas)
        history["logits"].append(policy.logits.detach().cpu().tolist())
        history["loss"].append(loss.item())
        history["kl"].append(kl.item() if hasattr(kl, "item") else float(kl))
        history["entropy"].append(_mixture_entropy(policy).item())
        if eval_split is not None:
            with torch.no_grad():
                eval_y, eval_labels = eval_split
                eval_h = cfg.beta * (policy.log_prob(eval_y) - ref_policy.log_prob(eval_y))
                if cfg.kl_mode == "batch":
                    eval_kl = policy.kl_to_ref(ref_policy, n_samples=cfg.dataset_size)
                elif cfg.kl_mode == "running":
                    eval_kl = running_kl
                else:
                    eval_kl = torch.tensor(cfg.kl_fixed, device=cfg.device)
                eval_z = eval_h - eval_kl
                eval_v = torch.zeros_like(eval_z)
                eval_v[eval_labels == 1.0] = torch.sigmoid(eval_z[eval_labels == 1.0])
                eval_v[eval_labels == 0.0] = torch.sigmoid(-cfg.kto_gamma * eval_z[eval_labels == 0.0])
                eval_loss = torch.mean(1 - eval_v).item()
        else:
            eval_loss = loss.item()
        history["eval_loss"].append(eval_loss)

    return policy, sigmas, history, {
        "train_size": int(y_train.numel()),
        "eval_size": int(0 if eval_split is None else eval_split[0].numel()),
    }
