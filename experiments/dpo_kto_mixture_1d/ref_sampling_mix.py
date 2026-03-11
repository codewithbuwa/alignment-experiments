"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Compare KL-estimation modes for the Gaussian-mixture KTO setup.
"""

import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import MixtureDPOKTOConfig, ExperimentConfig
from src.montage import make_montage
from src.ref_policies import make_reference_mixture
from src.train_mix import train_kto_mixture
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def _quartile_indices(n_steps: int):
    if n_steps <= 1:
        return [0]
    return [
        0,
        int(0.25 * (n_steps - 1)),
        int(0.5 * (n_steps - 1)),
        int(0.75 * (n_steps - 1)),
        n_steps - 1,
    ]


def _delete_paths(paths):
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def _mixture_pdf(y, mus, sigmas, logits):
    weights = torch.softmax(torch.tensor(logits), dim=0)
    pdf = torch.zeros_like(y)
    for w, m, s in zip(weights, mus, sigmas):
        pdf = pdf + w * (1.0 / (s * torch.sqrt(torch.tensor(2 * torch.pi)))) * \
            torch.exp(-0.5 * ((y - m) / s) ** 2)
    return pdf


def plot_reference_sampling(y_vals, ref_policy, policies, labels, output_path):
    ref_density = torch.exp(ref_policy.log_prob(y_vals)).detach()

    plt.figure(figsize=(10, 6))
    plt.plot(y_vals.numpy(), ref_density.numpy(), "k--", label="Reference policy", linewidth=2)
    for policy, label in zip(policies, labels):
        density = torch.exp(policy.log_prob(y_vals)).detach()
        plt.plot(y_vals.numpy(), density.numpy(), label=label, linewidth=1.5)

    plt.axvspan(5.5, 8.5, alpha=0.1, color="green", label="Desirable Zone")
    plt.xlabel("y")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()


def main():
    cfg = MixtureDPOKTOConfig()
    base_cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_mixture_1d", f"reference_sampling_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)

    ref_policy = make_reference_mixture(cfg.n_components, base_cfg.mu_ref, base_cfg.sigma_ref, cfg.device)

    modes = ["batch", "running", "fixed"]
    policies = []
    sigmas = []
    histories = []

    for mode in modes:
        cfg_mode = MixtureDPOKTOConfig(**cfg.__dict__)
        cfg_mode.kl_mode = mode
        policy, sigma_hist, history, _ = train_kto_mixture(ref_policy, cfg_mode)
        policies.append(policy)
        sigmas.append(sigma_hist)
        histories.append(history)

    plot_reference_sampling(
        y_vals,
        ref_policy,
        policies,
        [f"KTO {m}" for m in modes],
        os.path.join(figures_dir, "reference_sampling_mixture.png"),
    )

    plt.figure(figsize=(10, 6))
    for mode, s in zip(modes, sigmas):
        for i in range(cfg.n_components):
            plt.plot([v[i] for v in s], label=f"{mode}_sigma{i}")
    plt.xlabel("Training Step")
    plt.ylabel("Sigma (per component)")
    plt.title("KL Estimation Mode: Entropy Dynamics")
    plt.legend(ncol=3)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "reference_sampling_mixture_dynamics.png"))
    plt.close()

    quartiles = _quartile_indices(len(sigmas[0]))
    pct = [0, 25, 50, 75, 100]
    quartile_paths = []
    for idx, p in zip(quartiles, pct):
        plt.figure(figsize=(10, 6))
        ref_density = torch.exp(ref_policy.log_prob(y_vals)).detach()
        plt.plot(y_vals.numpy(), ref_density.numpy(), "k--", label="Reference policy", linewidth=2)
        for mode, hist in zip(modes, histories):
            density = _mixture_pdf(y_vals, hist["mus"][idx], hist["sigmas"][idx], hist["logits"][idx])
            plt.plot(y_vals.numpy(), density.numpy(), label=f"KTO {mode} {p}%", linewidth=1.5)
        plt.axvspan(5.5, 8.5, alpha=0.1, color="green", label="Desirable Zone")
        plt.xlabel("y")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f"Mixture KL Modes at {p}% Training")
        q_path = os.path.join(figures_dir, f"reference_sampling_mixture_q{p}.png")
        plt.savefig(q_path)
        plt.close()
        quartile_paths.append(q_path)

    make_montage(quartile_paths, os.path.join(figures_dir, "reference_sampling_mixture_quartiles_montage.png"), cols=3)
    _delete_paths(quartile_paths)

    export_report_figure(
        os.path.join(figures_dir, "reference_sampling_mixture_quartiles_montage.png"),
        "mix_reference_sampling_montage.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "reference_sampling_mixture.png"),
        "mix_reference_sampling.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "reference_sampling_mixture_dynamics.png"),
        "mix_reference_sampling_dynamics.png",
    )

    custom_ref = make_reference_mixture(
        n_components=2,
        mu_ref=base_cfg.mu_ref,
        sigma_ref=base_cfg.sigma_ref,
        device=cfg.device,
        mu_init=torch.tensor([3.0, 9.0]),
    )

    custom_histories = []
    for mode in modes:
        cfg_mode = MixtureDPOKTOConfig(**cfg.__dict__)
        cfg_mode.kl_mode = mode
        _, _, history, _ = train_kto_mixture(custom_ref, cfg_mode)
        custom_histories.append(history)

    custom_paths = []
    for idx, p in zip(quartiles, pct):
        plt.figure(figsize=(10, 6))
        ref_density = torch.exp(custom_ref.log_prob(y_vals)).detach()
        plt.plot(y_vals.numpy(), ref_density.numpy(), "k--", label="Reference policy", linewidth=2)
        for mode, hist in zip(modes, custom_histories):
            density = _mixture_pdf(y_vals, hist["mus"][idx], hist["sigmas"][idx], hist["logits"][idx])
            plt.plot(y_vals.numpy(), density.numpy(), label=f"KTO {mode} {p}%", linewidth=1.5)
        plt.axvspan(5.5, 8.5, alpha=0.1, color="green", label="Desirable Zone")
        plt.xlabel("y")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f"Custom Mixture KL Modes at {p}% Training")
        q_path = os.path.join(figures_dir, f"reference_sampling_mixture_custom_q{p}.png")
        plt.savefig(q_path)
        plt.close()
        custom_paths.append(q_path)

    make_montage(custom_paths, os.path.join(figures_dir, "reference_sampling_mixture_custom_quartiles_montage.png"), cols=3)
    _delete_paths(custom_paths)

    export_report_figure(
        os.path.join(figures_dir, "reference_sampling_mixture_custom_quartiles_montage.png"),
        "mix_reference_sampling_custom_montage.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "reference_sampling_quartiles.json"),
        {
            "quartile_steps": quartiles,
            "histories": histories,
            "custom_histories": custom_histories,
        },
    )

    update_latest_paths("mix_reference_sampling", output_root)

    print(f"Saved mixture reference sampling results to: {output_root}")


if __name__ == "__main__":
    main()
