"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Compare single and mixture policies under the same supervision-ratio sweep.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig, MixtureDPOKTOConfig
from src.data import make_dpo_pairs, make_kto_samples
from src.ref_policies import make_reference_mixture
from src.train import train_dpo, train_kto
from src.train_mix import train_dpo_mixture, train_kto_mixture
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def parse_alpha_list(alpha_csv: str):
    return [float(x.strip()) for x in alpha_csv.split(",") if x.strip()]


def _effective_good_mass_dpo(y_w: torch.Tensor, cfg: ExperimentConfig) -> float:
    zone_min = cfg.target - cfg.zone_half_width
    zone_max = cfg.target + cfg.zone_half_width
    return ((y_w >= zone_min) & (y_w <= zone_max)).float().mean().item()


def _effective_good_mass_kto(labels: torch.Tensor) -> float:
    return labels.float().mean().item()


def plot_summary(results, out_path, n_components):
    x = results["alpha"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(x, results["dpo_single_mu"], marker="o", color="black", label="Single mu")
    for k in range(n_components):
        axes[0, 0].plot(x, results[f"dpo_mix_mu_c{k}"], marker="o", label=f"Mix C{k} mu")
    axes[0, 0].set_title("DPO: Final Mu by Component")
    axes[0, 0].set_xlabel("DPO good_ratio / KTO alpha")
    axes[0, 0].set_ylabel("Mu")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(x, results["kto_single_mu"], marker="o", color="black", label="Single mu")
    for k in range(n_components):
        axes[0, 1].plot(x, results[f"kto_mix_mu_c{k}"], marker="o", label=f"Mix C{k} mu")
    axes[0, 1].set_title("KTO: Final Mu by Component")
    axes[0, 1].set_xlabel("DPO good_ratio / KTO alpha")
    axes[0, 1].set_ylabel("Mu")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(x, results["dpo_single_sigma"], marker="o", color="black", label="Single sigma")
    for k in range(n_components):
        axes[1, 0].plot(x, results[f"dpo_mix_sigma_c{k}"], marker="o", label=f"Mix C{k} sigma")
    axes[1, 0].set_title("DPO: Final Sigma by Component")
    axes[1, 0].set_xlabel("DPO good_ratio / KTO alpha")
    axes[1, 0].set_ylabel("Sigma")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(x, results["kto_single_sigma"], marker="o", color="black", label="Single sigma")
    for k in range(n_components):
        axes[1, 1].plot(x, results[f"kto_mix_sigma_c{k}"], marker="o", label=f"Mix C{k} sigma")
    axes[1, 1].set_title("KTO: Final Sigma by Component")
    axes[1, 1].set_xlabel("DPO good_ratio / KTO alpha")
    axes[1, 1].set_ylabel("Sigma")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_training_dynamics(tracked, out_path, n_components):
    if tracked is None:
        return

    steps = len(tracked["dpo_single"]["mu"])
    x = torch.arange(steps)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

    for k in range(n_components):
        axes[0, 0].plot(x, tracked["kto_mix"]["mus"][:, k], label=f"C{k}")
        axes[1, 0].plot(x, tracked["kto_mix"]["sigmas"][:, k], label=f"C{k}")
        axes[2, 0].plot(x, tracked["kto_mix"]["weights"][:, k], label=f"C{k}")

        axes[0, 1].plot(x, tracked["dpo_mix"]["mus"][:, k], label=f"C{k}")
        axes[1, 1].plot(x, tracked["dpo_mix"]["sigmas"][:, k], label=f"C{k}")
        axes[2, 1].plot(x, tracked["dpo_mix"]["weights"][:, k], label=f"C{k}")

    axes[0, 0].set_title(f"KTO Mixture (alpha={tracked['alpha']:.2f})")
    axes[0, 1].set_title(f"DPO Mixture (good_ratio={tracked['alpha']:.2f})")

    axes[0, 0].set_ylabel("Mu")
    axes[1, 0].set_ylabel("Sigma")
    axes[2, 0].set_ylabel("Weight")
    axes[0, 1].set_ylabel("Mu")
    axes[1, 1].set_ylabel("Sigma")
    axes[2, 1].set_ylabel("Weight")

    for r in range(3):
        for c in range(2):
            axes[r, c].grid(True, alpha=0.3)
            axes[r, c].legend()

    axes[2, 0].set_xlabel("Training step")
    axes[2, 1].set_xlabel("Training step")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Robustness: single vs mixture for DPO/KTO")
    parser.add_argument("--alphas", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--delta", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--kto-mode", choices=["analytic", "batch", "running", "fixed"], default="batch")
    parser.add_argument("--track-alpha", type=float, default=0.5)
    parser.add_argument("--summary-out", type=str, default=None)
    parser.add_argument("--dynamics-out", type=str, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    mix_cfg = MixtureDPOKTOConfig()
    if args.steps is not None:
        cfg.steps = args.steps
        mix_cfg.steps = args.steps
    if args.lr is not None:
        cfg.lr = args.lr
        mix_cfg.lr = args.lr
    mix_cfg.n_components = args.n_components
    mix_cfg.delta = args.delta
    mix_cfg.kl_mode = args.kto_mode if args.kto_mode != "analytic" else "batch"

    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"robustness_single_vs_mixture_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    summary_out = args.summary_out or os.path.join(figures_dir, "robustness_params_single_vs_mixture.png")
    dynamics_out = args.dynamics_out or os.path.join(figures_dir, "robustness_training_dynamics_single_vs_mixture.png")

    alphas = parse_alpha_list(args.alphas)

    results = {
        "alpha": [],
        "dpo_single_mu": [],
        "dpo_single_sigma": [],
        "dpo_single_eff": [],
        "kto_single_mu": [],
        "kto_single_sigma": [],
        "kto_single_eff": [],
    }
    for k in range(args.n_components):
        results[f"dpo_mix_mu_c{k}"] = []
        results[f"dpo_mix_sigma_c{k}"] = []
        results[f"dpo_mix_weight_c{k}"] = []
        results[f"kto_mix_mu_c{k}"] = []
        results[f"kto_mix_sigma_c{k}"] = []
        results[f"kto_mix_weight_c{k}"] = []

    tracked = None

    for alpha in alphas:
        y_w, y_l = make_dpo_pairs(
            cfg.mu_ref,
            cfg.sigma_ref,
            cfg.target,
            cfg.dataset_size,
            cfg.device,
            good_ratio=alpha,
            zone_half_width=cfg.zone_half_width,
        )
        dpo_out = train_dpo(y_w, y_l, cfg)
        dpo_eff = _effective_good_mass_dpo(y_w, cfg)

        y, labels = make_kto_samples(
            cfg.mu_ref,
            cfg.sigma_ref,
            cfg.target,
            cfg.zone_half_width,
            cfg.dataset_size,
            cfg.kto_good_fraction,
            cfg.device,
            delta=args.delta,
            good_ratio=alpha,
        )
        kto_out = train_kto(y, labels, cfg)
        kto_eff = _effective_good_mass_kto(labels)

        ref_policy = make_reference_mixture(mix_cfg.n_components, cfg.mu_ref, cfg.sigma_ref, cfg.device)
        dpo_mix, _, dpo_mix_hist, _ = train_dpo_mixture(ref_policy, mix_cfg, good_ratio=alpha)
        kto_mix, _, kto_mix_hist, _ = train_kto_mixture(ref_policy, mix_cfg, good_ratio=alpha)

        results["alpha"].append(alpha)
        results["dpo_single_mu"].append(dpo_out["policy"].mu.item())
        results["dpo_single_sigma"].append(dpo_out["policy"].sigma.item())
        results["dpo_single_eff"].append(dpo_eff)
        results["kto_single_mu"].append(kto_out["policy"].mu.item())
        results["kto_single_sigma"].append(kto_out["policy"].sigma.item())
        results["kto_single_eff"].append(kto_eff)

        dpo_mus = dpo_mix.mus.detach().cpu().tolist()
        dpo_sigmas = dpo_mix.sigmas().detach().cpu().tolist()
        dpo_weights = torch.softmax(dpo_mix.logits.detach().cpu(), dim=0).tolist()
        kto_mus = kto_mix.mus.detach().cpu().tolist()
        kto_sigmas = kto_mix.sigmas().detach().cpu().tolist()
        kto_weights = torch.softmax(kto_mix.logits.detach().cpu(), dim=0).tolist()

        for k in range(args.n_components):
            results[f"dpo_mix_mu_c{k}"].append(dpo_mus[k])
            results[f"dpo_mix_sigma_c{k}"].append(dpo_sigmas[k])
            results[f"dpo_mix_weight_c{k}"].append(dpo_weights[k])
            results[f"kto_mix_mu_c{k}"].append(kto_mus[k])
            results[f"kto_mix_sigma_c{k}"].append(kto_sigmas[k])
            results[f"kto_mix_weight_c{k}"].append(kto_weights[k])

        if tracked is None and abs(alpha - args.track_alpha) < 1e-9:
            tracked = {
                "alpha": alpha,
                "dpo_single": dpo_out["history"],
                "kto_single": kto_out["history"],
                "dpo_mix": {
                    "mus": torch.tensor(dpo_mix_hist["mus"]),
                    "sigmas": torch.tensor(dpo_mix_hist["sigmas"]),
                    "weights": torch.softmax(torch.tensor(dpo_mix_hist["logits"]), dim=1),
                },
                "kto_mix": {
                    "mus": torch.tensor(kto_mix_hist["mus"]),
                    "sigmas": torch.tensor(kto_mix_hist["sigmas"]),
                    "weights": torch.softmax(torch.tensor(kto_mix_hist["logits"]), dim=1),
                },
            }

    plot_summary(results, summary_out, args.n_components)
    plot_training_dynamics(tracked, dynamics_out, args.n_components)

    export_report_figure(
        summary_out,
        "robustness_params_single_vs_mixture.png",
    )
    export_report_figure(
        dynamics_out,
        "robustness_training_dynamics_single_vs_mixture.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(os.path.join(runs_dir, "robustness_summary.json"), results)

    update_latest_paths("robustness_single_vs_mixture", output_root)

    print(f"Saved robustness results to: {output_root}")


if __name__ == "__main__":
    main()
