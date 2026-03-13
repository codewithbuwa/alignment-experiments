"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Compare 10% and 50% imbalance for single-Gaussian DPO and KTO.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig
from src.data import make_dpo_pairs, make_kto_samples
from src.distributions import gaussian_pdf
from src.train import train_dpo, train_kto
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def parse_ratio_list(csv_text: str):
    return [float(x.strip()) for x in csv_text.split(",") if x.strip()]


def _shared_policy_eval(policy, cfg: ExperimentConfig, n_samples: int):
    with torch.no_grad():
        y = policy.sample(n_samples)
        zone_min = cfg.target - cfg.zone_half_width
        zone_max = cfg.target + cfg.zone_half_width
        reward = -(y - cfg.target).abs()
    return {
        "mean_reward": reward.mean().item(),
        "reward_std": reward.std(unbiased=False).item(),
        "zone_occupancy": ((y >= zone_min) & (y <= zone_max)).float().mean().item(),
        "sigma_summary": policy.sigma.item(),
    }


def plot_entropy(dpo_results, kto_results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for good_ratio, payload in dpo_results.items():
        label = f"good_ratio={good_ratio:.1f}"
        axes[0].plot(payload["history"]["entropy"], label=label)
    for alpha, payload in kto_results.items():
        label = f"alpha={alpha:.1f}"
        axes[1].plot(payload["history"]["entropy"], label=label)

    axes[0].set_title("DPO Entropy")
    axes[1].set_title("KTO Entropy")
    for ax in axes:
        ax.set_xlabel("Training step")
        ax.set_ylabel("Entropy")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_parameter_dynamics(dpo_results, kto_results, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for good_ratio, payload in dpo_results.items():
        label = f"good_ratio={good_ratio:.1f}"
        axes[0, 0].plot(payload["history"]["mu"], label=label)
        axes[1, 0].plot(payload["history"]["sigma"], label=label)
    for alpha, payload in kto_results.items():
        label = f"alpha={alpha:.1f}"
        axes[0, 1].plot(payload["history"]["mu"], label=label)
        axes[1, 1].plot(payload["history"]["sigma"], label=label)

    axes[0, 0].set_title("DPO Mu")
    axes[1, 0].set_title("DPO Sigma")
    axes[0, 1].set_title("KTO Mu")
    axes[1, 1].set_title("KTO Sigma")
    for ax in axes.flatten():
        ax.set_xlabel("Training step")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0, 0].set_ylabel("Mu")
    axes[1, 0].set_ylabel("Sigma")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_density_grid(dpo_results, kto_results, cfg: ExperimentConfig, out_path):
    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)
    ref_pdf = gaussian_pdf(y_vals, cfg.mu_ref, cfg.sigma_ref)
    dpo_ratios = sorted(dpo_results.keys())
    kto_alphas = sorted(kto_results.keys())

    n_cols = max(len(dpo_ratios), len(kto_alphas))
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), sharex=True, sharey=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    zone_mask = (y_vals >= cfg.target - cfg.zone_half_width) & (y_vals <= cfg.target + cfg.zone_half_width)
    zone_x = y_vals[zone_mask]

    for col in range(n_cols):
        dpo_ax = axes[0, col]
        kto_ax = axes[1, col]

        if col < len(dpo_ratios):
            good_ratio = dpo_ratios[col]
            payload = dpo_results[good_ratio]
            dpo_pdf = gaussian_pdf(y_vals, payload["policy"].mu.item(), payload["policy"].sigma.item())
            dpo_ax.plot(y_vals.numpy(), ref_pdf.numpy(), linestyle="--", color="black", label="Reference")
            dpo_ax.plot(y_vals.numpy(), dpo_pdf.numpy(), color="tab:blue", label="Policy")
            dpo_ax.fill_between(zone_x.numpy(), y1=1, alpha=0.1, label="Desirable Zone")
            dpo_ax.set_title(f"DPO good_ratio={good_ratio:.1f}")
            dpo_ax.grid(True, alpha=0.3)
            dpo_ax.text(
                0.02,
                0.98,
                (
                    f"mu={payload['policy'].mu.item():.2f}\n"
                    f"sigma={payload['policy'].sigma.item():.2f}"
                ),
                transform=dpo_ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
            )
        else:
            dpo_ax.axis("off")

        if col < len(kto_alphas):
            alpha = kto_alphas[col]
            payload = kto_results[alpha]
            kto_pdf = gaussian_pdf(y_vals, payload["policy"].mu.item(), payload["policy"].sigma.item())
            kto_ax.plot(y_vals.numpy(), ref_pdf.numpy(), linestyle="--", color="black", label="Reference")
            kto_ax.plot(y_vals.numpy(), kto_pdf.numpy(), color="tab:blue", label="Policy")
            kto_ax.fill_between(zone_x.numpy(), y1=1, alpha=0.1, label="Desirable Zone")
            kto_ax.set_title(f"KTO alpha={alpha:.1f}")
            kto_ax.grid(True, alpha=0.3)
            kto_ax.text(
                0.02,
                0.98,
                (
                    f"mu={payload['policy'].mu.item():.2f}\n"
                    f"sigma={payload['policy'].sigma.item():.2f}"
                ),
                transform=kto_ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
            )
        else:
            kto_ax.axis("off")

    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density")
    for ax in axes[-1, :]:
        ax.set_xlabel("y")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_shared_evaluation(dpo_results, kto_results, out_path):
    labels = [f"DPO {good_ratio:.1f}" for good_ratio in sorted(dpo_results.keys())]
    labels += [f"KTO {alpha:.1f}" for alpha in sorted(kto_results.keys())]
    metrics = {
        "Mean Reward": [dpo_results[k]["shared_eval"]["mean_reward"] for k in sorted(dpo_results.keys())]
        + [kto_results[k]["shared_eval"]["mean_reward"] for k in sorted(kto_results.keys())],
        "Final Sigma": [dpo_results[k]["shared_eval"]["sigma_summary"] for k in sorted(dpo_results.keys())]
        + [kto_results[k]["shared_eval"]["sigma_summary"] for k in sorted(kto_results.keys())],
    }
    colors = ["tab:blue"] * len(dpo_results) + ["tab:orange"] * len(kto_results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (title, values) in zip(axes, metrics.items()):
        ax.bar(range(len(labels)), values, color=colors)
        ax.set_title(f"Shared Evaluation: {title}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare DPO good_ratio settings and KTO alpha settings for single Gaussians.")
    parser.add_argument("--dpo-good-ratios", type=str, default="0.1,1.0")
    parser.add_argument("--kto-alphas", type=str, default="0.1,0.5")
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--eval-samples", type=int, default=5000)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"imbalance_compare_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    dpo_results = {}
    for good_ratio in parse_ratio_list(args.dpo_good_ratios):
        y_w, y_l = make_dpo_pairs(
            cfg.mu_ref,
            cfg.sigma_ref,
            cfg.target,
            cfg.dataset_size,
            cfg.device,
            good_ratio=good_ratio,
            zone_half_width=cfg.zone_half_width,
        )
        dpo_out = train_dpo(y_w, y_l, cfg)
        dpo_results[good_ratio] = {
            "policy": dpo_out["policy"],
            "history": dpo_out["history"],
            "splits": dpo_out["splits"],
        }

    kto_results = {}
    for alpha in parse_ratio_list(args.kto_alphas):
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
        kto_results[alpha] = {
            "policy": kto_out["policy"],
            "history": kto_out["history"],
            "splits": kto_out["splits"],
        }

    entropy_path = os.path.join(figures_dir, "imbalance_entropy.png")
    density_path = os.path.join(figures_dir, "imbalance_density_grid.png")
    params_path = os.path.join(figures_dir, "imbalance_parameter_dynamics.png")
    generalization_path = os.path.join(figures_dir, "imbalance_shared_eval.png")

    for payload in dpo_results.values():
        payload["shared_eval"] = _shared_policy_eval(payload["policy"], cfg, args.eval_samples)

    for payload in kto_results.values():
        payload["shared_eval"] = _shared_policy_eval(payload["policy"], cfg, args.eval_samples)

    plot_entropy(dpo_results, kto_results, entropy_path)
    plot_density_grid(dpo_results, kto_results, cfg, density_path)
    plot_parameter_dynamics(dpo_results, kto_results, params_path)
    plot_shared_evaluation(dpo_results, kto_results, generalization_path)

    export_report_figure(entropy_path, "single_imbalance_sigma.png")
    export_report_figure(density_path, "single_imbalance_density.png")
    export_report_figure(params_path, "single_imbalance_params.png")
    export_report_figure(generalization_path, "single_imbalance_shared_eval.png")

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "imbalance_summary.json"),
        {
            "dpo": {
                str(good_ratio): {
                    "final_mu": payload["policy"].mu.item(),
                    "final_sigma": payload["policy"].sigma.item(),
                    "splits": payload["splits"],
                    "shared_eval": payload["shared_eval"],
                }
                for good_ratio, payload in dpo_results.items()
            },
            "kto": {
                str(alpha): {
                    "final_mu": payload["policy"].mu.item(),
                    "final_sigma": payload["policy"].sigma.item(),
                    "splits": payload["splits"],
                    "shared_eval": payload["shared_eval"],
                }
                for alpha, payload in kto_results.items()
            },
        },
    )

    update_latest_paths("single_imbalance_compare", output_root)
    print(f"Saved single imbalance comparison results to: {output_root}")


if __name__ == "__main__":
    main()
