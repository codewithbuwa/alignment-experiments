"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Compare left/right/around initialization under imbalanced DPO and KTO supervision.
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


def _init_scenarios(cfg: ExperimentConfig):
    return {
        "around": cfg.target,
        "left": cfg.target - 4.0,
        "right": cfg.target + 4.0,
    }


def _plot_grid(results, row_values, row_label, title_prefix, cfg: ExperimentConfig, out_path: str):
    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)
    ref_pdf = gaussian_pdf(y_vals, cfg.mu_ref, cfg.sigma_ref)
    inits = list(_init_scenarios(cfg).keys())

    fig, axes = plt.subplots(len(row_values), len(inits), figsize=(5 * len(inits), 4 * len(row_values)), sharex=True, sharey=True)
    if len(row_values) == 1:
        axes = axes.reshape(1, -1)

    zone_mask = (y_vals >= cfg.target - cfg.zone_half_width) & (y_vals <= cfg.target + cfg.zone_half_width)
    zone_x = y_vals[zone_mask]

    for i, row_value in enumerate(row_values):
        for j, init_name in enumerate(inits):
            ax = axes[i, j]
            payload = results[row_value][init_name]
            policy = payload["policy"]
            pdf = gaussian_pdf(y_vals, policy.mu.item(), policy.sigma.item())
            ax.plot(y_vals.numpy(), ref_pdf.numpy(), linestyle="--", color="black", label="Reference")
            ax.plot(y_vals.numpy(), pdf.numpy(), color="tab:blue", label="Policy")
            ax.fill_between(zone_x.numpy(), y1=1, alpha=0.1, label="Desirable Zone")
            ax.set_title(f"{row_label}={row_value:.1f}, init={init_name}")
            ax.grid(True, alpha=0.3)
            ax.text(
                0.02,
                0.98,
                (
                    f"init_mu={payload['init_mu']:.2f}\n"
                    f"final_mu={policy.mu.item():.2f}\n"
                    f"final_sigma={policy.sigma.item():.2f}"
                ),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
            )

    axes[0, 0].legend(loc="upper right")
    for ax in axes[-1, :]:
        ax.set_xlabel("y")
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")

    fig.suptitle(title_prefix, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_parameter_summary(dpo_results, kto_results, dpo_good_ratios, kto_alphas, cfg: ExperimentConfig, out_path: str):
    init_names = list(_init_scenarios(cfg).keys())
    x = range(len(init_names))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for good_ratio in dpo_good_ratios:
        mus = [dpo_results[good_ratio][name]["policy"].mu.item() for name in init_names]
        sigmas = [dpo_results[good_ratio][name]["policy"].sigma.item() for name in init_names]
        axes[0, 0].plot(x, mus, marker="o", label=f"good_ratio={good_ratio:.1f}")
        axes[1, 0].plot(x, sigmas, marker="o", label=f"good_ratio={good_ratio:.1f}")

    for alpha in kto_alphas:
        mus = [kto_results[alpha][name]["policy"].mu.item() for name in init_names]
        sigmas = [kto_results[alpha][name]["policy"].sigma.item() for name in init_names]
        axes[0, 1].plot(x, mus, marker="o", label=f"alpha={alpha:.1f}")
        axes[1, 1].plot(x, sigmas, marker="o", label=f"alpha={alpha:.1f}")

    axes[0, 0].set_title("DPO Final Mu by Initialization")
    axes[1, 0].set_title("DPO Final Sigma by Initialization")
    axes[0, 1].set_title("KTO Final Mu by Initialization")
    axes[1, 1].set_title("KTO Final Sigma by Initialization")

    for ax in axes[1, :]:
        ax.set_xticks(list(x))
        ax.set_xticklabels(init_names)
    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0, 0].set_ylabel("Mu")
    axes[1, 0].set_ylabel("Sigma")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare initialization sensitivity under imbalanced DPO and KTO supervision.")
    parser.add_argument("--dpo-good-ratios", type=str, default="1.0,0.1")
    parser.add_argument("--kto-alphas", type=str, default="0.5,0.1")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"init_imbalance_compare_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    dpo_good_ratios = parse_ratio_list(args.dpo_good_ratios)
    kto_alphas = parse_ratio_list(args.kto_alphas)
    inits = _init_scenarios(cfg)

    dpo_results = {good_ratio: {} for good_ratio in dpo_good_ratios}
    kto_results = {alpha: {} for alpha in kto_alphas}

    # Train every init/imbalance combination independently so we can answer
    # whether the learned policy still moves toward the target from both sides.
    for good_ratio in dpo_good_ratios:
        y_w, y_l = make_dpo_pairs(
            cfg.mu_ref,
            cfg.sigma_ref,
            cfg.target,
            cfg.dataset_size,
            cfg.device,
            good_ratio=good_ratio,
            zone_half_width=cfg.zone_half_width,
        )
        for init_name, init_mu in inits.items():
            cfg_init = ExperimentConfig(**cfg.__dict__)
            cfg_init.init_mu = init_mu
            cfg_init.init_sigma = cfg.sigma_ref
            dpo_out = train_dpo(y_w, y_l, cfg_init)
            dpo_results[good_ratio][init_name] = {"policy": dpo_out["policy"], "history": dpo_out["history"], "init_mu": init_mu}

    for alpha in kto_alphas:
        y, labels = make_kto_samples(
            cfg.mu_ref,
            cfg.sigma_ref,
            cfg.target,
            cfg.zone_half_width,
            cfg.dataset_size,
            cfg.kto_good_fraction,
            cfg.device,
            delta=1.0,
            good_ratio=alpha,
        )
        for init_name, init_mu in inits.items():
            cfg_init = ExperimentConfig(**cfg.__dict__)
            cfg_init.init_mu = init_mu
            cfg_init.init_sigma = cfg.sigma_ref
            kto_out = train_kto(y, labels, cfg_init)
            kto_results[alpha][init_name] = {"policy": kto_out["policy"], "history": kto_out["history"], "init_mu": init_mu}

    dpo_grid_path = os.path.join(figures_dir, "dpo_init_imbalance_grid.png")
    kto_grid_path = os.path.join(figures_dir, "kto_init_imbalance_grid.png")
    summary_path = os.path.join(figures_dir, "init_imbalance_parameter_summary.png")

    _plot_grid(dpo_results, dpo_good_ratios, "good_ratio", "DPO Under Initialization + Imbalance", cfg, dpo_grid_path)
    _plot_grid(kto_results, kto_alphas, "alpha", "KTO Under Initialization + Imbalance", cfg, kto_grid_path)
    _plot_parameter_summary(dpo_results, kto_results, dpo_good_ratios, kto_alphas, cfg, summary_path)

    export_report_figure(dpo_grid_path, "single_dpo_init_imbalance_grid.png")
    export_report_figure(kto_grid_path, "single_kto_init_imbalance_grid.png")
    export_report_figure(summary_path, "single_init_imbalance_parameter_summary.png")

    save_json(
        os.path.join(runs_dir, "init_imbalance_summary.json"),
        {
            "dpo": {
                str(good_ratio): {
                    init_name: {
                        "init_mu": payload["init_mu"],
                        "final_mu": payload["policy"].mu.item(),
                        "final_sigma": payload["policy"].sigma.item(),
                    }
                    for init_name, payload in init_payloads.items()
                }
                for good_ratio, init_payloads in dpo_results.items()
            },
            "kto": {
                str(alpha): {
                    init_name: {
                        "init_mu": payload["init_mu"],
                        "final_mu": payload["policy"].mu.item(),
                        "final_sigma": payload["policy"].sigma.item(),
                    }
                    for init_name, payload in init_payloads.items()
                }
                for alpha, init_payloads in kto_results.items()
            },
        },
    )

    update_latest_paths("init_imbalance_compare", output_root)
    print(f"Saved init imbalance comparison results to: {output_root}")


if __name__ == "__main__":
    main()
