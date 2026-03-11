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


def parse_alpha_list(alpha_csv: str):
    return [float(x.strip()) for x in alpha_csv.split(",") if x.strip()]


def _param_box(ax, dpo_out, kto_out):
    text = "\n".join(
        [
            f"DPO: mu={dpo_out['policy'].mu.item():.2f}, sigma={dpo_out['policy'].sigma.item():.2f}",
            f"KTO: mu={kto_out['policy'].mu.item():.2f}, sigma={kto_out['policy'].sigma.item():.2f}",
        ]
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )


def plot_entropy(results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for alpha, payload in results.items():
        label = f"alpha={alpha:.1f}"
        axes[0].plot(payload["dpo"]["history"]["entropy"], label=label)
        axes[1].plot(payload["kto"]["history"]["entropy"], label=label)

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


def plot_parameter_dynamics(results, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for alpha, payload in results.items():
        label = f"alpha={alpha:.1f}"
        axes[0, 0].plot(payload["dpo"]["history"]["mu"], label=label)
        axes[1, 0].plot(payload["dpo"]["history"]["sigma"], label=label)
        axes[0, 1].plot(payload["kto"]["history"]["mu"], label=label)
        axes[1, 1].plot(payload["kto"]["history"]["sigma"], label=label)

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


def plot_density_grid(results, cfg: ExperimentConfig, out_path):
    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)
    ref_pdf = gaussian_pdf(y_vals, cfg.mu_ref, cfg.sigma_ref)
    alphas = sorted(results.keys())

    fig, axes = plt.subplots(2, len(alphas), figsize=(5 * len(alphas), 8), sharex=True, sharey=True)
    if len(alphas) == 1:
        axes = axes.reshape(2, 1)

    zone_mask = (y_vals >= cfg.target - cfg.zone_half_width) & (y_vals <= cfg.target + cfg.zone_half_width)
    zone_x = y_vals[zone_mask]

    for col, alpha in enumerate(alphas):
        payload = results[alpha]
        dpo_pdf = gaussian_pdf(y_vals, payload["dpo"]["policy"].mu.item(), payload["dpo"]["policy"].sigma.item())
        kto_pdf = gaussian_pdf(y_vals, payload["kto"]["policy"].mu.item(), payload["kto"]["policy"].sigma.item())

        dpo_ax = axes[0, col]
        kto_ax = axes[1, col]

        for ax, pdf, title in [
            (dpo_ax, dpo_pdf, f"DPO alpha={alpha:.1f}"),
            (kto_ax, kto_pdf, f"KTO alpha={alpha:.1f}"),
        ]:
            ax.plot(y_vals.numpy(), ref_pdf.numpy(), linestyle="--", color="black", label="Reference")
            ax.plot(y_vals.numpy(), pdf.numpy(), color="tab:blue", label="Policy")
            ax.fill_between(zone_x.numpy(), y1=1, alpha=0.1, label="Desirable Zone")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        dpo_ax.text(
            0.02,
            0.98,
            f"mu={payload['dpo']['policy'].mu.item():.2f}\nsigma={payload['dpo']['policy'].sigma.item():.2f}",
            transform=dpo_ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
        kto_ax.text(
            0.02,
            0.98,
            f"mu={payload['kto']['policy'].mu.item():.2f}\nsigma={payload['kto']['policy'].sigma.item():.2f}",
            transform=kto_ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )

    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density")
    for ax in axes[-1, :]:
        ax.set_xlabel("y")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare 10% vs 50% imbalance for single-Gaussian DPO and KTO.")
    parser.add_argument("--alphas", type=str, default="0.1,0.5")
    parser.add_argument("--delta", type=float, default=0.0)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"imbalance_compare_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    results = {}
    for alpha in parse_alpha_list(args.alphas):
        y_w, y_l = make_dpo_pairs(
            cfg.mu_ref,
            cfg.sigma_ref,
            cfg.target,
            cfg.dataset_size,
            cfg.device,
            good_ratio=alpha,
            zone_half_width=cfg.zone_half_width,
        )
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
        results[alpha] = {
            "dpo": train_dpo(y_w, y_l, cfg),
            "kto": train_kto(y, labels, cfg),
        }

    entropy_path = os.path.join(figures_dir, "imbalance_entropy.png")
    density_path = os.path.join(figures_dir, "imbalance_density_grid.png")
    params_path = os.path.join(figures_dir, "imbalance_parameter_dynamics.png")

    plot_entropy(results, entropy_path)
    plot_density_grid(results, cfg, density_path)
    plot_parameter_dynamics(results, params_path)

    export_report_figure(entropy_path, "single_imbalance_entropy.png")
    export_report_figure(density_path, "single_imbalance_density.png")
    export_report_figure(params_path, "single_imbalance_params.png")

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "imbalance_summary.json"),
        {
            str(alpha): {
                "dpo": {
                    "final_mu": payload["dpo"]["policy"].mu.item(),
                    "final_sigma": payload["dpo"]["policy"].sigma.item(),
                    "splits": payload["dpo"]["splits"],
                },
                "kto": {
                    "final_mu": payload["kto"]["policy"].mu.item(),
                    "final_sigma": payload["kto"]["policy"].sigma.item(),
                    "splits": payload["kto"]["splits"],
                },
            }
            for alpha, payload in results.items()
        },
    )

    update_latest_paths("single_imbalance_compare", output_root)
    print(f"Saved single imbalance comparison results to: {output_root}")


if __name__ == "__main__":
    main()
