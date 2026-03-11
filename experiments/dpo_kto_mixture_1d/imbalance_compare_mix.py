"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Compare 10% and 50% imbalance for mixture DPO and KTO.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig, MixtureDPOKTOConfig
from src.ref_policies import make_reference_mixture
from src.train_mix import train_dpo_mixture, train_kto_mixture
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def parse_ratio_list(csv_text: str):
    return [float(x.strip()) for x in csv_text.split(",") if x.strip()]


def _mixture_pdf(y, mus, sigmas, logits):
    weights = torch.softmax(torch.tensor(logits), dim=0)
    pdf = torch.zeros_like(y)
    for w, m, s in zip(weights, mus, sigmas):
        coeff = 1.0 / (s * torch.sqrt(torch.tensor(2 * torch.pi)))
        pdf = pdf + w * coeff * torch.exp(-0.5 * ((y - m) / s) ** 2)
    return pdf


def _format_components(mus, sigmas, logits):
    weights = torch.softmax(torch.tensor(logits), dim=0)
    return "\n".join(
        [f"C{k}: mu={mus[k]:.2f}, sigma={sigmas[k]:.2f}, w={weights[k].item():.2f}" for k in range(len(mus))]
    )


def _effective_good_mass_dpo(y_w: torch.Tensor, cfg: ExperimentConfig) -> float:
    zone_min = cfg.target - cfg.zone_half_width
    zone_max = cfg.target + cfg.zone_half_width
    return ((y_w >= zone_min) & (y_w <= zone_max)).float().mean().item()


def _effective_good_mass_kto_from_history(policy, cfg: ExperimentConfig, n_samples: int = 2000) -> float:
    with torch.no_grad():
        y = policy.sample(n_samples)
        zone_min = cfg.target - cfg.zone_half_width
        zone_max = cfg.target + cfg.zone_half_width
        return ((y >= zone_min) & (y <= zone_max)).float().mean().item()


def plot_entropy(dpo_results, kto_results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for good_ratio, payload in dpo_results.items():
        axes[0].plot(payload["history"]["entropy"], label=f"good_ratio={good_ratio:.1f}, eff={payload['effective_good_mass']:.2f}")
    for alpha, payload in kto_results.items():
        axes[1].plot(payload["history"]["entropy"], label=f"alpha={alpha:.1f}, eff={payload['effective_good_mass']:.2f}")

    axes[0].set_title("DPO Mixture Entropy")
    axes[1].set_title("KTO Mixture Entropy")
    for ax in axes:
        ax.set_xlabel("Training step")
        ax.set_ylabel("Entropy")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_parameter_dynamics(dpo_results, kto_results, out_path):
    dpo_ratios = sorted(dpo_results.keys())
    kto_alphas = sorted(kto_results.keys())
    n_cols = max(len(dpo_ratios), len(kto_alphas))
    fig, axes = plt.subplots(3, n_cols * 2, figsize=(6 * n_cols, 10), sharex=True)
    if n_cols == 1:
        axes = axes.reshape(3, 2)

    for i, good_ratio in enumerate(dpo_ratios):
        payload = dpo_results[good_ratio]
        for row, key in enumerate(["mus", "sigmas", "weights"]):
            dpo_ax = axes[row, 2 * i]
            dpo_series = payload["series"][key]
            for c in range(dpo_series.shape[1]):
                dpo_ax.plot(dpo_series[:, c], label=f"C{c}")
            dpo_ax.set_title(f"DPO {key} good_ratio={good_ratio:.1f}")
            dpo_ax.grid(True, alpha=0.3)
            dpo_ax.legend()

    for i, alpha in enumerate(kto_alphas):
        payload = kto_results[alpha]
        for row, key in enumerate(["mus", "sigmas", "weights"]):
            kto_ax = axes[row, 2 * i + 1]
            kto_series = payload["series"][key]
            for c in range(kto_series.shape[1]):
                kto_ax.plot(kto_series[:, c], label=f"C{c}")
            kto_ax.set_title(f"KTO {key} alpha={alpha:.1f}")
            kto_ax.grid(True, alpha=0.3)
            kto_ax.legend()

    for ax in axes[-1, :]:
        ax.set_xlabel("Training step")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_density_grid(dpo_results, kto_results, cfg: MixtureDPOKTOConfig, base_cfg: ExperimentConfig, ref_policy, out_path):
    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)
    ref_pdf = torch.exp(ref_policy.log_prob(y_vals)).detach()
    dpo_ratios = sorted(dpo_results.keys())
    kto_alphas = sorted(kto_results.keys())

    n_cols = max(len(dpo_ratios), len(kto_alphas))
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), sharex=True, sharey=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    zone_fill = ((y_vals >= base_cfg.target - base_cfg.zone_half_width) & (y_vals <= base_cfg.target + base_cfg.zone_half_width)).float().numpy() * 0.5

    for col in range(n_cols):
        dpo_ax = axes[0, col]
        kto_ax = axes[1, col]

        if col < len(dpo_ratios):
            good_ratio = dpo_ratios[col]
            dpo_hist = dpo_results[good_ratio]["history"]
            dpo_pdf = _mixture_pdf(y_vals, dpo_hist["mus"][-1], dpo_hist["sigmas"][-1], dpo_hist["logits"][-1])
            dpo_ax.plot(y_vals.numpy(), ref_pdf.numpy(), linestyle="--", color="black", label="Reference")
            dpo_ax.plot(y_vals.numpy(), dpo_pdf.numpy(), color="tab:blue", label="Policy")
            dpo_ax.fill_between(y_vals.numpy(), 0, zone_fill, alpha=0.15, label="Desirable Zone")
            dpo_ax.set_title(f"DPO good_ratio={good_ratio:.1f}")
            dpo_ax.grid(True, alpha=0.3)
            dpo_ax.text(
                0.02,
                0.98,
                (
                    f"{_format_components(dpo_hist['mus'][-1], dpo_hist['sigmas'][-1], dpo_hist['logits'][-1])}\n"
                    f"eff={dpo_results[good_ratio]['effective_good_mass']:.2f}"
                ),
                transform=dpo_ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
            )
        else:
            dpo_ax.axis("off")

        if col < len(kto_alphas):
            alpha = kto_alphas[col]
            kto_hist = kto_results[alpha]["history"]
            kto_pdf = _mixture_pdf(y_vals, kto_hist["mus"][-1], kto_hist["sigmas"][-1], kto_hist["logits"][-1])
            kto_ax.plot(y_vals.numpy(), ref_pdf.numpy(), linestyle="--", color="black", label="Reference")
            kto_ax.plot(y_vals.numpy(), kto_pdf.numpy(), color="tab:blue", label="Policy")
            kto_ax.fill_between(y_vals.numpy(), 0, zone_fill, alpha=0.15, label="Desirable Zone")
            kto_ax.set_title(f"KTO alpha={alpha:.1f}")
            kto_ax.grid(True, alpha=0.3)
            kto_ax.text(
                0.02,
                0.98,
                (
                    f"{_format_components(kto_hist['mus'][-1], kto_hist['sigmas'][-1], kto_hist['logits'][-1])}\n"
                    f"eff={kto_results[alpha]['effective_good_mass']:.2f}"
                ),
                transform=kto_ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
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


def main():
    parser = argparse.ArgumentParser(description="Compare DPO good_ratio settings and KTO alpha settings for mixtures.")
    parser.add_argument("--dpo-good-ratios", type=str, default="0.1,1.0")
    parser.add_argument("--kto-alphas", type=str, default="0.1,0.5")
    parser.add_argument("--n-components", type=int, default=2)
    args = parser.parse_args()

    cfg = MixtureDPOKTOConfig()
    cfg.n_components = args.n_components
    base_cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_mixture_1d", f"imbalance_compare_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    ref_policy = make_reference_mixture(cfg.n_components, base_cfg.mu_ref, base_cfg.sigma_ref, cfg.device)
    dpo_results = {}
    for good_ratio in parse_ratio_list(args.dpo_good_ratios):
        dpo_policy, _, dpo_hist, dpo_splits = train_dpo_mixture(ref_policy, cfg, good_ratio=good_ratio)
        dpo_results[good_ratio] = {
            "policy": dpo_policy,
            "history": dpo_hist,
            "splits": dpo_splits,
            "series": {
                "mus": torch.tensor(dpo_hist["mus"]),
                "sigmas": torch.tensor(dpo_hist["sigmas"]),
                "weights": torch.softmax(torch.tensor(dpo_hist["logits"]), dim=1),
            },
            "effective_good_mass": _effective_good_mass_kto_from_history(dpo_policy, base_cfg),
        }

    kto_results = {}
    for alpha in parse_ratio_list(args.kto_alphas):
        kto_policy, _, kto_hist, kto_splits = train_kto_mixture(ref_policy, cfg, good_ratio=alpha)
        kto_results[alpha] = {
            "policy": kto_policy,
            "history": kto_hist,
            "splits": kto_splits,
            "series": {
                "mus": torch.tensor(kto_hist["mus"]),
                "sigmas": torch.tensor(kto_hist["sigmas"]),
                "weights": torch.softmax(torch.tensor(kto_hist["logits"]), dim=1),
            },
            "effective_good_mass": _effective_good_mass_kto_from_history(kto_policy, base_cfg),
        }

    entropy_path = os.path.join(figures_dir, "imbalance_entropy_mix.png")
    density_path = os.path.join(figures_dir, "imbalance_density_grid_mix.png")
    params_path = os.path.join(figures_dir, "imbalance_parameter_dynamics_mix.png")

    plot_entropy(dpo_results, kto_results, entropy_path)
    plot_density_grid(dpo_results, kto_results, cfg, base_cfg, ref_policy, density_path)
    plot_parameter_dynamics(dpo_results, kto_results, params_path)

    export_report_figure(entropy_path, "mix_imbalance_entropy.png")
    export_report_figure(density_path, "mix_imbalance_density.png")
    export_report_figure(params_path, "mix_imbalance_params.png")

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "imbalance_summary.json"),
        {
            "dpo": {
                str(good_ratio): {
                    "final_mus": payload["history"]["mus"][-1],
                    "final_sigmas": payload["history"]["sigmas"][-1],
                    "final_logits": payload["history"]["logits"][-1],
                    "splits": payload["splits"],
                    "effective_good_mass": payload["effective_good_mass"],
                }
                for good_ratio, payload in dpo_results.items()
            },
            "kto": {
                str(alpha): {
                    "final_mus": payload["history"]["mus"][-1],
                    "final_sigmas": payload["history"]["sigmas"][-1],
                    "final_logits": payload["history"]["logits"][-1],
                    "splits": payload["splits"],
                    "effective_good_mass": payload["effective_good_mass"],
                }
                for alpha, payload in kto_results.items()
            },
        },
    )

    update_latest_paths("mix_imbalance_compare", output_root)
    print(f"Saved mixture imbalance comparison results to: {output_root}")


if __name__ == "__main__":
    main()
