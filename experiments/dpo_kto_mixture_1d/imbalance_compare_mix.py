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


def parse_alpha_list(alpha_csv: str):
    return [float(x.strip()) for x in alpha_csv.split(",") if x.strip()]


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


def plot_entropy(results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for alpha, payload in results.items():
        label = f"alpha={alpha:.1f}"
        axes[0].plot(payload["dpo"]["history"]["entropy"], label=label)
        axes[1].plot(payload["kto"]["history"]["entropy"], label=label)

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


def plot_parameter_dynamics(results, out_path):
    alphas = sorted(results.keys())
    fig, axes = plt.subplots(3, len(alphas) * 2, figsize=(6 * len(alphas), 10), sharex=True)
    if len(alphas) == 1:
        axes = axes.reshape(3, 2)

    for i, alpha in enumerate(alphas):
        payload = results[alpha]
        for row, key in enumerate(["mus", "sigmas", "weights"]):
            dpo_ax = axes[row, 2 * i]
            kto_ax = axes[row, 2 * i + 1]
            dpo_series = payload["dpo"]["series"][key]
            kto_series = payload["kto"]["series"][key]
            for c in range(dpo_series.shape[1]):
                dpo_ax.plot(dpo_series[:, c], label=f"C{c}")
                kto_ax.plot(kto_series[:, c], label=f"C{c}")
            dpo_ax.set_title(f"DPO {key} alpha={alpha:.1f}")
            kto_ax.set_title(f"KTO {key} alpha={alpha:.1f}")
            dpo_ax.grid(True, alpha=0.3)
            kto_ax.grid(True, alpha=0.3)
            dpo_ax.legend()
            kto_ax.legend()

    for ax in axes[-1, :]:
        ax.set_xlabel("Training step")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_density_grid(results, cfg: MixtureDPOKTOConfig, base_cfg: ExperimentConfig, ref_policy, out_path):
    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)
    ref_pdf = torch.exp(ref_policy.log_prob(y_vals)).detach()
    alphas = sorted(results.keys())

    fig, axes = plt.subplots(2, len(alphas), figsize=(5 * len(alphas), 8), sharex=True, sharey=True)
    if len(alphas) == 1:
        axes = axes.reshape(2, 1)

    zone_fill = ((y_vals >= base_cfg.target - base_cfg.zone_half_width) & (y_vals <= base_cfg.target + base_cfg.zone_half_width)).float().numpy() * 0.5

    for col, alpha in enumerate(alphas):
        payload = results[alpha]
        dpo_hist = payload["dpo"]["history"]
        kto_hist = payload["kto"]["history"]

        dpo_pdf = _mixture_pdf(y_vals, dpo_hist["mus"][-1], dpo_hist["sigmas"][-1], dpo_hist["logits"][-1])
        kto_pdf = _mixture_pdf(y_vals, kto_hist["mus"][-1], kto_hist["sigmas"][-1], kto_hist["logits"][-1])

        for ax, pdf, hist, title in [
            (axes[0, col], dpo_pdf, dpo_hist, f"DPO alpha={alpha:.1f}"),
            (axes[1, col], kto_pdf, kto_hist, f"KTO alpha={alpha:.1f}"),
        ]:
            ax.plot(y_vals.numpy(), ref_pdf.numpy(), linestyle="--", color="black", label="Reference")
            ax.plot(y_vals.numpy(), pdf.numpy(), color="tab:blue", label="Policy")
            ax.fill_between(y_vals.numpy(), 0, zone_fill, alpha=0.15, label="Desirable Zone")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.text(
                0.02,
                0.98,
                _format_components(hist["mus"][-1], hist["sigmas"][-1], hist["logits"][-1]),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
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
    parser = argparse.ArgumentParser(description="Compare 10% vs 50% imbalance for mixture DPO and KTO.")
    parser.add_argument("--alphas", type=str, default="0.1,0.5")
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
    results = {}
    for alpha in parse_alpha_list(args.alphas):
        dpo_policy, _, dpo_hist, dpo_splits = train_dpo_mixture(ref_policy, cfg, good_ratio=alpha)
        kto_policy, _, kto_hist, kto_splits = train_kto_mixture(ref_policy, cfg, good_ratio=alpha)
        results[alpha] = {
            "dpo": {
                "policy": dpo_policy,
                "history": dpo_hist,
                "splits": dpo_splits,
                "series": {
                    "mus": torch.tensor(dpo_hist["mus"]),
                    "sigmas": torch.tensor(dpo_hist["sigmas"]),
                    "weights": torch.softmax(torch.tensor(dpo_hist["logits"]), dim=1),
                },
            },
            "kto": {
                "policy": kto_policy,
                "history": kto_hist,
                "splits": kto_splits,
                "series": {
                    "mus": torch.tensor(kto_hist["mus"]),
                    "sigmas": torch.tensor(kto_hist["sigmas"]),
                    "weights": torch.softmax(torch.tensor(kto_hist["logits"]), dim=1),
                },
            },
        }

    entropy_path = os.path.join(figures_dir, "imbalance_entropy_mix.png")
    density_path = os.path.join(figures_dir, "imbalance_density_grid_mix.png")
    params_path = os.path.join(figures_dir, "imbalance_parameter_dynamics_mix.png")

    plot_entropy(results, entropy_path)
    plot_density_grid(results, cfg, base_cfg, ref_policy, density_path)
    plot_parameter_dynamics(results, params_path)

    export_report_figure(entropy_path, "mix_imbalance_entropy.png")
    export_report_figure(density_path, "mix_imbalance_density.png")
    export_report_figure(params_path, "mix_imbalance_params.png")

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "imbalance_summary.json"),
        {
            str(alpha): {
                "dpo": {
                    "final_mus": payload["dpo"]["history"]["mus"][-1],
                    "final_sigmas": payload["dpo"]["history"]["sigmas"][-1],
                    "final_logits": payload["dpo"]["history"]["logits"][-1],
                    "splits": payload["dpo"]["splits"],
                },
                "kto": {
                    "final_mus": payload["kto"]["history"]["mus"][-1],
                    "final_sigmas": payload["kto"]["history"]["sigmas"][-1],
                    "final_logits": payload["kto"]["history"]["logits"][-1],
                    "splits": payload["kto"]["splits"],
                },
            }
            for alpha, payload in results.items()
        },
    )

    update_latest_paths("mix_imbalance_compare", output_root)
    print(f"Saved mixture imbalance comparison results to: {output_root}")


if __name__ == "__main__":
    main()
