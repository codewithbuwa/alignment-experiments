"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Show 3-component mixture density milestones under left/right/around initialization.
"""

import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig, MixtureDPOKTOConfig
from src.ref_policies import make_reference_mixture
from src.train_mix import train_dpo_mixture, train_kto_mixture
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def _make_init_means(center: float, spread: float, n_components: int):
    if n_components == 1:
        return [center]
    return torch.linspace(center - spread, center + spread, n_components).tolist()


def _scenario_centers(cfg: ExperimentConfig):
    return {
        "around": cfg.target,
        "left": cfg.target - 4.0,
        "right": cfg.target + 4.0,
    }


def _weights_from_logits(logits):
    return torch.softmax(torch.tensor(logits), dim=0).tolist()


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


def _milestone_indices(steps: int):
    return [0, int(0.25 * (steps - 1)), int(0.5 * (steps - 1)), int(0.75 * (steps - 1)), steps - 1]


def _plot_density_milestones(summary, ref_pdf, y_vals, zone_half_width: float, target: float, out_path: str):
    scenarios = list(summary["dpo"].keys())
    indices = _milestone_indices(len(summary["dpo"][scenarios[0]]["history"]["mus"]))
    pct = [0, 25, 50, 75, 100]

    fig, axes = plt.subplots(len(scenarios), len(indices), figsize=(4.2 * len(indices), 3.6 * len(scenarios)), sharex=True, sharey=True)
    if len(scenarios) == 1:
        axes = axes.reshape(1, -1)

    zone_fill = ((y_vals >= target - zone_half_width) & (y_vals <= target + zone_half_width)).float().numpy() * 0.5

    for i, scenario in enumerate(scenarios):
        for j, (idx, p) in enumerate(zip(indices, pct)):
            ax = axes[i, j]
            dpo_hist = summary["dpo"][scenario]["history"]
            kto_hist = summary["kto"][scenario]["history"]
            dpo_pdf = _mixture_pdf(y_vals, dpo_hist["mus"][idx], dpo_hist["sigmas"][idx], dpo_hist["logits"][idx])
            kto_pdf = _mixture_pdf(y_vals, kto_hist["mus"][idx], kto_hist["sigmas"][idx], kto_hist["logits"][idx])

            ax.plot(y_vals.numpy(), ref_pdf.numpy(), linestyle="--", color="black", label="Reference")
            ax.plot(y_vals.numpy(), dpo_pdf.numpy(), color="tab:blue", label="DPO")
            ax.plot(y_vals.numpy(), kto_pdf.numpy(), color="tab:orange", label="KTO")
            ax.fill_between(y_vals.numpy(), 0, zone_fill, alpha=0.12, color="gray")
            ax.set_title(f"{scenario}, {p}\\%")
            ax.grid(True, alpha=0.25)
            ax.text(
                0.02,
                0.98,
                (
                    "DPO\n"
                    f"{_format_components(dpo_hist['mus'][idx], dpo_hist['sigmas'][idx], dpo_hist['logits'][idx])}\n"
                    "KTO\n"
                    f"{_format_components(kto_hist['mus'][idx], kto_hist['sigmas'][idx], kto_hist['logits'][idx])}"
                ),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=6.5,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.7"},
            )

    axes[0, 0].legend(loc="upper right", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("y")
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")

    fig.suptitle("3-Component Mixture Initialization: Density Milestones", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_componentwise(summary, out_path: str, n_components: int):
    scenarios = list(summary["dpo"].keys())
    x = range(len(scenarios))
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

    for comp in range(n_components):
        axes[0, 0].plot(x, [summary["dpo"][s]["final_mus"][comp] for s in scenarios], marker="o", label=f"C{comp}")
        axes[1, 0].plot(x, [summary["dpo"][s]["final_sigmas"][comp] for s in scenarios], marker="o", label=f"C{comp}")
        axes[2, 0].plot(x, [summary["dpo"][s]["final_weights"][comp] for s in scenarios], marker="o", label=f"C{comp}")

        axes[0, 1].plot(x, [summary["kto"][s]["final_mus"][comp] for s in scenarios], marker="o", label=f"C{comp}")
        axes[1, 1].plot(x, [summary["kto"][s]["final_sigmas"][comp] for s in scenarios], marker="o", label=f"C{comp}")
        axes[2, 1].plot(x, [summary["kto"][s]["final_weights"][comp] for s in scenarios], marker="o", label=f"C{comp}")

    axes[0, 0].set_title("DPO Final Component Means")
    axes[1, 0].set_title("DPO Final Component Sigmas")
    axes[2, 0].set_title("DPO Final Component Weights")
    axes[0, 1].set_title("KTO Final Component Means")
    axes[1, 1].set_title("KTO Final Component Sigmas")
    axes[2, 1].set_title("KTO Final Component Weights")

    axes[0, 0].set_ylabel("Mu")
    axes[1, 0].set_ylabel("Sigma")
    axes[2, 0].set_ylabel("Weight")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[-1, :]:
        ax.set_xticks(list(x))
        ax.set_xticklabels(scenarios)
        ax.set_xlabel("Initialization scenario")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_training_dynamics(summary, out_path: str, n_components: int):
    scenarios = list(summary["dpo"].keys())
    fig, axes = plt.subplots(3, len(scenarios) * 2, figsize=(5 * len(scenarios), 10), sharex=True)
    if len(scenarios) == 1:
        axes = axes.reshape(3, 2)

    for col, scenario in enumerate(scenarios):
        dpo_hist = summary["dpo"][scenario]["history"]
        kto_hist = summary["kto"][scenario]["history"]

        dpo_mus = torch.tensor(dpo_hist["mus"])
        dpo_sigmas = torch.tensor(dpo_hist["sigmas"])
        dpo_weights = torch.softmax(torch.tensor(dpo_hist["logits"]), dim=1)

        kto_mus = torch.tensor(kto_hist["mus"])
        kto_sigmas = torch.tensor(kto_hist["sigmas"])
        kto_weights = torch.softmax(torch.tensor(kto_hist["logits"]), dim=1)

        dpo_axes = [axes[0, 2 * col], axes[1, 2 * col], axes[2, 2 * col]]
        kto_axes = [axes[0, 2 * col + 1], axes[1, 2 * col + 1], axes[2, 2 * col + 1]]

        for comp in range(n_components):
            dpo_axes[0].plot(dpo_mus[:, comp], label=f"C{comp}")
            dpo_axes[1].plot(dpo_sigmas[:, comp], label=f"C{comp}")
            dpo_axes[2].plot(dpo_weights[:, comp], label=f"C{comp}")

            kto_axes[0].plot(kto_mus[:, comp], label=f"C{comp}")
            kto_axes[1].plot(kto_sigmas[:, comp], label=f"C{comp}")
            kto_axes[2].plot(kto_weights[:, comp], label=f"C{comp}")

        dpo_axes[0].set_title(f"DPO Mu ({scenario})")
        dpo_axes[1].set_title(f"DPO Sigma ({scenario})")
        dpo_axes[2].set_title(f"DPO Weight ({scenario})")
        kto_axes[0].set_title(f"KTO Mu ({scenario})")
        kto_axes[1].set_title(f"KTO Sigma ({scenario})")
        kto_axes[2].set_title(f"KTO Weight ({scenario})")

        for ax in dpo_axes + kto_axes:
            ax.grid(True, alpha=0.3)
            ax.legend()

    for ax in axes[-1, :]:
        ax.set_xlabel("Training step")
    axes[0, 0].set_ylabel("Mu")
    axes[1, 0].set_ylabel("Sigma")
    axes[2, 0].set_ylabel("Weight")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    cfg = MixtureDPOKTOConfig()
    cfg.n_components = 3
    base_cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_mixture_1d", f"init_componentwise_3mix_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    ref_policy = make_reference_mixture(cfg.n_components, base_cfg.mu_ref, base_cfg.sigma_ref, cfg.device)
    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)
    ref_pdf = torch.exp(ref_policy.log_prob(y_vals)).detach()
    summary = {"dpo": {}, "kto": {}}

    for scenario, center in _scenario_centers(base_cfg).items():
        cfg_init = MixtureDPOKTOConfig(**cfg.__dict__)
        cfg_init.init_means = _make_init_means(center, 0.8, cfg.n_components)
        cfg_init.init_sigmas = [base_cfg.sigma_ref for _ in range(cfg.n_components)]
        cfg_init.init_logits = [0.0 for _ in range(cfg.n_components)]

        dpo_policy, _, dpo_hist, _ = train_dpo_mixture(ref_policy, cfg_init)
        kto_policy, _, kto_hist, _ = train_kto_mixture(ref_policy, cfg_init)

        summary["dpo"][scenario] = {
            "init_means": cfg_init.init_means,
            "final_mus": dpo_policy.mus.detach().cpu().tolist(),
            "final_sigmas": dpo_policy.sigmas().detach().cpu().tolist(),
            "final_weights": _weights_from_logits(dpo_policy.logits.detach().cpu()),
            "history": dpo_hist,
        }
        summary["kto"][scenario] = {
            "init_means": cfg_init.init_means,
            "final_mus": kto_policy.mus.detach().cpu().tolist(),
            "final_sigmas": kto_policy.sigmas().detach().cpu().tolist(),
            "final_weights": _weights_from_logits(kto_policy.logits.detach().cpu()),
            "history": kto_hist,
        }

    density_path = os.path.join(figures_dir, "init_componentwise_3mix.png")
    component_path = os.path.join(figures_dir, "init_componentwise_3mix_summary.png")
    dynamics_path = os.path.join(figures_dir, "init_componentwise_3mix_dynamics.png")
    _plot_density_milestones(summary, ref_pdf, y_vals, cfg.zone_half_width, cfg.target, density_path)
    _plot_componentwise(summary, component_path, cfg.n_components)
    _plot_training_dynamics(summary, dynamics_path, cfg.n_components)
    export_report_figure(density_path, "mix_init_componentwise_3mix.png")
    export_report_figure(component_path, "mix_init_componentwise_3mix_summary.png")
    export_report_figure(dynamics_path, "mix_init_componentwise_3mix_dynamics.png")

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(os.path.join(runs_dir, "init_componentwise_3mix_summary.json"), summary)

    update_latest_paths("mix_init_componentwise_3mix", output_root)
    print(f"Saved 3-component init comparison results to: {output_root}")


if __name__ == "__main__":
    main()
