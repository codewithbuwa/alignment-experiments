import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import MixtureDPOKTOConfig, ExperimentConfig
from src.ref_policies import make_reference_mixture
from src.train_mix import train_dpo_mixture, train_kto_mixture
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def _milestone_indices(n_steps: int):
    if n_steps <= 1:
        return [0]
    return [
        int(0.25 * (n_steps - 1)),
        int(0.5 * (n_steps - 1)),
        int(0.75 * (n_steps - 1)),
        n_steps - 1,
    ]


def _mixture_component_pdfs(y_vals, mus, sigmas, logits):
    weights = torch.softmax(torch.tensor(logits), dim=0)
    comps = []
    total = torch.zeros_like(y_vals)
    for w, m, s in zip(weights, mus, sigmas):
        comp = w * (1.0 / (s * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-0.5 * ((y_vals - m) / s) ** 2)
        comps.append(comp)
        total = total + comp
    return comps, total, weights


def plot_component_evolution(history, title, output_path):
    steps = list(range(len(history["mus"])))
    n_components = len(history["mus"][0])

    weights_hist = []
    for logits in history["logits"]:
        weights_hist.append(torch.softmax(torch.tensor(logits), dim=0).tolist())

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    for i in range(n_components):
        axes[0].plot(steps, [m[i] for m in history["mus"]], label=f"mu{i}")
    axes[0].set_title("Means")
    axes[0].legend(ncol=3)

    for i in range(n_components):
        axes[1].plot(steps, [s[i] for s in history["sigmas"]], label=f"sigma{i}")
    axes[1].set_title("Sigmas")
    axes[1].legend(ncol=3)

    for i in range(n_components):
        axes[2].plot(steps, [w[i] for w in weights_hist], label=f"w{i}")
    axes[2].set_title("Weights")
    axes[2].set_xlabel("step")
    axes[2].legend(ncol=3)

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_density_milestones(history, title, output_path, y_min, y_max, zone_min, zone_max):
    y_vals = torch.linspace(y_min, y_max, 1000)
    idxs = _milestone_indices(len(history["mus"]))
    pct = [25, 50, 75, 100]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, idx, p in zip(axes, idxs, pct):
        comps, total, weights = _mixture_component_pdfs(
            y_vals, history["mus"][idx], history["sigmas"][idx], history["logits"][idx]
        )

        for k, comp in enumerate(comps):
            ax.plot(y_vals.numpy(), comp.numpy(), linestyle="--", label=f"C{k}")
        ax.plot(y_vals.numpy(), total.numpy(), color="black", linewidth=2, label="Mixture")
        ax.fill_between(
            y_vals.numpy(),
            0,
            ((y_vals >= zone_min) & (y_vals <= zone_max)).float().numpy() * 0.5,
            alpha=0.15,
            label="Desirable Zone",
        )

        params = []
        for k in range(len(history["mus"][idx])):
            params.append(
                f"C{k}: mu={history['mus'][idx][k]:.2f}, sigma={history['sigmas'][idx][k]:.2f}, w={weights[k].item():.2f}"
            )

        ax.set_title(f"{p}% epochs")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.02,
            0.98,
            "\n".join(params),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    cfg = MixtureDPOKTOConfig()
    base_cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_mixture_1d", f"component_evolution_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    ref_policy = make_reference_mixture(cfg.n_components, base_cfg.mu_ref, base_cfg.sigma_ref, cfg.device)

    _, _, dpo_hist, _ = train_dpo_mixture(ref_policy, cfg)
    _, _, kto_hist, _ = train_kto_mixture(ref_policy, cfg)

    plot_component_evolution(
        dpo_hist,
        "DPO Mixture Component Evolution",
        os.path.join(figures_dir, "dpo_component_evolution.png"),
    )
    plot_component_evolution(
        kto_hist,
        "KTO Mixture Component Evolution",
        os.path.join(figures_dir, "kto_component_evolution.png"),
    )

    zone_min = base_cfg.target - base_cfg.zone_half_width
    zone_max = base_cfg.target + base_cfg.zone_half_width

    plot_density_milestones(
        dpo_hist,
        "DPO Mixture Density (25% Milestones)",
        os.path.join(figures_dir, "dpo_density_milestones.png"),
        cfg.y_min,
        cfg.y_max,
        zone_min,
        zone_max,
    )

    plot_density_milestones(
        kto_hist,
        "KTO Mixture Density (25% Milestones)",
        os.path.join(figures_dir, "kto_density_milestones.png"),
        cfg.y_min,
        cfg.y_max,
        zone_min,
        zone_max,
    )

    export_report_figure(
        os.path.join(figures_dir, "dpo_component_evolution.png"),
        "mix_component_evolution_dpo.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "kto_component_evolution.png"),
        "mix_component_evolution_kto.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "dpo_density_milestones.png"),
        "mix_density_milestones_dpo.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "kto_density_milestones.png"),
        "mix_density_milestones_kto.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "component_evolution.json"),
        {
            "dpo": dpo_hist,
            "kto": kto_hist,
            "milestone_steps": _milestone_indices(len(dpo_hist["mus"])),
        },
    )

    update_latest_paths("mix_component_evolution", output_root)

    print(f"Saved component evolution results to: {output_root}")


if __name__ == "__main__":
    main()
