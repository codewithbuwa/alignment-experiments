"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Sweep KTO zone width and plot the resulting density and sigma changes.
"""

import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig
from src.data import make_kto_samples
from src.distributions import gaussian_pdf
from src.train import train_kto
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def main():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"kto_zone_sweep_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    epsilons = []
    final_sigmas = []
    policies = []

    for eps in cfg.zone_sweep:
        cfg_eps = ExperimentConfig(**cfg.__dict__)
        cfg_eps.zone_half_width = eps

        y, labels = make_kto_samples(
            cfg.mu_ref,
            cfg.sigma_ref,
            cfg.target,
            cfg_eps.zone_half_width,
            cfg.dataset_size,
            cfg.kto_good_fraction,
            cfg.device,
        )
        out = train_kto(y, labels, cfg_eps)

        epsilons.append(eps)
        final_sigmas.append(out["policy"].sigma.item())
        policies.append(out["policy"])

    plt.figure()
    plt.plot(epsilons, final_sigmas, marker="o")
    plt.xlabel("zone half-width (epsilon)")
    plt.ylabel("final sigma")
    plt.title("KTO Zone Width Sweep")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "kto_zone_sweep.png"))
    plt.close()

    export_report_figure(
        os.path.join(figures_dir, "kto_zone_sweep.png"),
        "single_kto_zone_sweep.png",
    )

    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)
    ref_pdf = gaussian_pdf(y_vals, cfg.mu_ref, cfg.sigma_ref)
    colors = ["b", "r", "g", "orange", "purple"]

    plt.figure(figsize=(10, 6))
    plt.plot(y_vals, ref_pdf, "k--", label="Reference", linewidth=1.5)
    for pol, color, eps in zip(policies, colors, epsilons):
        density = torch.exp(pol.log_prob(y_vals)).detach()
        plt.plot(y_vals, density, color=color, label=f"delta = {eps}", linewidth=1.5)

    plt.axvspan(
        cfg.target - cfg.zone_half_width,
        cfg.target + cfg.zone_half_width,
        alpha=0.1,
        color="green",
        label="Base desirable zone",
    )
    plt.xlabel("y")
    plt.ylabel("Density")
    plt.title("Impact of delta on KTO density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    density_out = os.path.join(figures_dir, "impact_of_delta_on_kto_density.png")
    plt.savefig(density_out)
    plt.close()

    export_report_figure(
        density_out,
        "impact_of_delta_on_kto_density.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "kto_zone_sweep.json"),
        {"epsilons": epsilons, "final_sigmas": final_sigmas},
    )

    update_latest_paths("kto_zone_sweep", output_root)

    print(f"Saved KTO zone sweep results to: {output_root}")


if __name__ == "__main__":
    main()
