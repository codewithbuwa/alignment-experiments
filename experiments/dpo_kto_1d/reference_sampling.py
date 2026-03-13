"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Compare KTO KL-estimation modes for the single-Gaussian setup.
"""

import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig
from src.data import make_kto_samples
from src.distributions import gaussian_pdf
from src.montage import make_montage
from src.train import train_kto
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


def main():
    cfg = ExperimentConfig()
    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"reference_sampling_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)
    ref_pdf = gaussian_pdf(y_vals, cfg.mu_ref, cfg.sigma_ref)

    modes = ["analytic", "batch", "running", "fixed"]
    policies = []
    sigmas = []
    histories = []

    for mode in modes:
        set_seed(cfg.seed)
        cfg_mode = ExperimentConfig(**cfg.__dict__)
        cfg_mode.kl_mode = mode
        cfg_mode.kl_fixed = 0.3

        y, labels = make_kto_samples(
            cfg.mu_ref,
            cfg.sigma_ref,
            cfg.target,
            cfg.zone_half_width,
            cfg.dataset_size,
            cfg.kto_good_fraction,
            cfg.device,
            delta=1.5,
        )
        out = train_kto(y, labels, cfg_mode)
        policies.append(out["policy"])
        sigmas.append(out["history"]["sigma"])
        histories.append(out["history"])

    plt.figure()
    plt.plot(y_vals.numpy(), ref_pdf.numpy(), label="Reference policy")
    for mode, pol in zip(modes, policies):
        pdf = gaussian_pdf(y_vals, pol.mu.item(), pol.sigma.item())
        plt.plot(y_vals.numpy(), pdf.numpy(), label=f"KTO {mode}")
    plt.legend()
    plt.title("Density Projection")
    plt.savefig(os.path.join(figures_dir, "reference_sampling.png"))
    plt.close()

    plt.figure()
    for mode, s in zip(modes, sigmas):
        plt.plot(s, label=mode)
    plt.legend()
    plt.title("Reference Sampling Sigma Dynamics")
    plt.xlabel("step")
    plt.ylabel("sigma")
    plt.savefig(os.path.join(figures_dir, "reference_sampling_entropy_dynamics.png"))
    plt.close()

    quartiles = _quartile_indices(len(sigmas[0]))
    pct = [0, 25, 50, 75, 100]
    quartile_paths = []
    for idx, p in zip(quartiles, pct):
        plt.figure()
        plt.plot(y_vals.numpy(), ref_pdf.numpy(), label="Reference policy")
        for mode, hist in zip(modes, histories):
            mu = hist["mu"][idx]
            sigma = hist["sigma"][idx]
            pdf = gaussian_pdf(y_vals, mu, sigma)
            plt.plot(y_vals.numpy(), pdf.numpy(), label=f"KTO {mode} {p}%")
        plt.legend()
        plt.title(f"Density Projection at {p}% Training")
        q_path = os.path.join(figures_dir, f"reference_sampling_q{p}.png")
        plt.savefig(q_path)
        plt.close()
        quartile_paths.append(q_path)

    make_montage(quartile_paths, os.path.join(figures_dir, "reference_sampling_quartiles_montage.png"), cols=3)
    _delete_paths(quartile_paths)

    export_report_figure(
        os.path.join(figures_dir, "reference_sampling_quartiles_montage.png"),
        "single_reference_sampling_montage.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "reference_sampling.png"),
        "single_reference_sampling.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "reference_sampling_entropy_dynamics.png"),
        "single_reference_sampling_entropy_dynamics.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "reference_sampling_quartiles.json"),
        {
            "quartile_steps": quartiles,
            "histories": histories,
        },
    )

    update_latest_paths("reference_sampling", output_root)

    print(f"Saved reference sampling results to: {output_root}")


if __name__ == "__main__":
    main()
