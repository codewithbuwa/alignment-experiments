"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Plot mixture density evolution and final density overlays.
"""

import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import MixtureDPOKTOConfig, ExperimentConfig
from src.montage import make_montage
from src.ref_policies import make_reference_mixture
from src.train_mix import train_dpo_mixture, train_kto_mixture
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


def _mixture_pdf(y, mus, sigmas, logits):
    weights = torch.softmax(torch.tensor(logits), dim=0)
    pdf = torch.zeros_like(y)
    for w, m, s in zip(weights, mus, sigmas):
        pdf = pdf + w * (1.0 / (s * torch.sqrt(torch.tensor(2 * torch.pi)))) * \
            torch.exp(-0.5 * ((y - m) / s) ** 2)
    return pdf


def main():
    cfg = MixtureDPOKTOConfig()
    base_cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_mixture_1d", f"density_overlay_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    ref_policy = make_reference_mixture(cfg.n_components, base_cfg.mu_ref, base_cfg.sigma_ref, cfg.device)

    dpo_policy, _, dpo_hist, _ = train_dpo_mixture(ref_policy, cfg)
    kto_policy, _, kto_hist, _ = train_kto_mixture(ref_policy, cfg)

    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)
    ref_pdf = torch.exp(ref_policy.log_prob(y_vals)).detach()

    dpo_pdf = torch.exp(dpo_policy.log_prob(y_vals)).detach()
    kto_pdf = torch.exp(kto_policy.log_prob(y_vals)).detach()

    plt.figure()
    plt.plot(y_vals.numpy(), ref_pdf.numpy(), label="Reference Mixture", linestyle="--")
    plt.plot(y_vals.numpy(), dpo_pdf.numpy(), label="DPO Mixture")
    plt.plot(y_vals.numpy(), kto_pdf.numpy(), label="KTO Mixture")
    plt.fill_between(
        y_vals.numpy(),
        0,
        ((y_vals >= base_cfg.target - base_cfg.zone_half_width) & (y_vals <= base_cfg.target + base_cfg.zone_half_width)).float().numpy() * 0.5,
        alpha=0.2,
        label="Desirable Zone",
    )
    plt.legend()
    plt.title("Mixture Densities after Training")
    plt.xlabel("y")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "mixture_density_overlay.png"))
    plt.close()

    quartiles = _quartile_indices(len(dpo_hist["sigmas"]))
    pct = [0, 25, 50, 75, 100]
    quartile_paths = []
    for idx, p in zip(quartiles, pct):
        dpo_pdf_q = _mixture_pdf(y_vals, dpo_hist["mus"][idx], dpo_hist["sigmas"][idx], dpo_hist["logits"][idx])
        kto_pdf_q = _mixture_pdf(y_vals, kto_hist["mus"][idx], kto_hist["sigmas"][idx], kto_hist["logits"][idx])

        plt.figure()
        plt.plot(y_vals.numpy(), ref_pdf.numpy(), label="Reference Mixture", linestyle="--")
        plt.plot(y_vals.numpy(), dpo_pdf_q.numpy(), label=f"DPO {p}%")
        plt.plot(y_vals.numpy(), kto_pdf_q.numpy(), label=f"KTO {p}%")
        plt.fill_between(
            y_vals.numpy(),
            0,
            ((y_vals >= base_cfg.target - base_cfg.zone_half_width) & (y_vals <= base_cfg.target + base_cfg.zone_half_width)).float().numpy() * 0.5,
            alpha=0.2,
            label="Desirable Zone",
        )
        plt.legend()
        plt.title(f"Mixture Densities at {p}% Training")
        plt.xlabel("y")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        q_path = os.path.join(figures_dir, f"mixture_density_overlay_q{p}.png")
        plt.savefig(q_path)
        plt.close()
        quartile_paths.append(q_path)

    montage_path = os.path.join(figures_dir, "mixture_density_overlay_quartiles_montage.png")
    make_montage(quartile_paths, montage_path, cols=3)
    _delete_paths(quartile_paths)

    export_report_figure(
        montage_path,
        "mix_density_montage.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "density_quartiles.json"),
        {
            "quartile_steps": quartiles,
            "dpo": dpo_hist,
            "kto": kto_hist,
        },
    )

    update_latest_paths("mix_density_overlay", output_root)

    print(f"Saved mixture density overlay results to: {output_root}")


if __name__ == "__main__":
    main()
