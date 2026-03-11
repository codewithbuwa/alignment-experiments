import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig
from src.data import make_dpo_pairs, make_kto_samples
from src.distributions import gaussian_pdf
from src.montage import make_montage
from src.train import train_dpo, train_kto
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
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"density_overlay_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    y_w, y_l = make_dpo_pairs(cfg.mu_ref, cfg.sigma_ref, cfg.target, cfg.dataset_size, cfg.device)
    dpo_out = train_dpo(y_w, y_l, cfg)

    y, labels = make_kto_samples(
        cfg.mu_ref,
        cfg.sigma_ref,
        cfg.target,
        cfg.zone_half_width,
        cfg.dataset_size,
        cfg.kto_good_fraction,
        cfg.device,
    )
    kto_out = train_kto(y, labels, cfg)

    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)
    ref_pdf = gaussian_pdf(y_vals, cfg.mu_ref, cfg.sigma_ref)

    # Final overlay
    dpo_pdf = gaussian_pdf(y_vals, dpo_out["policy"].mu.item(), dpo_out["policy"].sigma.item())
    kto_pdf = gaussian_pdf(y_vals, kto_out["policy"].mu.item(), kto_out["policy"].sigma.item())

    plt.figure()
    plt.plot(y_vals.numpy(), ref_pdf.numpy(), label="Reference")
    plt.plot(y_vals.numpy(), dpo_pdf.numpy(), label="DPO")
    plt.plot(y_vals.numpy(), kto_pdf.numpy(), label="KTO")

    shade_mask = (y_vals >= cfg.target - cfg.zone_half_width) & (y_vals <= cfg.target + cfg.zone_half_width)
    shade_x = y_vals[shade_mask]
    plt.fill_between(shade_x.numpy(), y1=1, alpha=0.1, label="Reference region")

    plt.legend()
    plt.title("Density Projection")
    plt.savefig(os.path.join(figures_dir, "density_projection.png"))
    plt.close()

    # Quartile overlays -> montage only
    quartiles = _quartile_indices(len(dpo_out["history"]["sigma"]))
    pct = [0, 25, 50, 75, 100]
    quartile_paths = []
    for idx, p in zip(quartiles, pct):
        dpo_pdf_q = gaussian_pdf(y_vals, dpo_out["history"]["mu"][idx], dpo_out["history"]["sigma"][idx])
        kto_pdf_q = gaussian_pdf(y_vals, kto_out["history"]["mu"][idx], kto_out["history"]["sigma"][idx])

        plt.figure()
        plt.plot(y_vals.numpy(), ref_pdf.numpy(), label="Reference")
        plt.plot(y_vals.numpy(), dpo_pdf_q.numpy(), label=f"DPO {p}%")
        plt.plot(y_vals.numpy(), kto_pdf_q.numpy(), label=f"KTO {p}%")
        plt.fill_between(shade_x.numpy(), y1=1, alpha=0.1, label="Reference region")
        plt.legend()
        plt.title(f"Density Projection at {p}% Training")
        q_path = os.path.join(figures_dir, f"density_projection_q{p}.png")
        plt.savefig(q_path)
        plt.close()
        quartile_paths.append(q_path)

    montage_path = os.path.join(figures_dir, "density_projection_quartiles_montage.png")
    make_montage(quartile_paths, montage_path, cols=3)
    _delete_paths(quartile_paths)

    export_report_figure(
        montage_path,
        "single_density_montage.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "density_projection.png"),
        "single_density_projection.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "density_quartiles.json"),
        {
            "quartile_steps": quartiles,
            "dpo_mu": [dpo_out["history"]["mu"][i] for i in quartiles],
            "dpo_sigma": [dpo_out["history"]["sigma"][i] for i in quartiles],
            "kto_mu": [kto_out["history"]["mu"][i] for i in quartiles],
            "kto_sigma": [kto_out["history"]["sigma"][i] for i in quartiles],
        },
    )

    update_latest_paths("density_overlay", output_root)

    print(f"Saved density overlay results to: {output_root}")


if __name__ == "__main__":
    main()
