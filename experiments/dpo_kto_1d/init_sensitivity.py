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


def _plot_density(y_vals, ref_pdf, dpo_pdf, kto_pdf, title, output_path, annotation=None):
    plt.figure()
    plt.plot(y_vals.numpy(), ref_pdf.numpy(), label="Reference")
    plt.plot(y_vals.numpy(), dpo_pdf.numpy(), label="DPO")
    plt.plot(y_vals.numpy(), kto_pdf.numpy(), label="KTO")
    shade_mask = (y_vals >= 5.5) & (y_vals <= 8.5)
    shade_x = y_vals[shade_mask]
    plt.fill_between(shade_x.numpy(), y1=1, alpha=0.1, label="Reference region")
    plt.legend()
    plt.title(title)
    if annotation is not None:
        plt.gca().text(
            0.02,
            0.98,
            annotation,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
    plt.savefig(output_path)
    plt.close()


def main():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"init_sensitivity_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    y_w, y_l = make_dpo_pairs(cfg.mu_ref, cfg.sigma_ref, cfg.target, cfg.dataset_size, cfg.device)
    y, labels = make_kto_samples(
        cfg.mu_ref,
        cfg.sigma_ref,
        cfg.target,
        cfg.zone_half_width,
        cfg.dataset_size,
        cfg.kto_good_fraction,
        cfg.device,
    )

    inits = {
        "around": cfg.target,
        "left": cfg.target - 4.0,
        "right": cfg.target + 4.0,
    }

    summary = {"dpo": {}, "kto": {}}

    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)
    ref_pdf = gaussian_pdf(y_vals, cfg.mu_ref, cfg.sigma_ref)

    for name, mu_init in inits.items():
        cfg_init = ExperimentConfig(**cfg.__dict__)
        cfg_init.init_mu = mu_init
        cfg_init.init_sigma = cfg.sigma_ref

        dpo_out = train_dpo(y_w, y_l, cfg_init)
        kto_out = train_kto(y, labels, cfg_init)

        summary["dpo"][name] = {
            "init_mu": mu_init,
            "final_mu": dpo_out["history"]["mu"][-1],
            "final_sigma": dpo_out["history"]["sigma"][-1],
        }
        summary["kto"][name] = {
            "init_mu": mu_init,
            "final_mu": kto_out["history"]["mu"][-1],
            "final_sigma": kto_out["history"]["sigma"][-1],
        }

        quartiles = _quartile_indices(len(dpo_out["history"]["sigma"]))
        pct = [0, 25, 50, 75, 100]
        quartile_paths = []
        for idx, p in zip(quartiles, pct):
            dpo_pdf_q = gaussian_pdf(y_vals, dpo_out["history"]["mu"][idx], dpo_out["history"]["sigma"][idx])
            kto_pdf_q = gaussian_pdf(y_vals, kto_out["history"]["mu"][idx], kto_out["history"]["sigma"][idx])

            q_path = os.path.join(figures_dir, f"density_{name}_q{p}.png")
            _plot_density(
                y_vals,
                ref_pdf,
                dpo_pdf_q,
                kto_pdf_q,
                f"Density ({name}) at {p}%",
                q_path,
                annotation=(
                    f"DPO: mu={dpo_out['history']['mu'][idx]:.2f}, sigma={dpo_out['history']['sigma'][idx]:.2f}\n"
                    f"KTO: mu={kto_out['history']['mu'][idx]:.2f}, sigma={kto_out['history']['sigma'][idx]:.2f}"
                ),
            )
            quartile_paths.append(q_path)

        montage_path = os.path.join(figures_dir, f"density_{name}_quartiles_montage.png")
        make_montage(quartile_paths, montage_path, cols=3)
        _delete_paths(quartile_paths)

        export_report_figure(
            montage_path,
            f"single_init_density_{name}.png",
        )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(os.path.join(runs_dir, "init_summary.json"), summary)

    update_latest_paths("init_sensitivity", output_root)

    print(f"Saved init sensitivity results to: {output_root}")


if __name__ == "__main__":
    main()
