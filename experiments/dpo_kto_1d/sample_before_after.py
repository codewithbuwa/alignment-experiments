"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Visualize sampled points before training and after training for single-Gaussian DPO and KTO across initializations.
"""

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


def _strip(ax, samples: torch.Tensor, title: str, color: str, cfg: ExperimentConfig):
    y = torch.zeros_like(samples)
    ax.scatter(samples.numpy(), y.numpy(), s=12, alpha=0.6, color=color)
    ax.axvspan(cfg.target - cfg.zone_half_width, cfg.target + cfg.zone_half_width, alpha=0.1, color="gray")
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xlim(cfg.y_min, cfg.y_max)
    ax.set_xlabel("y")
    ax.grid(True, alpha=0.2, axis="x")


def _hist(ax, samples: torch.Tensor, pdf_x: torch.Tensor, pdf_y: torch.Tensor, title: str, color: str, cfg: ExperimentConfig):
    ax.hist(samples.numpy(), bins=35, density=True, alpha=0.45, color=color)
    ax.plot(pdf_x.numpy(), pdf_y.numpy(), color="black", linewidth=2)
    ax.axvspan(cfg.target - cfg.zone_half_width, cfg.target + cfg.zone_half_width, alpha=0.1, color="gray")
    ax.set_title(title)
    ax.set_xlim(cfg.y_min, cfg.y_max)
    ax.set_xlabel("y")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.2)


def _init_scenarios(cfg: ExperimentConfig):
    return {
        "around": cfg.target,
        "left": cfg.target - 4.0,
        "right": cfg.target + 4.0,
    }


def main():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"sample_before_after_{timestamp}")
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

    n_samples = 250
    ref_samples = cfg.mu_ref + cfg.sigma_ref * torch.randn(n_samples)
    pdf_x = torch.linspace(cfg.y_min, cfg.y_max, 1000)
    ref_pdf = gaussian_pdf(pdf_x, cfg.mu_ref, cfg.sigma_ref)

    summary = {"reference": {"mu": cfg.mu_ref, "sigma": cfg.sigma_ref}, "scenarios": {}}

    for name, init_mu in _init_scenarios(cfg).items():
        cfg_init = ExperimentConfig(**cfg.__dict__)
        cfg_init.init_mu = init_mu
        cfg_init.init_sigma = cfg.sigma_ref

        dpo_out = train_dpo(y_w, y_l, cfg_init)
        kto_out = train_kto(y, labels, cfg_init)

        dpo_samples = dpo_out["policy"].sample(n_samples).detach().cpu()
        kto_samples = kto_out["policy"].sample(n_samples).detach().cpu()

        dpo_pdf = gaussian_pdf(pdf_x, dpo_out["policy"].mu.item(), dpo_out["policy"].sigma.item())
        kto_pdf = gaussian_pdf(pdf_x, kto_out["policy"].mu.item(), kto_out["policy"].sigma.item())

        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)

        _strip(axes[0, 0], ref_samples, f"Before Training: Reference Samples", "tab:gray", cfg)
        _strip(axes[0, 1], dpo_samples, f"After Training: DPO Samples ({name})", "tab:blue", cfg)
        _strip(axes[0, 2], kto_samples, f"After Training: KTO Samples ({name})", "tab:orange", cfg)

        _hist(axes[1, 0], ref_samples, pdf_x, ref_pdf, "Reference Distribution", "tab:gray", cfg)
        _hist(axes[1, 1], dpo_samples, pdf_x, dpo_pdf, "DPO Distribution", "tab:blue", cfg)
        _hist(axes[1, 2], kto_samples, pdf_x, kto_pdf, "KTO Distribution", "tab:orange", cfg)

        fig.suptitle(f"Sampled Points Before and After Training ({name})", y=1.02)
        fig.tight_layout()

        figure_path = os.path.join(figures_dir, f"sample_before_after_{name}.png")
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        export_report_figure(figure_path, f"single_sample_before_after_{name}.png")

        summary["scenarios"][name] = {
            "init_mu": init_mu,
            "dpo": {"final_mu": dpo_out["policy"].mu.item(), "final_sigma": dpo_out["policy"].sigma.item()},
            "kto": {"final_mu": kto_out["policy"].mu.item(), "final_sigma": kto_out["policy"].sigma.item()},
        }

    save_json(os.path.join(runs_dir, "sample_summary.json"), summary)

    update_latest_paths("sample_before_after", output_root)
    print(f"Saved sample before/after results to: {output_root}")


if __name__ == "__main__":
    main()
