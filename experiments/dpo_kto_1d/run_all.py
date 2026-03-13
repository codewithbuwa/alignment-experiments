"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Generate the main single-Gaussian DPO vs KTO figures.
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig
from src.data import make_dpo_pairs, make_kto_samples
from src.plots import plot_entropy_comparison, plot_main_panels, plot_zone_sweep
from src.train import train_dpo, train_kto, config_to_dict
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Run DPO vs KTO 1D experiments")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--zone-half-width", type=float, default=None)
    parser.add_argument("--kto-gamma", type=float, default=None)
    parser.add_argument("--kl-mode", type=str, default=None)
    parser.add_argument("--kl-grad", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig()

    if args.steps is not None:
        cfg.steps = args.steps
    if args.dataset_size is not None:
        cfg.dataset_size = args.dataset_size
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.zone_half_width is not None:
        cfg.zone_half_width = args.zone_half_width
    if args.kto_gamma is not None:
        cfg.kto_gamma = args.kto_gamma
    if args.kl_mode is not None:
        cfg.kl_mode = args.kl_mode
    if args.kl_grad:
        cfg.kl_grad = True

    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = args.output or os.path.join("results", "dpo_kto_1d", timestamp)
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    # DPO dataset
    y_w, y_l = make_dpo_pairs(cfg.mu_ref, cfg.sigma_ref, cfg.target, cfg.dataset_size, cfg.device)
    dpo_out = train_dpo(y_w, y_l, cfg)

    # KTO balanced
    kto_bal_cfg = ExperimentConfig(**config_to_dict(cfg))
    kto_bal_cfg.kto_good_fraction = 0.5
    y_bal, labels_bal = make_kto_samples(
        cfg.mu_ref,
        cfg.sigma_ref,
        cfg.target,
        cfg.zone_half_width,
        cfg.dataset_size,
        kto_bal_cfg.kto_good_fraction,
        cfg.device,
    )
    kto_bal_out = train_kto(y_bal, labels_bal, kto_bal_cfg)

    # KTO imbalanced
    kto_imb_cfg = ExperimentConfig(**config_to_dict(cfg))
    kto_imb_cfg.kto_good_fraction = 0.1
    y_imb, labels_imb = make_kto_samples(
        cfg.mu_ref,
        cfg.sigma_ref,
        cfg.target,
        cfg.zone_half_width,
        cfg.dataset_size,
        kto_imb_cfg.kto_good_fraction,
        cfg.device,
    )
    kto_imb_out = train_kto(y_imb, labels_imb, kto_imb_cfg)

    # Plots
    plot_main_panels(
        os.path.join(figures_dir, "main_panels.png"),
        cfg.y_min,
        cfg.y_max,
        cfg.mu_ref,
        cfg.sigma_ref,
        dpo_out["policy"],
        kto_bal_out["policy"],
        dpo_out["history"],
        kto_bal_out["history"],
    )

    plot_entropy_comparison(
        os.path.join(figures_dir, "entropy_sensitivity.png"),
        dpo_out["history"],
        kto_bal_out["history"],
        kto_imb_out["history"],
    )

    export_report_figure(
        os.path.join(figures_dir, "main_panels.png"),
        "single_main_panels.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "entropy_sensitivity.png"),
        "single_sigma_sensitivity.png",
    )

    # Zone sweep
    epsilons = []
    final_sigmas = []
    for eps in cfg.zone_sweep:
        sweep_cfg = ExperimentConfig(**config_to_dict(cfg))
        sweep_cfg.zone_half_width = eps
        sweep_cfg.kto_good_fraction = 0.5

        y_sweep, labels_sweep = make_kto_samples(
            cfg.mu_ref,
            cfg.sigma_ref,
            cfg.target,
            eps,
            cfg.dataset_size,
            sweep_cfg.kto_good_fraction,
            cfg.device,
        )
        sweep_out = train_kto(y_sweep, labels_sweep, sweep_cfg)

        epsilons.append(eps)
        final_sigmas.append(sweep_out["policy"].sigma.item())

    plot_zone_sweep(
        os.path.join(figures_dir, "kto_zone_sweep.png"),
        epsilons,
        final_sigmas,
    )

    export_report_figure(
        os.path.join(figures_dir, "kto_zone_sweep.png"),
        "single_kto_zone_sweep.png",
    )

    # Save artifacts
    save_json(os.path.join(output_root, "config.json"), config_to_dict(cfg))

    save_json(
        os.path.join(runs_dir, "dpo_history.json"),
        dpo_out["history"],
    )
    save_json(
        os.path.join(runs_dir, "kto_balanced_history.json"),
        kto_bal_out["history"],
    )
    save_json(
        os.path.join(runs_dir, "kto_imbalanced_history.json"),
        kto_imb_out["history"],
    )

    save_json(
        os.path.join(runs_dir, "kto_zone_sweep.json"),
        {"epsilons": epsilons, "final_sigmas": final_sigmas},
    )

    update_latest_paths("main_all", output_root)

    print(f"Saved results to: {output_root}")


if __name__ == "__main__":
    main()
