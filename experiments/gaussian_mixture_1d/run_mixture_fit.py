"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Fit a Gaussian mixture with MLE and track parameter recovery.
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.mixture import MixtureConfig, sample_mixture, fit_mixture_mle
from src.plots import plot_mixture_evolution, plot_mixture_fit
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, train_val_split, update_latest_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Fit a Gaussian mixture and track parameter evolution")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = MixtureConfig().with_defaults()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = args.output or os.path.join("results", "gaussian_mixture_1d", timestamp)
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    y = sample_mixture(cfg.target_weights, cfg.target_means, cfg.target_sigmas, cfg.dataset_size, cfg.device)

    train_idx, val_idx = train_val_split(cfg.dataset_size, cfg.val_fraction, cfg.seed)
    y_train = y[train_idx]
    y_val = y[val_idx]

    out = fit_mixture_mle(y_train, y_val, cfg)

    learned = {
        "weights": out["model"].weights().detach().cpu().tolist(),
        "means": out["model"].means.detach().cpu().tolist(),
        "sigmas": out["model"].sigmas().detach().cpu().tolist(),
    }

    target = {
        "weights": cfg.target_weights,
        "means": cfg.target_means,
        "sigmas": cfg.target_sigmas,
    }

    plot_mixture_fit(
        os.path.join(figures_dir, "mixture_fit.png"),
        cfg.y_min,
        cfg.y_max,
        target,
        learned,
    )

    plot_mixture_evolution(
        os.path.join(figures_dir, "mixture_evolution.png"),
        out["history"],
    )

    export_report_figure(
        os.path.join(figures_dir, "mixture_fit.png"),
        "gaussian_mixture_fit.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "mixture_evolution.png"),
        "gaussian_mixture_evolution.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(os.path.join(runs_dir, "history.json"), out["history"])
    save_json(os.path.join(runs_dir, "learned.json"), learned)

    update_latest_paths("gaussian_mixture", output_root)

    print(f"Saved mixture results to: {output_root}")


if __name__ == "__main__":
    main()
