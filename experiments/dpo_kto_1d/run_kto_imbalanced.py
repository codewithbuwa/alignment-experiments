"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Run the imbalanced single-Gaussian KTO experiment.
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig
from src.data import make_kto_samples
from src.train import train_kto
from src.utils import ensure_dir, get_timestamp, save_json, set_seed, update_latest_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Run KTO experiment (imbalanced)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig()
    cfg.kto_good_fraction = 0.1
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = args.output or os.path.join("results", "dpo_kto_1d", f"kto_imbalanced_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

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

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(os.path.join(runs_dir, "kto_history.json"), kto_out["history"])

    update_latest_paths("kto_imbalanced", output_root)

    print(f"Saved KTO imbalanced results to: {output_root}")


if __name__ == "__main__":
    main()
