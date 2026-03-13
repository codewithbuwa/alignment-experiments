"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Compare balanced and imbalanced KTO sigma dynamics.
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig
from src.data import make_kto_samples
from src.train import train_kto
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def main():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"kto_data_sensitivity_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    y_bal, labels_bal = make_kto_samples(
        cfg.mu_ref,
        cfg.sigma_ref,
        cfg.target,
        cfg.zone_half_width,
        cfg.dataset_size,
        cfg.kto_good_fraction,
        cfg.device,
        delta=1.0,
        good_ratio=0.5,
    )
    bal_out = train_kto(y_bal, labels_bal, cfg)

    y_imb, labels_imb = make_kto_samples(
        cfg.mu_ref,
        cfg.sigma_ref,
        cfg.target,
        cfg.zone_half_width,
        cfg.dataset_size,
        cfg.kto_good_fraction,
        cfg.device,
        delta=1.0,
        good_ratio=0.1,
    )
    imb_out = train_kto(y_imb, labels_imb, cfg)

    plt.figure()
    plt.plot(bal_out["history"]["sigma"], label="Balanced (50/50)")
    plt.plot(imb_out["history"]["sigma"], label="Imbalanced (10/90)")
    plt.xlabel("Training Step")
    plt.ylabel("Sigma")
    plt.title("Data Sensitivity Test")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "kto_data_sensitivity.png"))
    plt.close()

    export_report_figure(
        os.path.join(figures_dir, "kto_data_sensitivity.png"),
        "single_kto_data_sensitivity.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)

    update_latest_paths("kto_data_sensitivity", output_root)

    print(f"Saved KTO data sensitivity results to: {output_root}")


if __name__ == "__main__":
    main()
