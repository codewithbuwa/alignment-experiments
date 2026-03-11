import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig
from src.data import make_dpo_pairs
from src.train import train_dpo
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def main():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"dpo_data_sensitivity_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    y_w, y_l = make_dpo_pairs(
        cfg.mu_ref,
        cfg.sigma_ref,
        cfg.target,
        cfg.dataset_size,
        cfg.device,
        good_ratio=0.5,
        zone_half_width=cfg.zone_half_width,
    )
    bal_out = train_dpo(y_w, y_l, cfg)

    y_w, y_l = make_dpo_pairs(
        cfg.mu_ref,
        cfg.sigma_ref,
        cfg.target,
        cfg.dataset_size,
        cfg.device,
        good_ratio=0.1,
        zone_half_width=cfg.zone_half_width,
    )
    imb_out = train_dpo(y_w, y_l, cfg)

    plt.figure()
    plt.plot(bal_out["history"]["sigma"], label="Balanced (50/50)")
    plt.plot(imb_out["history"]["sigma"], label="Imbalanced (10/90)", linestyle="dashed")
    plt.xlabel("Training Step")
    plt.ylabel("Sigma")
    plt.title("Data Sensitivity Test (DPO)")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "dpo_data_sensitivity.png"))
    plt.close()

    export_report_figure(
        os.path.join(figures_dir, "dpo_data_sensitivity.png"),
        "single_dpo_data_sensitivity.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(os.path.join(runs_dir, "dpo_balanced_history.json"), bal_out["history"])
    save_json(os.path.join(runs_dir, "dpo_imbalanced_history.json"), imb_out["history"])

    update_latest_paths("dpo_data_sensitivity", output_root)

    print(f"Saved DPO data sensitivity results to: {output_root}")


if __name__ == "__main__":
    main()
