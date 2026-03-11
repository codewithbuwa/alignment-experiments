import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig
from src.data import make_dpo_pairs, make_kto_samples
from src.train import train_dpo, train_kto
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def main():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"data_sensitivity_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    dpo_sigmas = []
    kto_sigmas = []

    for alpha in cfg.data_sensitivity_alphas:
        y_w, y_l = make_dpo_pairs(
            cfg.mu_ref,
            cfg.sigma_ref,
            cfg.target,
            cfg.dataset_size,
            cfg.device,
            good_ratio=alpha,
            zone_half_width=cfg.zone_half_width,
        )
        dpo_out = train_dpo(y_w, y_l, cfg)
        dpo_sigmas.append(dpo_out["policy"].sigma.item())

        y, labels = make_kto_samples(
            cfg.mu_ref,
            cfg.sigma_ref,
            cfg.target,
            cfg.zone_half_width,
            cfg.dataset_size,
            cfg.kto_good_fraction,
            cfg.device,
            delta=1.0,
            good_ratio=alpha,
        )
        kto_out = train_kto(y, labels, cfg)
        kto_sigmas.append(kto_out["policy"].sigma.item())

    plt.figure()
    plt.plot(cfg.data_sensitivity_alphas, dpo_sigmas, marker="o", label="DPO")
    plt.plot(cfg.data_sensitivity_alphas, kto_sigmas, marker="o", label="KTO")
    plt.xlabel("Supervision Strength (alpha)")
    plt.ylabel("Final Sigma")
    plt.title("DPO vs KTO Sensitivity Grid")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "data_sensitivity.png"))
    plt.close()

    export_report_figure(
        os.path.join(figures_dir, "data_sensitivity.png"),
        "single_data_sensitivity.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)

    update_latest_paths("data_sensitivity", output_root)

    print(f"Saved data sensitivity results to: {output_root}")


if __name__ == "__main__":
    main()
