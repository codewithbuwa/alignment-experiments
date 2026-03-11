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
    output_root = os.path.join("results", "dpo_kto_1d", f"beta_sweep_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    dpo_sigmas = []
    kto_sigmas = []

    for beta in cfg.beta_sweep:
        cfg_beta = ExperimentConfig(**cfg.__dict__)
        cfg_beta.beta = beta

        y_w, y_l = make_dpo_pairs(cfg.mu_ref, cfg.sigma_ref, cfg.target, cfg.dataset_size, cfg.device)
        dpo_out = train_dpo(y_w, y_l, cfg_beta)

        y, labels = make_kto_samples(
            cfg.mu_ref,
            cfg.sigma_ref,
            cfg.target,
            cfg.zone_half_width,
            cfg.dataset_size,
            cfg.kto_good_fraction,
            cfg.device,
        )
        kto_out = train_kto(y, labels, cfg_beta)

        dpo_sigmas.append(dpo_out["policy"].sigma.item())
        kto_sigmas.append(kto_out["policy"].sigma.item())

    plt.figure()
    plt.plot(cfg.beta_sweep, dpo_sigmas, marker="o", label="DPO")
    plt.plot(cfg.beta_sweep, kto_sigmas, marker="o", label="KTO")
    plt.xlabel("Beta")
    plt.ylabel("Final Sigma")
    plt.legend()
    plt.title("Final Sigma vs Beta")
    plt.savefig(os.path.join(figures_dir, "sigma_vs_beta.png"))
    plt.close()

    export_report_figure(
        os.path.join(figures_dir, "sigma_vs_beta.png"),
        "single_beta_sweep.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)

    update_latest_paths("beta_sweep", output_root)

    print(f"Saved beta sweep results to: {output_root}")


if __name__ == "__main__":
    main()
