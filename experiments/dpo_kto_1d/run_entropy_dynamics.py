import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig
from src.data import make_dpo_pairs, make_kto_samples
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


def main():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"entropy_dynamics_{timestamp}")
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

    dpo_sigmas = dpo_out["history"]["sigma"]
    kto_sigmas = kto_out["history"]["sigma"]
    quartiles = _quartile_indices(len(dpo_sigmas))

    plt.figure()
    plt.plot(dpo_sigmas, label="DPO")
    plt.plot(kto_sigmas, label="KTO")
    for idx in quartiles:
        plt.axvline(idx, color="gray", alpha=0.2)
        plt.scatter(idx, dpo_sigmas[idx], color="tab:blue", s=20)
        plt.scatter(idx, kto_sigmas[idx], color="tab:orange", s=20)

    plt.xlabel("Training Step")
    plt.ylabel("Sigma")
    plt.title("Entropy Dynamics")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "entropy_dynamics.png"))
    plt.close()

    export_report_figure(
        os.path.join(figures_dir, "entropy_dynamics.png"),
        "single_entropy_dynamics.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "entropy_quartiles.json"),
        {
            "quartile_steps": quartiles,
            "dpo_sigmas": [dpo_sigmas[i] for i in quartiles],
            "kto_sigmas": [kto_sigmas[i] for i in quartiles],
        },
    )

    update_latest_paths("entropy_dynamics", output_root)

    print(f"Saved entropy dynamics results to: {output_root}")


if __name__ == "__main__":
    main()
