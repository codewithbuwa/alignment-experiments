"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Plot final implicit reward curves and oracle comparison for single Gaussians.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig
from src.data import make_dpo_pairs, make_kto_samples
from src.losses import implicit_reward
from src.train import train_dpo, train_kto
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def plot_implicit_reward(y_vals, r_vals, title, output_path):
    plt.figure()
    plt.plot(y_vals.numpy(), r_vals.numpy())
    plt.title(title)
    plt.xlabel("y")
    plt.ylabel("r(y)")
    plt.savefig(output_path)
    plt.close()


def plot_reward_comparison(y_vals, dpo_r, kto_r, target, output_path):
    oracle_reward = -(y_vals - target).abs()
    plt.figure(figsize=(10, 6))
    plt.plot(y_vals.numpy(), dpo_r.numpy(), label=f"DPO reward: max = {torch.max(dpo_r).item():.2f}")
    plt.plot(y_vals.numpy(), kto_r.numpy(), label=f"KTO reward: max = {torch.max(kto_r).item():.2f}")
    plt.plot(y_vals.numpy(), oracle_reward.numpy(), label="Oracle reward")
    plt.title("Reward comparison")
    plt.xlabel("y")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def main():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_1d", f"reward_plot_{timestamp}")
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

    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 10000)
    dpo_r = implicit_reward(y_vals, dpo_out["policy"].mu, dpo_out["policy"].rho, cfg.mu_ref, cfg.sigma_ref, cfg.beta)
    kto_r = implicit_reward(y_vals, kto_out["policy"].mu, kto_out["policy"].rho, cfg.mu_ref, cfg.sigma_ref, cfg.beta)

    plot_implicit_reward(y_vals, dpo_r.detach(), "DPO Implicit Reward", os.path.join(figures_dir, "dpo_implicit_reward.png"))
    plot_implicit_reward(y_vals, kto_r.detach(), "KTO Implicit Reward", os.path.join(figures_dir, "kto_implicit_reward.png"))
    plot_reward_comparison(
        y_vals,
        dpo_r.detach(),
        kto_r.detach(),
        cfg.target,
        os.path.join(figures_dir, "reward_comparison.png"),
    )
    export_report_figure(
        os.path.join(figures_dir, "dpo_implicit_reward.png"),
        "single_dpo_implicit_reward.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "kto_implicit_reward.png"),
        "single_kto_implicit_reward.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "reward_comparison.png"),
        "reward.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "reward_summary.json"),
        {
            "dpo_final_mu": dpo_out["history"]["mu"][-1],
            "dpo_final_sigma": dpo_out["history"]["sigma"][-1],
            "kto_final_mu": kto_out["history"]["mu"][-1],
            "kto_final_sigma": kto_out["history"]["sigma"][-1],
            "train_eval_splits": {
                "dpo": dpo_out["splits"],
                "kto": kto_out["splits"],
            },
        },
    )

    update_latest_paths("reward_plot", output_root)

    print(f"Saved reward plot results to: {output_root}")


if __name__ == "__main__":
    main()
