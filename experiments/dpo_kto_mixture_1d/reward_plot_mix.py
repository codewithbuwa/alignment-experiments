import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import MixtureDPOKTOConfig, ExperimentConfig
from src.ref_policies import make_reference_mixture
from src.train_mix import train_dpo_mixture, train_kto_mixture
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def plot_implicit_reward(y_vals, r_vals, title, output_path):
    plt.figure()
    plt.plot(y_vals.numpy(), r_vals.numpy())
    plt.title(title)
    plt.xlabel("y")
    plt.ylabel("r(y)")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()


def main():
    cfg = MixtureDPOKTOConfig()
    base_cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_mixture_1d", f"reward_plot_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    ref_policy = make_reference_mixture(cfg.n_components, base_cfg.mu_ref, base_cfg.sigma_ref, cfg.device)

    dpo_policy, _, dpo_hist, dpo_splits = train_dpo_mixture(ref_policy, cfg)
    kto_policy, _, kto_hist, kto_splits = train_kto_mixture(ref_policy, cfg)

    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)
    ref_logp = ref_policy.log_prob(y_vals)

    dpo_r = cfg.beta * (dpo_policy.log_prob(y_vals) - ref_logp).detach()
    kto_r = cfg.beta * (kto_policy.log_prob(y_vals) - ref_logp).detach()

    plot_implicit_reward(
        y_vals,
        dpo_r,
        "DPO Mixture Implicit Reward",
        os.path.join(figures_dir, "dpo_mixture_implicit_reward.png"),
    )

    plot_implicit_reward(
        y_vals,
        kto_r,
        "KTO Mixture Implicit Reward",
        os.path.join(figures_dir, "kto_mixture_implicit_reward.png"),
    )
    export_report_figure(
        os.path.join(figures_dir, "dpo_mixture_implicit_reward.png"),
        "mix_dpo_implicit_reward.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "kto_mixture_implicit_reward.png"),
        "mix_kto_implicit_reward.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "reward_summary.json"),
        {
            "dpo_final": {
                "mus": dpo_hist["mus"][-1],
                "sigmas": dpo_hist["sigmas"][-1],
                "logits": dpo_hist["logits"][-1],
            },
            "kto_final": {
                "mus": kto_hist["mus"][-1],
                "sigmas": kto_hist["sigmas"][-1],
                "logits": kto_hist["logits"][-1],
            },
            "train_eval_splits": {
                "dpo": dpo_splits,
                "kto": kto_splits,
            },
        },
    )

    update_latest_paths("mix_reward_plot", output_root)

    print(f"Saved mixture reward plot results to: {output_root}")


if __name__ == "__main__":
    main()
