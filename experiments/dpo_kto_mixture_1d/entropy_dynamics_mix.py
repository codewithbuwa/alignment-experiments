import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import MixtureDPOKTOConfig, ExperimentConfig
from src.ref_policies import make_reference_mixture
from src.train_mix import train_dpo_mixture, train_kto_mixture
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
    cfg = MixtureDPOKTOConfig()
    base_cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_mixture_1d", f"entropy_dynamics_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    ref_policy = make_reference_mixture(cfg.n_components, base_cfg.mu_ref, base_cfg.sigma_ref, cfg.device)

    _, dpo_sigmas, dpo_hist, _ = train_dpo_mixture(ref_policy, cfg)
    _, kto_sigmas, kto_hist, _ = train_kto_mixture(ref_policy, cfg)

    quartiles = _quartile_indices(len(dpo_sigmas))

    plt.figure()
    for i in range(cfg.n_components):
        series = [s[i] for s in dpo_sigmas]
        plt.plot(series, label=f"DPO sigma_{i}")
        for idx in quartiles:
            plt.scatter(idx, series[idx], s=18)
    for i in range(cfg.n_components):
        series = [s[i] for s in kto_sigmas]
        plt.plot(series, label=f"KTO sigma_{i}", linestyle="--")
        for idx in quartiles:
            plt.scatter(idx, series[idx], s=18)

    for idx in quartiles:
        plt.axvline(idx, color="gray", alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("Sigma (per component)")
    plt.legend(ncol=2)
    plt.title("Mixture Entropy Dynamics")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "mixture_entropy_dynamics.png"))
    plt.close()

    export_report_figure(
        os.path.join(figures_dir, "mixture_entropy_dynamics.png"),
        "mix_entropy_dynamics.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(
        os.path.join(runs_dir, "entropy_quartiles.json"),
        {
            "quartile_steps": quartiles,
            "dpo_sigmas": [[dpo_sigmas[i][k] for i in quartiles] for k in range(cfg.n_components)],
            "kto_sigmas": [[kto_sigmas[i][k] for i in quartiles] for k in range(cfg.n_components)],
            "dpo": dpo_hist,
            "kto": kto_hist,
        },
    )

    update_latest_paths("mix_entropy_dynamics", output_root)

    print(f"Saved mixture entropy dynamics results to: {output_root}")


if __name__ == "__main__":
    main()
