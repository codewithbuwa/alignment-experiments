import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig
from src.data import make_dpo_pairs
from src.train import train_dpo
from src.utils import ensure_dir, get_timestamp, save_json, set_seed, update_latest_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Run DPO experiment")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = args.output or os.path.join("results", "dpo_kto_1d", f"dpo_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    y_w, y_l = make_dpo_pairs(cfg.mu_ref, cfg.sigma_ref, cfg.target, cfg.dataset_size, cfg.device)
    dpo_out = train_dpo(y_w, y_l, cfg)

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(os.path.join(runs_dir, "dpo_history.json"), dpo_out["history"])

    update_latest_paths("dpo", output_root)

    print(f"Saved DPO results to: {output_root}")


if __name__ == "__main__":
    main()
