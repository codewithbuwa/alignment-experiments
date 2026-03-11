#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
  echo
  echo "==> $1"
  shift
  "$@"
}

cd "$ROOT_DIR"

run "Single-Gaussian core: run_all" \
  python3 experiments/dpo_kto_1d/run_all.py

run "Single-Gaussian core: density overlay" \
  python3 experiments/dpo_kto_1d/density_overlay.py

run "Single-Gaussian core: reward plot" \
  python3 experiments/dpo_kto_1d/reward_plot.py

run "Single-Gaussian core: entropy dynamics" \
  python3 experiments/dpo_kto_1d/run_entropy_dynamics.py

run "KTO balanced" \
  python3 experiments/dpo_kto_1d/run_kto_balanced.py

run "KTO imbalanced" \
  python3 experiments/dpo_kto_1d/run_kto_imbalanced.py

run "KTO data sensitivity" \
  python3 experiments/dpo_kto_1d/kto_data_sensitivity.py

run "DPO data sensitivity" \
  python3 experiments/dpo_kto_1d/dpo_data_sensitivity.py

run "Single imbalance compare" \
  python3 experiments/dpo_kto_1d/imbalance_compare.py

run "Single supervision sweep" \
  python3 experiments/dpo_kto_1d/data_sensitivity.py

run "KTO zone sweep" \
  python3 experiments/dpo_kto_1d/run_kto_zone_sweep.py

run "Beta sweep" \
  python3 experiments/dpo_kto_1d/run_beta_sweep.py

run "Single reference sampling" \
  python3 experiments/dpo_kto_1d/reference_sampling.py

run "Single init sensitivity" \
  python3 experiments/dpo_kto_1d/init_sensitivity.py

run "Mixture suite" \
  python3 experiments/dpo_kto_mixture_1d/run_all_mix.py

run "Mixture density overlay" \
  python3 experiments/dpo_kto_mixture_1d/density_overlay_mix.py

run "Mixture imbalance compare" \
  python3 experiments/dpo_kto_mixture_1d/imbalance_compare_mix.py

run "Cross-family robustness" \
  python3 experiments/dpo_kto_1d/robustness_single_vs_mixture.py --alphas 0.1,0.3,0.5,0.7,0.9 --n-components 2

run "Gaussian mixture MLE" \
  python3 experiments/gaussian_mixture_1d/run_mixture_fit.py

run "Full report suite" \
  python3 experiments/run_report_suite.py

echo
echo "Latest outputs:"
cat results/_latest_paths.txt
