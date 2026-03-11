"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Run the report figure generation suite end-to-end.
"""

import os
import subprocess
import sys


SCRIPTS = [
    "dpo_kto_1d/run_all.py",
    "dpo_kto_1d/density_overlay.py",
    "dpo_kto_1d/reward_plot.py",
    "dpo_kto_1d/reference_sampling.py",
    "dpo_kto_1d/run_entropy_dynamics.py",
    "dpo_kto_1d/run_beta_sweep.py",
    "dpo_kto_1d/data_sensitivity.py",
    "dpo_kto_1d/imbalance_compare.py",
    "dpo_kto_1d/dpo_data_sensitivity.py",
    "dpo_kto_1d/kto_data_sensitivity.py",
    "dpo_kto_1d/init_sensitivity.py",
    "dpo_kto_1d/run_kto_zone_sweep.py",
    "dpo_kto_1d/robustness_single_vs_mixture.py",
    "dpo_kto_mixture_1d/run_all_mix.py",
    "gaussian_mixture_1d/run_mixture_fit.py",
]


def main():
    root = os.path.dirname(__file__)
    for rel_script in SCRIPTS:
        path = os.path.join(root, rel_script)
        print(f"Running {rel_script}")
        subprocess.run([sys.executable, path], check=True)


if __name__ == "__main__":
    main()
