"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Run the main Gaussian-mixture DPO and KTO experiment suite.
"""

import os
import subprocess
import sys


SCRIPTS = [
    "density_overlay_mix.py",
    "imbalance_compare_mix.py",
    "entropy_dynamics_mix.py",
    "ref_sampling_mix.py",
    "reward_plot_mix.py",
    "init_sensitivity_mix.py",
    "component_evolution_mix.py",
]


def main():
    root = os.path.dirname(__file__)
    for script in SCRIPTS:
        path = os.path.join(root, script)
        print(f"Running {script}")
        subprocess.run([sys.executable, path], check=True)


if __name__ == "__main__":
    main()
