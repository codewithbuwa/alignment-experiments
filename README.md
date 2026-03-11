# DPO vs KTO 1D Experiments

This workspace compares Direct Preference Optimization (DPO) and Kahneman-Tversky Optimization (KTO) in a 1D scalar world, then extends the same logic to Gaussian mixtures.

It is organized to match the structure and experimental intent of `Alignment_first_step.pdf`.

## Overview

- Policy family:
  Gaussian `N(mu, sigma^2)` with learnable `mu` and `rho = log(sigma)`.
- Reference policy:
  `N(mu=5, sigma=2)`.
- Oracle reward:
  `R(y) = -|y - 7|`.
- DPO data:
  sample pairs from the reference; the winner is the point closer to `7`.
- KTO data:
  sample single points from the reference; label as good if `y` falls inside the desirable zone.

## Repository Structure

- `src/`
  Core logic: configs, data generation, policies, losses, training, plots.
- `experiments/dpo_kto_1d/`
  Single-Gaussian DPO/KTO experiments.
- `experiments/dpo_kto_mixture_1d/`
  Mixture DPO/KTO experiments.
- `experiments/gaussian_mixture_1d/`
  Auxiliary Gaussian-mixture MLE experiment.
- `results/`
  Timestamped experiment outputs.
- `report/`
  Stable figures and LaTeX report.

Default split policy:
- DPO and KTO datasets are split into train/eval by default using `eval_fraction = 0.2`.
- Training uses the train split only.
- Held-out inference/evaluation loss is recorded in histories as `eval_loss`.

Deprecated folders:
- `experiments_single/`
- `experiment_mix/`

Their functionality has been replicated in `experiments/`.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

If you only want the main results:

1. Run the main single-Gaussian suite:

```bash
python3 experiments/dpo_kto_1d/run_all.py
```

2. Run the full report/presentation figure suite:

```bash
python3 experiments/run_report_suite.py
```

3. Check the latest run locations:

```bash
cat results/_latest_paths.txt
```

Full end-to-end run:

```bash
bash run_all_experiments.sh
```

## Command Blocks

Single-Gaussian core runs:

```bash
python3 experiments/dpo_kto_1d/run_all.py
python3 experiments/dpo_kto_1d/density_overlay.py
python3 experiments/dpo_kto_1d/reward_plot.py
python3 experiments/dpo_kto_1d/run_entropy_dynamics.py
```

Single-Gaussian KTO and imbalance runs:

```bash
python3 experiments/dpo_kto_1d/run_kto_balanced.py
python3 experiments/dpo_kto_1d/run_kto_imbalanced.py
python3 experiments/dpo_kto_1d/kto_data_sensitivity.py
python3 experiments/dpo_kto_1d/dpo_data_sensitivity.py
python3 experiments/dpo_kto_1d/imbalance_compare.py
python3 experiments/dpo_kto_1d/data_sensitivity.py
python3 experiments/dpo_kto_1d/run_kto_zone_sweep.py
python3 experiments/dpo_kto_1d/run_beta_sweep.py
python3 experiments/dpo_kto_1d/reference_sampling.py
python3 experiments/dpo_kto_1d/init_sensitivity.py
```

Mixture runs:

```bash
python3 experiments/dpo_kto_mixture_1d/run_all_mix.py
python3 experiments/dpo_kto_mixture_1d/density_overlay_mix.py
python3 experiments/dpo_kto_mixture_1d/imbalance_compare_mix.py
python3 experiments/dpo_kto_mixture_1d/entropy_dynamics_mix.py
python3 experiments/dpo_kto_mixture_1d/ref_sampling_mix.py
python3 experiments/dpo_kto_mixture_1d/reward_plot_mix.py
python3 experiments/dpo_kto_mixture_1d/init_sensitivity_mix.py
python3 experiments/dpo_kto_mixture_1d/component_evolution_mix.py
```

Cross-family and auxiliary runs:

```bash
python3 experiments/dpo_kto_1d/robustness_single_vs_mixture.py --alphas 0.1,0.3,0.5,0.7,0.9 --n-components 2
python3 experiments/gaussian_mixture_1d/run_mixture_fit.py
python3 experiments/run_report_suite.py
cat results/_latest_paths.txt
```

## Step-by-Step Experiment Guide

### Part 1: Core Single-Gaussian Story

1. Main three-panel result: density, implicit reward, entropy

```bash
python3 experiments/dpo_kto_1d/run_all.py
```

Use this first. It gives the main DPO vs KTO comparison aligned with the PDF.

Primary outputs:
- `results/dpo_kto_1d/<timestamp>/figures/main_panels.png`
- `results/dpo_kto_1d/<timestamp>/figures/entropy_sensitivity.png`
- `results/dpo_kto_1d/<timestamp>/figures/kto_zone_sweep.png`

2. Density evolution over training quartiles

```bash
python3 experiments/dpo_kto_1d/density_overlay.py
```

Question answered:
- How do DPO and KTO densities move at 0/25/50/75/100% training?

3. Final implicit reward and oracle comparison

```bash
python3 experiments/dpo_kto_1d/reward_plot.py
```

Question answered:
- How does learned implicit reward compare to the oracle reward landscape?

4. Entropy dynamics only

```bash
python3 experiments/dpo_kto_1d/run_entropy_dynamics.py
```

Question answered:
- Does DPO collapse entropy while KTO stabilizes?

### Part 2: KTO-Specific Checks

5. Balanced KTO run only

```bash
python3 experiments/dpo_kto_1d/run_kto_balanced.py
```

6. Imbalanced KTO run only

```bash
python3 experiments/dpo_kto_1d/run_kto_imbalanced.py
```

7. KTO data sensitivity: balanced vs imbalanced trajectories

```bash
python3 experiments/dpo_kto_1d/kto_data_sensitivity.py
```

Question answered:
- How robust is KTO when good feedback is scarce?

8. KTO zone-width sweep

```bash
python3 experiments/dpo_kto_1d/run_kto_zone_sweep.py
```

Question answered:
- Does final `sigma` scale with zone width?
- How does the final density change as the desirable zone expands?

### Part 3: DPO/KTO Sensitivity Sweeps

9. DPO vs KTO supervision-strength sweep

```bash
python3 experiments/dpo_kto_1d/data_sensitivity.py
```

Question answered:
- How do DPO and KTO respond as supervision strength `alpha` changes?

10. DPO-only balanced vs imbalanced comparison

```bash
python3 experiments/dpo_kto_1d/dpo_data_sensitivity.py
```

11. Dedicated DPO/KTO imbalance comparison (`10%` vs `50%`)

```bash
python3 experiments/dpo_kto_1d/imbalance_compare.py
```

Question answered:
- How do entropy, final distributions, and parameter dynamics differ between `10%` and `50%` supervision?
- Are DPO and KTO reacting differently under the same imbalance level?

12. Beta sweep

```bash
python3 experiments/dpo_kto_1d/run_beta_sweep.py
```

Question answered:
- How sensitive is final concentration to `beta`?

13. KL-mode comparison for KTO

```bash
python3 experiments/dpo_kto_1d/reference_sampling.py
```

Question answered:
- How much does KTO behavior depend on how `ref_KL` is estimated?

14. Initialization sensitivity

```bash
python3 experiments/dpo_kto_1d/init_sensitivity.py
```

Question answered:
- How do DPO and KTO behave when initialized around, left, or right of the target?
- What are the parameter values at each milestone panel?

### Part 4: Mixture DPO/KTO Experiments

15. Run all mixture experiments

```bash
python3 experiments/dpo_kto_mixture_1d/run_all_mix.py
```

This is the mixture counterpart of `run_all.py`.

16. Mixture density evolution

```bash
python3 experiments/dpo_kto_mixture_1d/density_overlay_mix.py
```

17. Dedicated mixture imbalance comparison (`10%` vs `50%`)

```bash
python3 experiments/dpo_kto_mixture_1d/imbalance_compare_mix.py
```

Question answered:
- How do mixture entropy, final densities, and component parameters differ between `10%` and `50%` imbalance?
- Which components absorb the imbalance under DPO versus KTO?

18. Mixture entropy dynamics per component

```bash
python3 experiments/dpo_kto_mixture_1d/entropy_dynamics_mix.py
```

19. Mixture KL-mode comparison

```bash
python3 experiments/dpo_kto_mixture_1d/ref_sampling_mix.py
```

20. Mixture reward plots

```bash
python3 experiments/dpo_kto_mixture_1d/reward_plot_mix.py
```

21. Mixture initialization sensitivity

```bash
python3 experiments/dpo_kto_mixture_1d/init_sensitivity_mix.py
```

Question answered:
- How do left/right/around initializations affect mixture learning?
- What are the component parameter values at each milestone panel?

22. Mixture component evolution and 25% milestone snapshots

```bash
python3 experiments/dpo_kto_mixture_1d/component_evolution_mix.py
```

Question answered:
- How do individual components move, sharpen, and reweight over training?

### Part 5: Cross-Family Robustness

23. Single Gaussian vs 2-component mixture robustness

```bash
python3 experiments/dpo_kto_1d/robustness_single_vs_mixture.py --alphas 0.1,0.3,0.5,0.7,0.9 --n-components 2
```

Question answered:
- How do single and mixture policies differ under the same supervision-ratio sweep?
- For mixtures, how does each component evolve separately?

### Part 6: Auxiliary Mixture-Fitting Experiment

24. Gaussian-mixture MLE

```bash
python3 experiments/gaussian_mixture_1d/run_mixture_fit.py
```

Question answered:
- Can standard MLE recover a target mixture, and how do weights/means/sigmas evolve?

## One-Command Figure Generation

To regenerate the full report/presentation figure set:

```bash
python3 experiments/run_report_suite.py
```

This populates:
- `results/` with timestamped runs
- `results/_latest_paths.txt` with the latest run directory for each experiment label
- `report/figures/` with stable figure names used by:
  `report/report.tex`
  `presentation.tex`

## Where to Look After Running

- Latest experiment outputs:
  `results/_latest_paths.txt`
- Report source:
  `report/report.tex`
- Presentation source:
  `presentation.tex`
- Stable exported figures:
  `report/figures/`

## Notes on KL Handling for KTO

The PDF explicitly warns that the `ref_KL` term is a decision-boundary shift and must be defined operationally.

Configuration lives in `src/config.py`:
- `kl_mode = analytic | fixed | running | batch`
- `kl_grad`
- `kl_fixed`
- `kl_ema_decay`
- `eval_fraction`

Default behavior:
- single Gaussian KTO uses analytic KL by default
- mixture KTO uses batch-estimated KL by default

## Recommended Run Order

If you want a disciplined reading order:

1. `run_all.py`
2. `density_overlay.py`
3. `reward_plot.py`
4. `run_entropy_dynamics.py`
5. `run_kto_zone_sweep.py`
6. `reference_sampling.py`
7. `data_sensitivity.py`
8. `init_sensitivity.py`
9. `run_all_mix.py`
10. `component_evolution_mix.py`
11. `robustness_single_vs_mixture.py`
12. `run_mixture_fit.py`
