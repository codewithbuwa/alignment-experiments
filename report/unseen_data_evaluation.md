# Unseen-Data Evaluation Report

## Objective

This report documents how the project now checks whether trained DPO and KTO policies behave consistently on newly generated data that was not used during training.

## What Is Being Tested

For each comparison run, the workflow now evaluates three stages:

1. `train`
   The final training objective value recorded on the training split.
2. `eval`
   The held-out objective value recorded on the internal `20%` evaluation split.
3. `fresh`
   A newly generated dataset sampled after training from the same data-generation process and evaluated with the trained policy.

The goal is to verify that conclusions about DPO and KTO are not artifacts of a single sampled dataset.

## Data Policy

### Single-Gaussian DPO

- Training pairs are generated with the current DPO rule:
  - if `good_ratio` is set, compute `kappa = 1 - good_ratio`
  - exclude candidate points inside `|y - 7| < kappa * sigma_ref`
  - form preference pairs from the retained points
  - winner is whichever point is closer to `7`
- Fresh-test pairs are generated with the exact same rule, but from newly sampled points.

### Single-Gaussian KTO

- Training samples are labeled by whether they lie in the desirable zone.
- Fresh-test samples are generated with the same zone and same class-ratio setting.

### Mixture DPO/KTO

- The same logic is reused, but the samples come from the reference mixture.
- Fresh-test data is drawn after training from the same mixture-based generator.

## Metrics

### DPO

On fresh unseen data, the scripts now report:

- `loss`
- `margin`
  Mean log-probability margin between winners and losers.
- `mean_reward`
  Mean oracle reward of winners.
- `effective_good_mass`
  Fraction of winners that fall inside the desirable zone.

### KTO

On fresh unseen data, the scripts now report:

- `loss`
- `mean_reward`
  Mean oracle reward on fresh labeled samples.
- `effective_good_mass`
  Positive-label fraction on fresh data for single-Gaussian KTO, or desirable-zone mass proxy for the mixture setting.

## New Figures

The comparison scripts now export generalization figures:

- `report/figures/single_imbalance_generalization.png`
- `report/figures/mix_imbalance_generalization.png`

Each figure summarizes:

- train loss
- eval loss
- fresh-test loss
- train effective-good-mass
- fresh effective-good-mass

## Updated Experiment Entry Points

### Single-Gaussian comparison

```bash
python3 experiments/dpo_kto_1d/imbalance_compare.py
```

Default controls:

- DPO: `good_ratio = 0.1, 1.0`
- KTO: `alpha = 0.1, 0.5`

### Mixture comparison

```bash
python3 experiments/dpo_kto_mixture_1d/imbalance_compare_mix.py
```

Default controls:

- DPO: `good_ratio = 0.1, 1.0`
- KTO: `alpha = 0.1, 0.5`

## Output Artifacts

Each run now saves fresh-test metrics in:

- `results/dpo_kto_1d/.../runs/imbalance_summary.json`
- `results/dpo_kto_mixture_1d/.../runs/imbalance_summary.json`

These JSON files include:

- final policy parameters
- split sizes
- effective-good-mass
- fresh-test metrics

## Interpretation Guidance

The main checks are:

1. If `fresh loss` is close to `eval loss`, the conclusion is stable on unseen data.
2. If `fresh effective_good_mass` stays close to the training value, the learned behavior generalizes to new samples.
3. If DPO and KTO still differ on fresh data in the same way they differed on training data, the qualitative story is more credible.

If large gaps appear between train/eval and fresh-test:

- the result may be sample-sensitive
- the data-generation rule may be too sharp
- or the policy may be overfitting the sampled supervision set
