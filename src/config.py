"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Shared experiment configuration dataclasses for single and mixture runs.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    seed: int = 42

    # Reference policy N(mu_ref, sigma_ref^2)
    mu_ref: float = 5.0
    sigma_ref: float = 2.0

    # Oracle target
    target: float = 7.0

    # Desirable zone Z = [target - zone_half_width, target + zone_half_width]
    zone_half_width: float = 1.5

    # Dataset and optimization
    dataset_size: int = 10000
    batch_size: int = 128
    steps: int = 2000
    lr: float = 1e-3
    eval_fraction: float = 0.2

    # DPO/KTO scalar
    beta: float = 1.0

    # Initialization (policy only)
    init_mu: Optional[float] = None
    init_sigma: Optional[float] = None

    # KTO
    kto_gamma: float = 1.33
    kto_good_fraction: float = 0.5

    # KL handling for KTO
    kl_mode: str = "analytic"  # analytic | fixed | running | batch
    kl_fixed: float = 0.0
    kl_ema_decay: float = 0.99
    kl_grad: bool = False

    # Plot range
    y_min: float = -2.0
    y_max: float = 14.0

    # Sweep
    zone_sweep: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    delta_sweep: List[float] = field(default_factory=lambda: [0.1, 1.0, 4.0])
    beta_sweep: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    data_sensitivity_alphas: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])

    device: str = "cpu"


@dataclass
class MixtureDPOKTOConfig:
    seed: int = 42
    n_components: int = 2
    dataset_size: int = 1000
    steps: int = 2000
    lr: float = 1e-3
    eval_fraction: float = 0.2

    target: float = 7.0
    reference_sigma: float = 2.0

    beta: float = 1.0
    kto_gamma: float = 1.33

    zone_half_width: float = 1.5
    delta: float = 1.5

    # Initialization (policy only)
    init_means: Optional[List[float]] = None
    init_sigmas: Optional[List[float]] = None
    init_logits: Optional[List[float]] = None

    # KL estimation for KTO mixture
    kl_mode: str = "batch"  # batch | running | fixed
    kl_fixed: float = 0.2
    kl_ema_decay: float = 0.3

    y_min: float = -2.0
    y_max: float = 14.0

    device: str = "cpu"


@dataclass
class MixtureConfig:
    seed: int = 123
    n_components: int = 3
    dataset_size: int = 4000
    val_fraction: float = 0.2
    steps: int = 1500
    batch_size: int = 256
    lr: float = 3e-3

    # Target mixture (fixed)
    target_weights: List[float] = None
    target_means: List[float] = None
    target_sigmas: List[float] = None

    # Initialization
    init_means: List[float] = None
    init_sigmas: List[float] = None
    init_logits: List[float] = None

    y_min: float = -5.0
    y_max: float = 15.0

    device: str = "cpu"

    def with_defaults(self):
        if self.target_weights is None:
            self.target_weights = [0.5, 0.3, 0.2]
        if self.target_means is None:
            self.target_means = [3.0, 7.0, 10.0]
        if self.target_sigmas is None:
            self.target_sigmas = [1.0, 0.8, 1.2]
        if self.init_means is None:
            self.init_means = [2.0, 6.0, 9.0]
        if self.init_sigmas is None:
            self.init_sigmas = [1.5, 1.5, 1.5]
        if self.init_logits is None:
            self.init_logits = [0.0, 0.0, 0.0]
        return self
