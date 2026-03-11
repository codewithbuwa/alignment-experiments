from __init__ import *
import experiments_single.imp_reward as ir
import experiments_single.reward_plot as rp
from policy.gaussian_mixture import REF_POLICY

if __name__ == "__main__":
    ref_mixture = REF_POLICY
    dpo_policy, _ = dp_mix.train_dpo_mixture(ref_mixture, beta=BETA)
    kto_policy, _ = kt_mix.train_kto_mixture(ref_mixture, beta=BETA, estimation_mode="batch")
    rp.plot_implicit_reward([dpo_policy, kto_policy], ref_mixture, BETA, "Mixture Implicit Reward", mode = "mixture")
