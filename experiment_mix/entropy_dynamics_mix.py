from __init__ import *

if __name__ == "__main__":
    ref_mixture = REF_POLICY
    dpo_policy, dpo_sigmas = dp_mix.train_dpo_mixture(ref_mixture, beta=BETA)
    kto_policy, kto_sigmas = kt_mix.train_kto_mixture(ref_mixture, beta=BETA, estimation_mode="batch")

    plt.plot(dpo_sigmas, label="DPO")
    plt.plot(kto_sigmas, label="KTO")
    plt.xlabel("Step")
    plt.ylabel("Average Sigma")
    plt.legend()
    plt.title("Mixture Entropy Dynamics")
    plt.grid(True, alpha=0.3)
    plt.savefig("images/mixture_entropy_dynamics.png")
    plt.show()