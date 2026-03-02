from __init__ import *

def run_beta_sweep(beta_values):
    dpo_sigmas = []
    kto_sigmas = []

    for beta in beta_values:
        dpo_policy = dp.train_dpo(beta)[0]
        kto_policy = kt.train_kto(beta)[0]

        dpo_sigmas.append(dpo_policy.sigma().item())
        kto_sigmas.append(kto_policy.sigma().item())

    plt.figure()
    plt.plot(beta_values, dpo_sigmas, marker='o', label="DPO")
    plt.plot(beta_values, kto_sigmas, marker='o', label="KTO")
    plt.xlabel("Beta")
    plt.ylabel("Final Sigma")
    plt.legend()
    plt.title("Final Sigma vs Beta")
    plt.savefig("images/sigma_vs_beta.png")
    plt.show()

    return dpo_sigmas, kto_sigmas

beta_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
run_beta_sweep(beta_values)