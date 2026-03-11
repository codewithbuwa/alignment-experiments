from __init__ import *

def run_sensitivity_grid(alphas, delta=1.0):
    dpo_sigmas = []
    kto_sigmas = []

    for alpha in alphas:
        # DPO
        w, l = build_dpo_dataset(good_ratio = alpha)
        dpo_policy = dp.train_dpo(BETA, w, l)[0]
        dpo_sigmas.append(dpo_policy.sigma().item())

        # KTO
        kto_policy, _ = kt.train_kto(BETA, delta, good_ratio=alpha)
        kto_sigmas.append(kto_policy.sigma().item())
        
    plt.figure()
    plt.plot(alphas, dpo_sigmas, marker='o', label="DPO")
    plt.plot(alphas, kto_sigmas, marker='o', label="KTO")
    plt.xlabel("Good ratio (alpha)")
    plt.ylabel("Final Sigma")
    plt.title("DPO vs KTO Data Robustness")
    plt.legend()
    plt.savefig("images/data_sensitivity.png")
    plt.show()

alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
run_sensitivity_grid(alphas = alphas)

