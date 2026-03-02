from __init__ import *
import experiments_single.imp_reward as ir

def plot_implicit_reward(policy, ref_policy, beta, title):
    y_vals = torch.linspace(-2, 14, 1000)
    r_vals = ir.implicit_reward(policy, ref_policy, y_vals, beta).detach()
    plt.figure()
    plt.plot(y_vals, r_vals)
    plt.title(title)
    plt.xlabel("y")
    plt.ylabel("r(y)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"images/{title}.png")
    plt.show()

if __name__ == "__main__":
    ref_mixture = REF_POLICY
    dpo_policy, _ = dp_mix.train_dpo_mixture(ref_mixture, beta=BETA)
    kto_policy, _ = kt_mix.train_kto_mixture(ref_mixture, beta=BETA, estimation_mode="batch")
    plot_implicit_reward(dpo_policy, ref_mixture, BETA, "DPO Mixture Implicit Reward")
    plot_implicit_reward(kto_policy, ref_mixture, BETA, "KTO Mixture Implicit Reward")