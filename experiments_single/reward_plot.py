from __init__ import *
import experiments_single.imp_reward as ir

REF_POLICY = GaussianPolicy(REF_MU, math.log(REF_SIGMA)).to(DEVICE)
def plot_implicit_reward(policy, beta, title="Implicit Reward"):
    y_vals = torch.linspace(-2, 14, 10000)
    r_vals = ir.implicit_reward(policy, REF_POLICY, y_vals, beta).detach()

    plt.figure()
    plt.plot(y_vals, r_vals)
    plt.title(title)
    plt.xlabel("y")
    plt.ylabel("r(y)")
    plt.savefig(f"images/{title}.png")
    plt.show()

dpo_policy = dp.train_dpo(beta=1.0)[0]
kto_policy = kt.train_kto(beta=1.0)[0]
plot_implicit_reward(dpo_policy, beta=1.0, title="DPO Implicit Reward")
plot_implicit_reward(kto_policy, beta=1.0, title="KTO Implicit Reward")
