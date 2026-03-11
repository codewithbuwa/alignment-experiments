from pyparsing import alphas

from __init__ import *
import experiments_single.imp_reward as ir

def plot_implicit_reward(policy, ref_policy, beta = BETA, title="Reward plot", mode = ""):
    y_vals = torch.linspace(-2, 14, 10000)
    r_vals = [ir.implicit_reward(pol, ref_policy, y_vals, beta).detach() for pol in policy]
    oracle_reward = lambda y: -abs(y-TARGET)
    r_vals.append(oracle_reward(y_vals))
    labels = ["DPO reward", "KTO reward", "Oracle reward"]
    plt.figure()
    for i, rval in enumerate(r_vals):
        plt.plot(y_vals, rval, label = f"{labels[i]}: max = {max(rval):.2f}")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("y")
    plt.ylabel("Reward (r(y))")
    plt.savefig(f"images/{labels[0][-6:]}_{mode}.png")
    plt.show()

if __name__ == "__main__":
    REF_POLICY = GaussianPolicy(REF_MU, math.log(REF_SIGMA)).to(DEVICE)

    dpo_policy = dp.train_dpo(beta=1.0)[0]
    kto_policy = kt.train_kto(beta=1.0)[0]

    plot_implicit_reward([dpo_policy, kto_policy], REF_POLICY, beta=1.0, mode = "single")
