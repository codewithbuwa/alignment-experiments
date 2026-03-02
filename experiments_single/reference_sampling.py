from __init__ import *
from policy.gaussian import *
from utils import *
import math

def reference_sampling(kto_policies: list[GaussianPolicy]):
    y_vals = torch.linspace(-2, 14, 1000)

    def gaussian_pdf(mu, sigma, y):
        return (1 / (sigma * math.sqrt(2 * math.pi))) * \
               torch.exp(-0.5 * ((y - mu) / sigma) ** 2)
    ref_pdf = gaussian_pdf(REF_MU, REF_SIGMA, y_vals)

    pdf = []
    modes = ["analytical", "batch", "running_avg", "fixed"]
    assert len(kto_policies) == len(modes)
    for i, kto_policy in zip(modes, kto_policies):
        kto_pdf = gaussian_pdf(kto_policy.mu.item(),
                            kto_policy.sigma().item(),
                            y_vals)
        pdf.append((kto_pdf, i))
    
    plt.figure()
    plt.plot(y_vals, ref_pdf, label="Reference policy")
    for kto_pdf, i in pdf:
        plt.plot(y_vals, kto_pdf, label=f"KTO {i}")
    plt.legend()
    plt.title("Density Projection")
    plt.savefig("images/reference_sampling.png")
    plt.show()

kto_anal, sigma_anal = kt.train_kto(beta=1.0, estimation_mode="analytical")
kto_batch, sigma_batch = kt.train_kto(beta=1.0, estimation_mode="batch")
kto_running_avg, sigma_rnavg = kt.train_kto(beta=1.0, estimation_mode="running_avg")
kto_fixed, sigma_fixed = kt.train_kto(beta=1.0, estimation_mode="fixed")

kto_policies = [kto_anal, kto_batch, kto_running_avg, kto_fixed]
reference_sampling(kto_policies)

plt.figure()
plt.plot(sigma_anal, label="analytical")
plt.plot(sigma_batch, label=f"batch")
plt.plot(sigma_rnavg, label="running avg")
plt.plot(sigma_fixed, label=f"fixed")
plt.legend()
plt.title("Density Projection")
plt.savefig("images/reference_sampling_entropy_dynamics.png")
plt.show()