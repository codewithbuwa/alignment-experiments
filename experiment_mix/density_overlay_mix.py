from __init__ import *

def plot_mixture_densities(ref_policy, dpo_policy, kto_policy):
    y_vals = torch.linspace(-2, 14, 1000)
    ref_pdf = torch.exp(ref_policy.log_prob(y_vals)).detach()
    dpo_pdf = torch.exp(dpo_policy.log_prob(y_vals)).detach()
    kto_pdf = torch.exp(kto_policy.log_prob(y_vals)).detach()

    plt.figure()
    plt.plot(y_vals, ref_pdf, label="Reference Mixture", linestyle='--')
    plt.plot(y_vals, dpo_pdf, label="DPO Mixture")
    plt.plot(y_vals, kto_pdf, label="KTO Mixture")
    plt.fill_between(y_vals, 0, ((y_vals >= 5.5) & (y_vals <= 8.5)) * 0.5, alpha=0.2, label="Desirable Zone")
    plt.legend()
    plt.title("Mixture Densities after Training")
    plt.xlabel("y")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.savefig("images/mixture_density_overlay.png")
    plt.show()

if __name__ == "__main__":
    # Create a reference mixture (e.g., two modes at 4 and 6, equal weights, sigma=1)
    ref_mixture = REF_POLICY

    # Train DPO and KTO on data from this reference
    dpo_policy, _ = dp_mix.train_dpo_mixture(ref_mixture, beta=BETA, n_components=2)
    kto_policy, _ = kt_mix.train_kto_mixture(ref_mixture, beta=BETA, delta=1.5, estimation_mode="batch", n_components=2)

    plot_mixture_densities(ref_mixture, dpo_policy, kto_policy)