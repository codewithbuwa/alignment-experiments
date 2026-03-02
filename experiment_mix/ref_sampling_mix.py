from __init__ import *
import math

def reference_sampling_mixture(kto_policies: list, ref_policy, title_suffix=""):
    """
    Compare KTO policies trained with different KL estimation modes.
    
    Args:
        kto_policies: List of trained KTO mixture policies
        ref_policy: Reference policy (mixture or single Gaussian)
        title_suffix: Optional suffix for plot title
    """
    y_vals = torch.linspace(-2, 14, 1000)
    
    # Compute reference density
    ref_density = torch.exp(ref_policy.log_prob(y_vals)).detach()
    
    # Compute densities for each KTO policy
    pdfs = []
    modes = ["batch", "running_avg", "fixed"]
    assert len(kto_policies) == len(modes)
    
    for mode, policy in zip(modes, kto_policies):
        density = torch.exp(policy.log_prob(y_vals)).detach()
        pdfs.append((density, mode))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_vals, ref_density, 'k--', label="Reference policy", linewidth=2)
    
    colors = ['b', 'r', 'g', 'orange']
    for (density, mode), color in zip(pdfs, colors):
        plt.plot(y_vals, density, color=color, label=f"KTO {mode}", linewidth=1.5)
    
    # Shade the desirable zone
    plt.axvspan(5.5, 8.5, alpha=0.1, color='green', label="Desirable Zone")
    
    plt.xlabel("y")
    plt.ylabel("Density")
    plt.title(f"Mixture Policy: KL Estimation Mode Comparison {title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.savefig(f"images/reference_sampling_mixture{title_suffix}.png".replace(" ", "_"))
    plt.show()
 

if __name__ == "__main__":
    torch.manual_seed(42)
    ref_mixture = REF_POLICY
    print("\nTraining KTO policies with different KL estimation modes...")
    
    
    kto_batch, sigmas_batch = kt_mix.train_kto_mixture(
        ref_policy=ref_mixture,
        beta=BETA,
        delta=1.5,
        estimation_mode="batch",
        n_components=2
    )
    
    kto_running, sigmas_running = kt_mix.train_kto_mixture(
        ref_policy=ref_mixture,
        beta=BETA,
        delta=1.5,
        estimation_mode="running_avg",
        n_components=2
    )
    
    kto_fixed, sigmas_fixed = kt_mix.train_kto_mixture(
        ref_policy=ref_mixture,
        beta=BETA,
        delta=1.5,
        estimation_mode="fixed",
        n_components=2
    )
    
    # Collect policies
    kto_policies = [kto_batch, kto_running, kto_fixed]
    
    
    # Plot densities
    reference_sampling_mixture(kto_policies, ref_mixture, title_suffix="_1")
    
    print("\n" + "="*50)
    
    ref_mixed = GaussianMixturePolicy(
        n_components=2,
        mu_init=torch.tensor([3.0, 9.0]), 
        log_sigma_init=torch.tensor([0.0, 0.0]),
        logits_init=torch.tensor([0.0, 0.0])
    ).to(DEVICE)
    
    
    # Train only analytical mode for this experiment
    kto_mixed_batch, _ = kt_mix.train_kto_mixture(
        ref_policy=ref_mixed,
        beta=BETA,
        delta=1.5,
        estimation_mode="batch",
        n_components=2
    )
    kto_running, sigmas_running = kt_mix.train_kto_mixture(
        ref_policy=ref_mixed,
        beta=BETA,
        delta=1.5,
        estimation_mode="running_avg",
        n_components=2
    )
    
    kto_fixed, sigmas_fixed = kt_mix.train_kto_mixture(
        ref_policy=ref_mixed,
        beta=BETA,
        delta=1.5,
        estimation_mode="fixed",
        n_components=2
    )
    
    # Collect policies
    kto_policies = [kto_batch, kto_running, kto_fixed]
    
    
    # Plot densities
    reference_sampling_mixture(kto_policies, ref_mixed, title_suffix="")
    
    plt.figure(figsize=(10, 6))
    plt.plot(sigmas_batch, 'r-', label="batch", alpha=0.7)
    plt.plot(sigmas_running, 'g-', label="running_avg", alpha=0.7)
    plt.plot(sigmas_fixed, 'orange', label="fixed", alpha=0.7)
    plt.xlabel("Training Step")
    plt.ylabel("Average Sigma")
    plt.title("KL Estimation Mode: Entropy Dynamics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("images/reference_sampling_mixture_dynamics.png")
    plt.show()