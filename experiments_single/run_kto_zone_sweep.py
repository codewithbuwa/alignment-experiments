from __init__ import *

def run_delta_sweep(deltas):
    results = []
    policies = []
    for delta in deltas:

        policy, sigmas = kt.train_kto(BETA, delta)
        final_sigma = policy.sigma().item()

        results.append((delta, final_sigma))
        policies.append(policy)            
    deltas, sigmas = zip(*results)

    plt.figure()
    plt.plot(deltas, sigmas, marker='o')
    plt.xlabel("Delta (Zone Half-Width)")
    plt.ylabel("Final Sigma")
    plt.title("KTO: Final Sigma vs Delta")
    plt.savefig("images/kto_Final_Sigma_vs_Delta.png")
    plt.show()

    return results, policies, sigmas
delta_values = [0.1, 1.0, 4.0]
delta_results, policies, sigmas = run_delta_sweep(delta_values)

# Plot
plt.figure(figsize=(10, 6))
y_vals = torch.linspace(-2, 14, 1000)
colors = ['b', 'r', 'g']

for pol, color, delta_vals in zip(policies, colors, delta_values): 
    density = torch.exp(pol.log_prob(y_vals)).detach()
    plt.plot(y_vals, density, color=color, label=f"Zone delta = {delta_vals}", linewidth=1.5)

# Shade the desirable zone
plt.axvspan(5.5, 8.5, alpha=0.1, color='green', label="Desirable Zone")
title = "Impact of delta on KTO density"
plt.xlabel("y")
plt.ylabel("Density")
plt.title(title)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"images/{title}.png".replace(" ", "_"))
plt.show()