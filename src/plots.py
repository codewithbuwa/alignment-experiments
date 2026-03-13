"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Shared plotting helpers for report-ready experiment figures.
"""

import os
import torch
import matplotlib.pyplot as plt

from .distributions import gaussian_pdf, gaussian_log_prob


def _implicit_reward_grid(y_grid, mu, sigma, mu_ref, sigma_ref):
    logp = gaussian_log_prob(y_grid, mu, sigma)
    logp_ref = gaussian_log_prob(y_grid, y_grid.new_tensor(mu_ref), y_grid.new_tensor(sigma_ref))
    return logp - logp_ref


def plot_main_panels(
    output_path: str,
    y_min: float,
    y_max: float,
    mu_ref: float,
    sigma_ref: float,
    dpo_policy,
    kto_policy,
    dpo_history,
    kto_history,
):
    y = torch.linspace(y_min, y_max, 400)

    ref_pdf = gaussian_pdf(y, mu_ref, sigma_ref)
    dpo_pdf = gaussian_pdf(y, dpo_policy.mu.item(), dpo_policy.sigma.item())
    kto_pdf = gaussian_pdf(y, kto_policy.mu.item(), kto_policy.sigma.item())

    dpo_r = _implicit_reward_grid(y, dpo_policy.mu, dpo_policy.sigma, mu_ref, sigma_ref)
    kto_r = _implicit_reward_grid(y, kto_policy.mu, kto_policy.sigma, mu_ref, sigma_ref)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(y.numpy(), ref_pdf.numpy(), label="ref", linewidth=2)
    axes[0].plot(y.numpy(), dpo_pdf.numpy(), label="dpo", linewidth=2)
    axes[0].plot(y.numpy(), kto_pdf.numpy(), label="kto", linewidth=2)
    axes[0].set_title("Density Projection")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("pdf")
    axes[0].legend()

    axes[1].plot(y.numpy(), dpo_r.detach().numpy(), label="dpo", linewidth=2)
    axes[1].plot(y.numpy(), kto_r.detach().numpy(), label="kto", linewidth=2)
    axes[1].set_title("Implicit Reward")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("h(y)")
    axes[1].legend()

    axes[2].plot(dpo_history["sigma"], label="dpo", linewidth=2)
    axes[2].plot(kto_history["sigma"], label="kto", linewidth=2)
    axes[2].set_title("Sigma Dynamics")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("sigma")
    axes[2].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_entropy_comparison(output_path: str, dpo_history, kto_bal_history, kto_imb_history):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(dpo_history["sigma"], label="dpo", linewidth=2)
    ax.plot(kto_bal_history["sigma"], label="kto_bal", linewidth=2)
    ax.plot(kto_imb_history["sigma"], label="kto_imb", linewidth=2)
    ax.set_title("Sigma Dynamics (Sensitivity)")
    ax.set_xlabel("step")
    ax.set_ylabel("sigma")
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_kto_sensitivity(output_path: str, kto_bal_history, kto_imb_history):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(kto_bal_history["sigma"], label="kto_bal", linewidth=2)
    ax.plot(kto_imb_history["sigma"], label="kto_imb", linewidth=2)
    ax.set_title("KTO Sigma Dynamics (Sensitivity)")
    ax.set_xlabel("step")
    ax.set_ylabel("sigma")
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_zone_sweep(output_path: str, epsilons, final_sigmas):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(epsilons, final_sigmas, marker="o", linewidth=2)
    ax.set_title("KTO Zone Width Sweep")
    ax.set_xlabel("zone half-width (epsilon)")
    ax.set_ylabel("final sigma")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _mixture_pdf(y, weights, means, sigmas):
    pdf = torch.zeros_like(y)
    for w, m, s in zip(weights, means, sigmas):
        pdf = pdf + w * gaussian_pdf(y, float(m), float(s))
    return pdf


def plot_mixture_fit(output_path: str, y_min: float, y_max: float, target, learned):
    y = torch.linspace(y_min, y_max, 500)
    target_pdf = _mixture_pdf(y, target["weights"], target["means"], target["sigmas"])
    learned_pdf = _mixture_pdf(y, learned["weights"], learned["means"], learned["sigmas"])

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(y.numpy(), target_pdf.numpy(), label="target", linewidth=2)
    ax.plot(y.numpy(), learned_pdf.numpy(), label="learned", linewidth=2)
    ax.set_title("Mixture Fit")
    ax.set_xlabel("y")
    ax.set_ylabel("pdf")
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mixture_evolution(output_path: str, history):
    steps = list(range(len(history["weights"])))
    k = len(history["weights"][0])

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    for i in range(k):
        axes[0].plot(steps, [w[i] for w in history["weights"]], label=f"w{i}")
    axes[0].set_title("Weights")
    axes[0].legend(ncol=3)

    for i in range(k):
        axes[1].plot(steps, [m[i] for m in history["means"]], label=f"mu{i}")
    axes[1].set_title("Means")
    axes[1].legend(ncol=3)

    for i in range(k):
        axes[2].plot(steps, [s[i] for s in history["sigmas"]], label=f"sigma{i}")
    axes[2].set_title("Sigmas")
    axes[2].set_xlabel("step")
    axes[2].legend(ncol=3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
