import argparse

import torch
import matplotlib.pyplot as plt

from utils import BETA, DATASET_SIZE, DEVICE, LAMBDA, LR, STEPS
from policy.gaussian_mixture import GaussianMixturePolicy, REF_POLICY
from dataset.dataset_mix import build_mixture_dpo_dataset, build_mixture_kto_dataset
from experiments_single.imp_reward import implicit_reward


def _init_history(steps: int, n_components: int):
    return {
        "mus": torch.zeros(steps, n_components),
        "sigmas": torch.zeros(steps, n_components),
        "weights": torch.zeros(steps, n_components),
    }


def _record(history, step: int, policy: GaussianMixturePolicy):
    with torch.no_grad():
        history["mus"][step] = policy.mus.detach().cpu()
        history["sigmas"][step] = policy.sigmas().detach().cpu()
        history["weights"][step] = policy.probs().detach().cpu()


def train_dpo_with_history(
    ref_policy: GaussianMixturePolicy = REF_POLICY,
    beta: float = BETA,
    n_components: int = 2,
    steps: int = STEPS,
    lr: float = LR,
):
    policy = GaussianMixturePolicy(n_components=n_components).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    y_w, y_l = build_mixture_dpo_dataset(ref_policy=ref_policy)
    history = _init_history(steps=steps, n_components=n_components)

    for step in range(steps):
        optimizer.zero_grad()
        h_w = implicit_reward(policy, ref_policy, y_w, beta)
        h_l = implicit_reward(policy, ref_policy, y_l, beta)
        loss = -torch.mean(torch.log(torch.sigmoid(h_w - h_l)))
        loss.backward()
        optimizer.step()
        _record(history, step, policy)

    return policy, history


def train_kto_with_history(
    ref_policy: GaussianMixturePolicy = REF_POLICY,
    beta: float = BETA,
    delta: float = 1.5,
    estimation_mode: str = "batch",
    n_components: int = 2,
    steps: int = STEPS,
    lr: float = LR,
    alpha: float = 0.3,
):
    policy = GaussianMixturePolicy(n_components=n_components).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    y_fixed, labels_fixed = build_mixture_kto_dataset(ref_policy=ref_policy, delta=delta)
    history = _init_history(steps=steps, n_components=n_components)
    running_kl = torch.tensor(0.0).to(DEVICE)

    for step in range(steps):
        optimizer.zero_grad()
        h = implicit_reward(policy, ref_policy, y_fixed, beta)

        if estimation_mode == "batch":
            with torch.no_grad():
                kl = policy.kl_to_ref(n_samples=DATASET_SIZE)
        elif estimation_mode == "running_avg":
            with torch.no_grad():
                y_sample = policy.sample(DATASET_SIZE)
                batch_kl = torch.mean(policy.log_prob(y_sample) - ref_policy.log_prob(y_sample)).detach()
                running_kl = (1 - alpha) * running_kl + alpha * batch_kl
                kl = running_kl
        elif estimation_mode == "fixed":
            kl = 0.2
        else:
            raise ValueError(f"Unknown estimation_mode: {estimation_mode}")

        z = h - kl
        v = torch.zeros_like(z)
        v[labels_fixed == 1.0] = torch.sigmoid(z[labels_fixed == 1.0])
        v[labels_fixed == 0.0] = torch.sigmoid(-LAMBDA * z[labels_fixed == 0.0])
        loss = torch.mean(1 - v)
        loss.backward()
        optimizer.step()
        _record(history, step, policy)

    return policy, history


def plot_component_evolution(history, title_prefix: str, out_path: str):
    steps = history["mus"].shape[0]
    n_components = history["mus"].shape[1]
    x = torch.arange(steps)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    metrics = ["mus", "sigmas", "weights"]
    labels = ["Mean (mu)", "Std (sigma)", "Mixture weight"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for ax, metric, ylabel in zip(axes, metrics, labels):
        for k in range(n_components):
            ax.plot(
                x,
                history[metric][:, k],
                label=f"Component {k}",
                color=colors[k % len(colors)],
                linewidth=1.6,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("Training step")
    fig.suptitle(f"{title_prefix}: Component Evolution", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)


def _weighted_component_pdf(y_vals, mu, sigma, weight):
    norm = 1.0 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))
    exp_term = torch.exp(-0.5 * ((y_vals - mu) / sigma) ** 2)
    return weight * norm * exp_term


def plot_density_milestones(history, title_prefix: str, out_path: str):
    steps = history["mus"].shape[0]
    n_components = history["mus"].shape[1]
    y_vals = torch.linspace(-2, 14, 1000)
    milestone_fracs = [0.25, 0.50, 0.75, 1.00]
    milestone_steps = [max(0, min(steps - 1, int(steps * frac) - 1)) for frac in milestone_fracs]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, frac, step_idx in zip(axes, milestone_fracs, milestone_steps):
        total_density = torch.zeros_like(y_vals)
        param_lines = []
        for k in range(n_components):
            mu_k = history["mus"][step_idx, k]
            sigma_k = history["sigmas"][step_idx, k]
            weight_k = history["weights"][step_idx, k]
            comp_density = _weighted_component_pdf(
                y_vals,
                mu_k,
                sigma_k,
                weight_k,
            )
            total_density += comp_density
            param_lines.append(
                f"C{k}: mu={mu_k.item():.2f}, sigma={sigma_k.item():.2f}, w={weight_k.item():.2f}"
            )
            ax.plot(
                y_vals,
                comp_density,
                linestyle="--",
                linewidth=1.2,
                color=colors[k % len(colors)],
                alpha=0.9,
                label=f"C{k}",
            )

        ax.plot(y_vals, total_density, color="magenta", linewidth=2.0, label="Mixture")
        ax.set_title(f"{int(frac * 100)}% epochs (step {step_idx + 1})")
        ax.text(
            0.02,
            0.98,
            "\n".join(param_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "0.7",
            },
        )
        ax.grid(True, alpha=0.25)

    for ax in axes[::2]:
        ax.set_ylabel("Density")
    for ax in axes[2:]:
        ax.set_xlabel("y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"{title_prefix}: Density at 25% Epoch Milestones", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)


def main():
    parser = argparse.ArgumentParser(description="Track Gaussian-mixture component evolution during training.")
    parser.add_argument("--algo", choices=["dpo", "kto"], default="kto")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--delta", type=float, default=1.5)
    parser.add_argument("--estimation-mode", choices=["batch", "running_avg", "fixed"], default="batch")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--density-out", type=str, default=None)
    args = parser.parse_args()

    if args.algo == "dpo":
        _, history = train_dpo_with_history(steps=args.steps, n_components=args.n_components)
        mode_tag = "dpo"
    else:
        _, history = train_kto_with_history(
            steps=args.steps,
            n_components=args.n_components,
            delta=args.delta,
            estimation_mode=args.estimation_mode,
        )
        mode_tag = f"kto_{args.estimation_mode}"

    out_path = args.out or f"images/mixture_component_evolution_{mode_tag}_{args.n_components}.png"
    density_out_path = args.density_out or f"images/mixture_density_milestones_{mode_tag}_{args.n_components}.png"
    plot_component_evolution(
        history=history,
        title_prefix=mode_tag.upper(),
        out_path=out_path,
    )
    plot_density_milestones(
        history=history,
        title_prefix=mode_tag.upper(),
        out_path=density_out_path,
    )
    print(f"Saved plot to {out_path}")
    print(f"Saved plot to {density_out_path}")


if __name__ == "__main__":
    main()
