import argparse

import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from dataset.dataset import build_dpo_dataset, build_kto_dataset
from dataset.dataset_mix import build_mixture_dpo_dataset, build_mixture_kto_dataset
from experiments_single.imp_reward import implicit_reward
from policy.gaussian import GaussianPolicy
from policy.gaussian_mixture import GaussianMixturePolicy, REF_POLICY
from utils import BETA, DATASET_SIZE, DEVICE, LAMBDA, LR, REF_MU, REF_SIGMA, STEPS


def train_dpo_single(w, l, beta=BETA, steps=STEPS, lr=LR):
    ref_policy = GaussianPolicy(REF_MU, torch.log(torch.tensor(REF_SIGMA))).to(DEVICE)
    policy = GaussianPolicy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    history = {"mu": torch.zeros(steps), "sigma": torch.zeros(steps)}

    for step in range(steps):
        optimizer.zero_grad()
        h_w = implicit_reward(policy, ref_policy, w, beta)
        h_l = implicit_reward(policy, ref_policy, l, beta)
        loss = -torch.mean(torch.log(torch.sigmoid(h_w - h_l)))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            history["mu"][step] = policy.mu.detach().cpu()
            history["sigma"][step] = policy.sigma().detach().cpu()

    return policy, history


def train_dpo_mixture_local(w, l, ref_policy=REF_POLICY, beta=BETA, n_components=2, steps=STEPS, lr=LR):
    policy = GaussianMixturePolicy(n_components=n_components).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    history = {
        "mus": torch.zeros(steps, n_components),
        "sigmas": torch.zeros(steps, n_components),
        "weights": torch.zeros(steps, n_components),
    }

    for step in range(steps):
        optimizer.zero_grad()
        h_w = implicit_reward(policy, ref_policy, w, beta)
        h_l = implicit_reward(policy, ref_policy, l, beta)
        loss = -torch.mean(torch.log(torch.sigmoid(h_w - h_l)))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            history["mus"][step] = policy.mus.detach().cpu()
            history["sigmas"][step] = policy.sigmas().detach().cpu()
            history["weights"][step] = policy.probs().detach().cpu()

    return policy, history


def train_kto_single(y, labels, beta=BETA, estimation_mode="batch", steps=STEPS, lr=LR, alpha=0.3):
    ref_policy = GaussianPolicy(REF_MU, torch.log(torch.tensor(REF_SIGMA))).to(DEVICE)
    policy = GaussianPolicy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    running_kl = torch.tensor(0.0).to(DEVICE)
    history = {"mu": torch.zeros(steps), "sigma": torch.zeros(steps)}

    for step in range(steps):
        optimizer.zero_grad()
        h = implicit_reward(policy, ref_policy, y, beta)

        if estimation_mode == "analytical":
            kl = policy.kl_to_ref().detach()
        elif estimation_mode == "batch":
            with torch.no_grad():
                y_sample = policy.sample(DATASET_SIZE)
                batch_kl = torch.mean(policy.log_prob(y_sample) - ref_policy.log_prob(y_sample)).detach()
                kl = batch_kl
        elif estimation_mode == "running_avg":
            with torch.no_grad():
                y_sample = policy.sample(DATASET_SIZE)
                batch_kl = torch.mean(policy.log_prob(y_sample) - ref_policy.log_prob(y_sample)).detach()
                running_kl = (1 - alpha) * running_kl + alpha * batch_kl
                kl = running_kl
        elif estimation_mode == "fixed":
            kl = 0.3
        else:
            raise ValueError(f"Unknown estimation_mode: {estimation_mode}")

        z = h - kl
        v = torch.zeros_like(z)
        v[labels == 1.0] = torch.sigmoid(z[labels == 1.0])
        v[labels == 0.0] = torch.sigmoid(-LAMBDA * z[labels == 0.0])
        loss = torch.mean(1 - v)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            history["mu"][step] = policy.mu.detach().cpu()
            history["sigma"][step] = policy.sigma().detach().cpu()

    return policy, history


def train_kto_mixture_local(
    y,
    labels,
    ref_policy=REF_POLICY,
    beta=BETA,
    estimation_mode="batch",
    n_components=2,
    steps=STEPS,
    lr=LR,
    alpha=0.3,
):
    policy = GaussianMixturePolicy(n_components=n_components).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    running_kl = torch.tensor(0.0).to(DEVICE)
    history = {
        "mus": torch.zeros(steps, n_components),
        "sigmas": torch.zeros(steps, n_components),
        "weights": torch.zeros(steps, n_components),
    }

    for step in range(steps):
        optimizer.zero_grad()
        h = implicit_reward(policy, ref_policy, y, beta)

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
        v[labels == 1.0] = torch.sigmoid(z[labels == 1.0])
        v[labels == 0.0] = torch.sigmoid(-LAMBDA * z[labels == 0.0])
        loss = torch.mean(1 - v)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            history["mus"][step] = policy.mus.detach().cpu()
            history["sigmas"][step] = policy.sigmas().detach().cpu()
            history["weights"][step] = policy.probs().detach().cpu()

    return policy, history


def sweep_robustness(alphas, delta=1.5, steps=STEPS, lr=LR, n_components=2, kto_mode="batch", track_alpha=0.5):
    results = {
        "alpha": [],
        "dpo_single_mu": [],
        "dpo_single_sigma": [],
        "kto_single_mu": [],
        "kto_single_sigma": [],
    }
    for k in range(n_components):
        results[f"dpo_mix_sigma_c{k}"] = []
        results[f"dpo_mix_mu_c{k}"] = []
        results[f"dpo_mix_weight_c{k}"] = []
        results[f"kto_mix_sigma_c{k}"] = []
        results[f"kto_mix_mu_c{k}"] = []
        results[f"kto_mix_weight_c{k}"] = []
    tracked = None

    for alpha in alphas:
        print(f"Running alpha={alpha:.2f}")

        w_single, l_single = build_dpo_dataset(good_ratio=alpha)
        w_mix, l_mix = build_mixture_dpo_dataset(ref_policy=REF_POLICY, good_ratio=alpha)
        y_single, labels_single = build_kto_dataset(delta=delta, good_ratio=alpha)
        y_mix, labels_mix = build_mixture_kto_dataset(ref_policy=REF_POLICY, delta=delta, good_ratio=alpha)

        dpo_single, dpo_single_hist = train_dpo_single(w_single, l_single, steps=steps, lr=lr)
        dpo_mix, dpo_mix_hist = train_dpo_mixture_local(
            w_mix,
            l_mix,
            ref_policy=REF_POLICY,
            n_components=n_components,
            steps=steps,
            lr=lr,
        )
        kto_single, kto_single_hist = train_kto_single(
            y_single,
            labels_single,
            estimation_mode=kto_mode,
            steps=steps,
            lr=lr,
        )
        kto_mix, kto_mix_hist = train_kto_mixture_local(
            y_mix,
            labels_mix,
            ref_policy=REF_POLICY,
            estimation_mode=kto_mode if kto_mode != "analytical" else "batch",
            n_components=n_components,
            steps=steps,
            lr=lr,
        )

        results["alpha"].append(alpha)
        results["dpo_single_mu"].append(dpo_single.mu.item())
        results["dpo_single_sigma"].append(dpo_single.sigma().item())
        results["kto_single_mu"].append(kto_single.mu.item())
        results["kto_single_sigma"].append(kto_single.sigma().item())

        dpo_mix_mus = dpo_mix.mus.detach().cpu().tolist()
        dpo_mix_sigmas = dpo_mix.sigmas().detach().cpu().tolist()
        dpo_mix_weights = dpo_mix.probs().detach().cpu().tolist()
        kto_mix_mus = kto_mix.mus.detach().cpu().tolist()
        kto_mix_sigmas = kto_mix.sigmas().detach().cpu().tolist()
        kto_mix_weights = kto_mix.probs().detach().cpu().tolist()

        for k in range(n_components):
            results[f"dpo_mix_sigma_c{k}"].append(dpo_mix_sigmas[k])
            results[f"dpo_mix_mu_c{k}"].append(dpo_mix_mus[k])
            results[f"dpo_mix_weight_c{k}"].append(dpo_mix_weights[k])
            results[f"kto_mix_sigma_c{k}"].append(kto_mix_sigmas[k])
            results[f"kto_mix_mu_c{k}"].append(kto_mix_mus[k])
            results[f"kto_mix_weight_c{k}"].append(kto_mix_weights[k])

        if tracked is None and abs(alpha - track_alpha) < 1e-9:
            tracked = {
                "alpha": alpha,
                "dpo_single": dpo_single_hist,
                "dpo_mix": dpo_mix_hist,
                "kto_single": kto_single_hist,
                "kto_mix": kto_mix_hist,
            }

    if tracked is None and len(alphas) > 0:
        idx = min(range(len(alphas)), key=lambda i: abs(alphas[i] - track_alpha))
        fallback_alpha = alphas[idx]
        print(f"track_alpha={track_alpha:.3f} not in list, no dynamics plot will be generated for exact match.")
        print(f"Use one of the provided alphas, e.g. --track-alpha {fallback_alpha}")
    return results, tracked


def plot_results(results, scale_out_path, n_components):
    x = results["alpha"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(x, results["dpo_single_mu"], marker="o", color="black", label="Single mu")
    for k in range(n_components):
        axes[0, 0].plot(x, results[f"dpo_mix_mu_c{k}"], marker="o", label=f"Mix C{k} mu")
    axes[0, 0].set_title("DPO: Final Mu by Component")
    axes[0, 0].set_xlabel("Good ratio (alpha)")
    axes[0, 0].set_ylabel("Mu")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(x, results["kto_single_mu"], marker="o", color="black", label="Single mu")
    for k in range(n_components):
        axes[0, 1].plot(x, results[f"kto_mix_mu_c{k}"], marker="o", label=f"Mix C{k} mu")
    axes[0, 1].set_title("KTO: Final Mu by Component")
    axes[0, 1].set_xlabel("Good ratio (alpha)")
    axes[0, 1].set_ylabel("Mu")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(x, results["dpo_single_sigma"], marker="o", color="black", label="Single sigma")
    for k in range(n_components):
        axes[1, 0].plot(x, results[f"dpo_mix_sigma_c{k}"], marker="o", label=f"Mix C{k} sigma")
    axes[1, 0].set_title("DPO: Final Sigma by Component")
    axes[1, 0].set_xlabel("Good ratio (alpha)")
    axes[1, 0].set_ylabel("Sigma")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(x, results["kto_single_sigma"], marker="o", color="black", label="Single sigma")
    for k in range(n_components):
        axes[1, 1].plot(x, results[f"kto_mix_sigma_c{k}"], marker="o", label=f"Mix C{k} sigma")
    axes[1, 1].set_title("KTO: Final Sigma by Component")
    axes[1, 1].set_xlabel("Good ratio (alpha)")
    axes[1, 1].set_ylabel("Sigma")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(scale_out_path, dpi=140)


def plot_training_dynamics(tracked, out_path, n_components):
    if tracked is None:
        return
    steps = tracked["dpo_single"]["mu"].shape[0]
    x = torch.arange(steps)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

    for k in range(n_components):
        axes[0, 1].plot(x, tracked["dpo_mix"]["mus"][:, k], label=f"C{k}")
        axes[1, 1].plot(x, tracked["dpo_mix"]["sigmas"][:, k], label=f"C{k}")
        axes[2, 1].plot(x, tracked["dpo_mix"]["weights"][:, k], label=f"C{k}")
        axes[0, 0].plot(x, tracked["kto_mix"]["mus"][:, k], label=f"C{k}")
        axes[1, 0].plot(x, tracked["kto_mix"]["sigmas"][:, k], label=f"C{k}")
        axes[2, 0].plot(x, tracked["kto_mix"]["weights"][:, k], label=f"C{k}")

    axes[0, 0].set_title(f"KTO Mixture (alpha={tracked['alpha']:.2f})")
    axes[0, 1].set_title(f"DPO Mixture (alpha={tracked['alpha']:.2f})")

    axes[0, 0].set_ylabel("Mu")
    axes[1, 0].set_ylabel("Sigma")
    axes[2, 0].set_ylabel("Weight")
    axes[0, 1].set_ylabel("Mu")
    axes[1, 1].set_ylabel("Sigma")
    axes[2, 1].set_ylabel("Weight")

    for r in range(3):
        for c in range(2):
            axes[r, c].grid(True, alpha=0.3)
            axes[r, c].legend()

    axes[2, 0].set_xlabel("Training step")
    axes[2, 1].set_xlabel("Training step")
    fig.suptitle("Mixture Parameter Dynamics During Training", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)


def print_results_table(results, n_components):
    header_parts = [
        "alpha",
        "dpo_s_mu",
        "dpo_s_sigma",
        "kto_s_mu",
        "kto_s_sigma",
    ]
    for k in range(n_components):
        header_parts.extend(
            [
                f"dpo_c{k}_mu",
                f"dpo_c{k}_sigma",
                f"dpo_c{k}_w",
                f"kto_c{k}_mu",
                f"kto_c{k}_sigma",
                f"kto_c{k}_w",
            ]
        )
    print("\n" + " | ".join(header_parts))

    for i, alpha in enumerate(results["alpha"]):
        row_parts = [
            f"{alpha:.2f}",
            f"{results['dpo_single_mu'][i]:.3f}",
            f"{results['dpo_single_sigma'][i]:.3f}",
            f"{results['kto_single_mu'][i]:.3f}",
            f"{results['kto_single_sigma'][i]:.3f}",
        ]
        for k in range(n_components):
            row_parts.extend(
                [
                    f"{results[f'dpo_mix_mu_c{k}'][i]:.3f}",
                    f"{results[f'dpo_mix_sigma_c{k}'][i]:.3f}",
                    f"{results[f'dpo_mix_weight_c{k}'][i]:.3f}",
                    f"{results[f'kto_mix_mu_c{k}'][i]:.3f}",
                    f"{results[f'kto_mix_sigma_c{k}'][i]:.3f}",
                    f"{results[f'kto_mix_weight_c{k}'][i]:.3f}",
                ]
            )
        print(" | ".join(row_parts))


def parse_alpha_list(alpha_csv):
    return [float(x.strip()) for x in alpha_csv.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Robustness experiment: compare single vs mixture Gaussian for DPO and KTO."
    )
    parser.add_argument("--alphas", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--delta", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--kto-mode", choices=["analytical", "batch", "running_avg", "fixed"], default="batch")
    parser.add_argument("--track-alpha", type=float, default=0.5)
    parser.add_argument("--summary-out", type=str, default=f"images/robustness_params_single_vs_mixture.png")
    parser.add_argument("--dynamics-out", type=str, default=f"images/robustness_training_dynamics_single_vs_mixture.png")
    args = parser.parse_args()

    alphas = parse_alpha_list(args.alphas)
    results, tracked = sweep_robustness(
        alphas=alphas,
        delta=args.delta,
        steps=args.steps,
        lr=args.lr,
        n_components=args.n_components,
        kto_mode=args.kto_mode,
        track_alpha=args.track_alpha,
    )
    plot_results(results, scale_out_path=args.summary_out, n_components=args.n_components)
    plot_training_dynamics(tracked, out_path=args.dynamics_out, n_components=args.n_components)
    print_results_table(results, n_components=args.n_components)
    print(f"\nSaved plot to {args.summary_out}")
    if tracked is not None:
        print(f"Saved plot to {args.dynamics_out}")


if __name__ == "__main__":
    main()
