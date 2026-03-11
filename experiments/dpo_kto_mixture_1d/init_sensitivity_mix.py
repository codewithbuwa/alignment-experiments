import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import MixtureDPOKTOConfig, ExperimentConfig
from src.montage import make_montage
from src.ref_policies import make_reference_mixture
from src.train_mix import train_dpo_mixture, train_kto_mixture
from src.utils import ensure_dir, export_report_figure, get_timestamp, save_json, set_seed, update_latest_paths


def _make_init_means(center: float, spread: float, n_components: int):
    if n_components == 1:
        return [center]
    return torch.linspace(center - spread, center + spread, n_components).tolist()


def _weights_from_logits(logits_hist):
    weights_hist = []
    for logits in logits_hist:
        w = torch.softmax(torch.tensor(logits), dim=0).tolist()
        weights_hist.append(w)
    return weights_hist


def _mixture_pdf(y, mus, sigmas, logits):
    weights = torch.softmax(torch.tensor(logits), dim=0)
    pdf = torch.zeros_like(y)
    for w, m, s in zip(weights, mus, sigmas):
        pdf = pdf + w * (1.0 / (s * torch.sqrt(torch.tensor(2 * torch.pi)))) * \
            torch.exp(-0.5 * ((y - m) / s) ** 2)
    return pdf


def _format_components(mus, sigmas, logits):
    weights = torch.softmax(torch.tensor(logits), dim=0)
    return "\n".join(
        [f"C{k}: mu={mus[k]:.2f}, sigma={sigmas[k]:.2f}, w={weights[k].item():.2f}" for k in range(len(mus))]
    )


def _plot_density(y_vals, ref_pdf, dpo_pdf, kto_pdf, title, output_path, annotation=None):
    plt.figure()
    plt.plot(y_vals.numpy(), ref_pdf.numpy(), label="Reference")
    plt.plot(y_vals.numpy(), dpo_pdf.numpy(), label="DPO")
    plt.plot(y_vals.numpy(), kto_pdf.numpy(), label="KTO")
    plt.fill_between(y_vals.numpy(), 0, ((y_vals >= 5.5) & (y_vals <= 8.5)).float().numpy() * 0.5, alpha=0.2, label="Desirable Zone")
    plt.legend()
    plt.title(title)
    plt.xlabel("y")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    if annotation is not None:
        plt.gca().text(
            0.02,
            0.98,
            annotation,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
    plt.savefig(output_path)
    plt.close()


def _plot_weights(weights_hist, title, output_path):
    plt.figure()
    steps = list(range(len(weights_hist)))
    k = len(weights_hist[0])
    for i in range(k):
        plt.plot(steps, [w[i] for w in weights_hist], label=f"w{i}")
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("weight")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()


def _delete_paths(paths):
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def main():
    cfg = MixtureDPOKTOConfig()
    base_cfg = ExperimentConfig()
    set_seed(cfg.seed)

    timestamp = get_timestamp()
    output_root = os.path.join("results", "dpo_kto_mixture_1d", f"init_sensitivity_{timestamp}")
    figures_dir = os.path.join(output_root, "figures")
    runs_dir = os.path.join(output_root, "runs")
    ensure_dir(figures_dir)
    ensure_dir(runs_dir)

    ref_policy = make_reference_mixture(cfg.n_components, base_cfg.mu_ref, base_cfg.sigma_ref, cfg.device)
    ref_pdf = torch.exp(ref_policy.log_prob(torch.linspace(cfg.y_min, cfg.y_max, 1000))).detach()

    scenarios = {
        "around": _make_init_means(cfg.target, 0.5, cfg.n_components),
        "left": _make_init_means(cfg.target - 4.0, 0.5, cfg.n_components),
        "right": _make_init_means(cfg.target + 4.0, 0.5, cfg.n_components),
    }

    summary = {"dpo": {}, "kto": {}}

    y_vals = torch.linspace(cfg.y_min, cfg.y_max, 1000)

    dpo_weight_paths = []
    kto_weight_paths = []

    for name, means in scenarios.items():
        cfg_init = MixtureDPOKTOConfig(**cfg.__dict__)
        cfg_init.init_means = means
        cfg_init.init_sigmas = [base_cfg.sigma_ref for _ in range(cfg.n_components)]
        cfg_init.init_logits = [0.0 for _ in range(cfg.n_components)]

        dpo_policy, _, dpo_hist, _ = train_dpo_mixture(ref_policy, cfg_init)
        kto_policy, _, kto_hist, _ = train_kto_mixture(ref_policy, cfg_init)

        dpo_weights = _weights_from_logits(dpo_hist["logits"])
        kto_weights = _weights_from_logits(kto_hist["logits"])

        dpo_w_path = os.path.join(figures_dir, f"dpo_weights_{name}.png")
        kto_w_path = os.path.join(figures_dir, f"kto_weights_{name}.png")

        _plot_weights(dpo_weights, f"DPO Weight Evolution ({name})", dpo_w_path)
        _plot_weights(kto_weights, f"KTO Weight Evolution ({name})", kto_w_path)

        dpo_weight_paths.append(dpo_w_path)
        kto_weight_paths.append(kto_w_path)

        quartiles = [0, int(0.25 * (cfg.steps - 1)), int(0.5 * (cfg.steps - 1)), int(0.75 * (cfg.steps - 1)), cfg.steps - 1]
        pct = [0, 25, 50, 75, 100]
        quartile_paths = []
        for idx, p in zip(quartiles, pct):
            dpo_pdf_q = _mixture_pdf(y_vals, dpo_hist["mus"][idx], dpo_hist["sigmas"][idx], dpo_hist["logits"][idx])
            kto_pdf_q = _mixture_pdf(y_vals, kto_hist["mus"][idx], kto_hist["sigmas"][idx], kto_hist["logits"][idx])

            q_path = os.path.join(figures_dir, f"density_{name}_q{p}.png")
            _plot_density(
                y_vals,
                ref_pdf,
                dpo_pdf_q,
                kto_pdf_q,
                f"Mixture Density ({name}) at {p}%",
                q_path,
                annotation=(
                    "DPO\n"
                    f"{_format_components(dpo_hist['mus'][idx], dpo_hist['sigmas'][idx], dpo_hist['logits'][idx])}\n"
                    "KTO\n"
                    f"{_format_components(kto_hist['mus'][idx], kto_hist['sigmas'][idx], kto_hist['logits'][idx])}"
                ),
            )
            quartile_paths.append(q_path)

        montage_path = os.path.join(figures_dir, f"density_{name}_quartiles_montage.png")
        make_montage(quartile_paths, montage_path, cols=3)
        _delete_paths(quartile_paths)

        export_report_figure(
            montage_path,
            f"mix_init_density_{name}.png",
        )

        summary["dpo"][name] = {
            "init_means": means,
            "final_weights": dpo_weights[-1],
        }
        summary["kto"][name] = {
            "init_means": means,
            "final_weights": kto_weights[-1],
        }

    make_montage(dpo_weight_paths, os.path.join(figures_dir, "dpo_weights_montage.png"), cols=3)
    make_montage(kto_weight_paths, os.path.join(figures_dir, "kto_weights_montage.png"), cols=3)
    _delete_paths(dpo_weight_paths)
    _delete_paths(kto_weight_paths)

    export_report_figure(
        os.path.join(figures_dir, "dpo_weights_montage.png"),
        "mix_dpo_weights_montage.png",
    )
    export_report_figure(
        os.path.join(figures_dir, "kto_weights_montage.png"),
        "mix_kto_weights_montage.png",
    )

    save_json(os.path.join(output_root, "config.json"), cfg.__dict__)
    save_json(os.path.join(runs_dir, "init_summary.json"), summary)

    update_latest_paths("mix_init_sensitivity", output_root)

    print(f"Saved mixture init sensitivity results to: {output_root}")


if __name__ == "__main__":
    main()
