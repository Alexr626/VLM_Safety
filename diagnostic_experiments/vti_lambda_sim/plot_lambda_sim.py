#!/usr/bin/env python3
"""Plot lambda_sim distributions from gated_rotation diagnostic runs."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import _normalize_model_name
from src.paths import diagnostic_plots_dir, diagnostic_results_dir

EXPERIMENT = "vti_lambda_sim"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--hook_site", default="mlp", choices=["mlp", "layer"])
    p.add_argument("--pope_split", default="random")
    return p.parse_args()


def main():
    args = parse_args()
    model_short = _normalize_model_name(args.model)
    results_dir = diagnostic_results_dir(EXPERIMENT, model_short)
    data_path = results_dir / f"lambda_sim_{args.hook_site}_{args.pope_split}.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run compute_lambda_sim.py first.")

    with open(data_path) as f:
        per_sample = json.load(f)

    all_lam: list[float] = []
    by_layer: dict[int, list[float]] = defaultdict(list)
    token_lams: Counter = Counter()

    for rec in per_sample:
        for row in rec.get("lambda_records", []):
            lam = row["lambda_sim"]
            layer = row["layer"]
            all_lam.append(lam)
            by_layer[layer].append(lam)
            tok = row.get("token") or "<unk>"
            token_lams[tok] += lam

    plots_dir = diagnostic_plots_dir(EXPERIMENT, model_short)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_lam, bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("lambda_sim")
    ax.set_ylabel("count")
    ax.set_title(f"lambda_sim distribution ({model_short}, {args.hook_site})")
    fig.tight_layout()
    fig.savefig(plots_dir / f"lambda_sim_hist_{args.hook_site}_{args.pope_split}.png", dpi=150)
    plt.close(fig)

    layers = sorted(by_layer)
    means = [float(np.mean(by_layer[l])) for l in layers]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, means, marker="o", markersize=3)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean lambda_sim")
    ax.set_title(f"mean lambda_sim by layer ({model_short})")
    fig.tight_layout()
    fig.savefig(plots_dir / f"lambda_sim_by_layer_{args.hook_site}_{args.pope_split}.png", dpi=150)
    plt.close(fig)

    top_tokens = token_lams.most_common(25)
    if top_tokens:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [t for t, _ in top_tokens]
        vals = [v / token_lams[t] for t, v in top_tokens]
        ax.barh(labels[::-1], vals[::-1])
        ax.set_xlabel("mean lambda_sim (summed over occurrences)")
        ax.set_title(f"top tokens by cumulative lambda_sim ({model_short})")
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"lambda_sim_top_tokens_{args.hook_site}_{args.pope_split}.png",
            dpi=150,
        )
        plt.close(fig)

    summary = {
        "model": model_short,
        "n_lambda_records": len(all_lam),
        "lambda_sim_mean": float(np.mean(all_lam)) if all_lam else None,
        "lambda_sim_std": float(np.std(all_lam)) if all_lam else None,
        "mean_by_layer": {str(l): float(np.mean(by_layer[l])) for l in layers},
    }
    with open(plots_dir / f"lambda_sim_summary_{args.hook_site}_{args.pope_split}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote plots -> {plots_dir}")


if __name__ == "__main__":
    main()
