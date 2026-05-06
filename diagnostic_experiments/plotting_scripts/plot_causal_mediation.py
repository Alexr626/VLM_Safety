#!/usr/bin/env python3
"""
3-panel layer-wise Recovery Rate plots for the causal mediation
analysis. One PNG per tier, plus an overlay PNG when multiple tiers
are passed (top vs. bottom comparison).

Inputs
------
    recovery_rates_{tier}{pct?}.json files produced by
    causal_mediation_mssbench.py.

Outputs
-------
    {output_dir}/recovery_rate_{tier}{pct?}.png
    {output_dir}/recovery_rate_overlay.png   (if >1 tier file passed)
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPONENT_TITLES = {
    "hidden_state": "Layer block (residual stream after layer)",
    "mlp": "MLP submodule (pre-residual-add contribution)",
    "attn": "Self-attention submodule (pre-residual-add contribution)",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True,
                   help="One or more recovery_rates_*.json files.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--show_median", action="store_true",
                   help="Overlay dashed median line in addition to mean.")
    return p.parse_args()


def _plot_single(data: dict, out_path: Path, show_median: bool):
    components = data.get("components", ["hidden_state", "mlp", "attn"])
    n_layers = data["n_layers"]
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(len(components), 1, figsize=(8, 9), dpi=150,
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, comp in zip(axes, components):
        per = data["per_layer"][comp]
        means = np.asarray(per["mean"], dtype=float)
        ses = np.asarray(per["se"], dtype=float)
        ax.axhline(0.0, color="gray", linewidth=0.7, linestyle=":")
        ax.axhline(1.0, color="gray", linewidth=0.7, linestyle=":")
        ax.plot(layers, means, color="#4C78A8", linewidth=1.6,
                marker="o", markersize=3, label="mean RR")
        ax.fill_between(layers, means - ses, means + ses,
                        color="#4C78A8", alpha=0.18, label="±1 bootstrap SE")
        if show_median:
            medians = np.asarray(per["median"], dtype=float)
            ax.plot(layers, medians, color="#E45756", linewidth=1.2,
                    linestyle="--", label="median RR")
        ax.set_ylabel("Recovery Rate")
        ax.set_title(COMPONENT_TITLES.get(comp, comp), fontsize=10)
        ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("Layer index")
    label = data.get("tier_label", "")
    label_str = f"{label}, " if label else ""
    title = (f"{data['model_short']} — {label_str}"
             f"n_pairs={data['n_pairs']} "
             f"(kept after eps={data['rr_denominator_eps']}: "
             f"{data['n_pairs_kept_after_eps']})")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_overlay(datas: List[dict], out_path: Path):
    components = datas[0].get("components", ["hidden_state", "mlp", "attn"])
    n_layers = datas[0]["n_layers"]
    layers = np.arange(n_layers)
    palette = ["#4C78A8", "#E45756", "#59A14F", "#9D7660", "#B279A2"]

    fig, axes = plt.subplots(len(components), 1, figsize=(8, 9), dpi=150,
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, comp in zip(axes, components):
        ax.axhline(0.0, color="gray", linewidth=0.7, linestyle=":")
        ax.axhline(1.0, color="gray", linewidth=0.7, linestyle=":")
        for d, color in zip(datas, palette):
            per = d["per_layer"][comp]
            means = np.asarray(per["mean"], dtype=float)
            ses = np.asarray(per["se"], dtype=float)
            tier = d.get("tier_label", "") or "default"
            label = f"{tier} (n={d['n_pairs_kept_after_eps']})"
            ax.plot(layers, means, color=color, linewidth=1.6, marker="o",
                    markersize=3, label=label)
            ax.fill_between(layers, means - ses, means + ses,
                            color=color, alpha=0.15)
        ax.set_ylabel("Recovery Rate")
        ax.set_title(COMPONENT_TITLES.get(comp, comp), fontsize=10)
        ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("Layer index")
    fig.suptitle(f"{datas[0]['model_short']} — recovery rate (overlay)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datas = []
    for r in args.results:
        with open(r) as f:
            datas.append(json.load(f))

    plottable = []
    for d in datas:
        if d.get("per_layer") is None:
            print(f"  [skip] {d.get('tier_label', '')}: per_layer is null "
                  f"(n_pairs_kept_after_eps={d.get('n_pairs_kept_after_eps', 0)})")
            continue
        label = d.get("tier_label", "")
        fname = f"recovery_rate_{label}.png" if label else "recovery_rate.png"
        out_path = out_dir / fname
        _plot_single(d, out_path, args.show_median)
        plottable.append(d)

    if len(plottable) >= 2:
        _plot_overlay(plottable, out_dir / "recovery_rate_overlay.png")


if __name__ == "__main__":
    main()
