#!/usr/bin/env python3
"""Plot Augmented Baseline: TT vs CT Projection Gaps"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_DIAGNOSTIC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DIAGNOSTIC_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from src.model import _normalize_model_name

_EXPERIMENT_NAME = "augmented_baseline"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    results_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME / "outputs" / "results"
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "augmented_baseline_stats.json") as f:
        data = json.load(f)
    data = [d for d in data if d["layer"] > 0]

    layers = np.array([d["layer"] for d in data])
    x = np.arange(len(layers))

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
    })
    COLOR_TT, COLOR_CT = "#2563eb", "#dc2626"

    def _save(fig, name):
        path = out_dir / name
        fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {path}")
        plt.close(fig)

    has_ct = "ct_gap" in data[0]
    tt_gaps = np.array([d.get("tt_gap", 0) or 0 for d in data])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, tt_gaps, "o-", color=COLOR_TT, markersize=5, linewidth=1.8,
            label="TT (caption + text)", alpha=0.85)
    if has_ct:
        ct_gaps = np.array([d.get("ct_gap", 0) or 0 for d in data])
        ax.plot(x, ct_gaps, "s-", color=COLOR_CT, markersize=5, linewidth=1.8,
                label="CT (cohesive text)", alpha=0.85)
    ax.axhline(0, color="#888", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Projection Gap (SSS - SSU)")
    ax.set_title(f"Baseline Projection Gap: TT vs CT Representations\n{model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    _save(fig, "gap_comparison_tt_ct.png")

    # P-values
    tt_p = np.clip([d.get("tt_p_mannwhitney", 1.0) or 1.0 for d in data], 1e-300, 1.0)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, tt_p, "o-", color=COLOR_TT, markersize=5, linewidth=1.5, label="TT p-value", alpha=0.85)
    if has_ct:
        ct_p = np.clip([d.get("ct_p_mannwhitney", 1.0) or 1.0 for d in data], 1e-300, 1.0)
        ax.plot(x, ct_p, "s-", color=COLOR_CT, markersize=5, linewidth=1.5, label="CT p-value", alpha=0.85)
    ax.axhline(0.05, color="#ef4444", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(0.001, color="#f97316", linestyle=":", linewidth=1, alpha=0.4)
    ax.set_yscale("log")
    ax.set_ylabel("p-value (log)")
    ax.set_title(f"Statistical Significance: SSS vs SSU ({model_name})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    _save(fig, "p_values_tt_ct.png")


if __name__ == "__main__":
    main()
