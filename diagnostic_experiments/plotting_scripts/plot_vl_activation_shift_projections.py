"""
Plot ShiftDC Diagnostic Results: SSS vs SSU Activation Shifts
"""

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

_EXPERIMENT_NAME = "shift_dc"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    experiment_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME
    results_dir = experiment_dir / "outputs" / "results"
    stats_path = results_dir / "vl_activation_shift" / "aggregate_stats.json"
    out_dir = results_dir / "vl_activation_shift" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not stats_path.exists():
        print(f"ERROR: {stats_path} not found. Run vl_activation_shift.py first.")
        sys.exit(1)

    with open(stats_path) as f:
        data = json.load(f)
    data = [d for d in data if d["n_sss"] > 0]

    layers = np.array([d["layer"] for d in data])
    sss_cos = np.array([d["SSS_mean_cosine"] for d in data])
    ssu_cos = np.array([d["SSU_mean_cosine"] for d in data])
    sss_proj = np.array([d["SSS_mean_proj"] for d in data])
    ssu_proj = np.array([d["SSU_mean_proj"] for d in data])
    p_cos = np.array([d["p_cosine"] for d in data])
    p_proj = np.array([d["p_proj"] for d in data])
    n_sss = data[0]["n_sss"]
    n_ssu = data[0]["n_ssu"]

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10.5, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc", "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.5,
    })
    COLOR_SSS = "#2563eb"
    COLOR_SSU = "#dc2626"
    SAFETY_BG = "#fef3c7"
    bar_width = 0.38
    x = np.arange(len(layers))
    title_model = model_name

    def _save(fig, name):
        path = out_dir / name
        fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved to {path}")
        plt.close(fig)

    # Cosine Similarity
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axvspan(5.5, 14.5, color=SAFETY_BG, alpha=0.5, zorder=0)
    ax.bar(x - bar_width/2, sss_cos, bar_width, color=COLOR_SSS, alpha=0.8,
           label="SSS (safe output)", edgecolor="white", linewidth=0.3)
    ax.bar(x + bar_width/2, ssu_cos, bar_width, color=COLOR_SSU, alpha=0.8,
           label="SSU (unsafe output)", edgecolor="white", linewidth=0.3)
    ax.set_ylabel("Mean Cosine Similarity\nwith Safety Direction")
    ax.set_title(f"Directional Alignment of Modality Shift with Safety Direction\n"
                 f"{title_model}  ·  SSS (n={n_sss}) vs SSU (n={n_ssu})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    _save(fig, "cosine_similarity.png")

    # Projection Magnitude
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axvspan(5.5, 14.5, color=SAFETY_BG, alpha=0.5, zorder=0)
    ax.bar(x - bar_width/2, sss_proj, bar_width, color=COLOR_SSS, alpha=0.8,
           label="SSS", edgecolor="white", linewidth=0.3)
    ax.bar(x + bar_width/2, ssu_proj, bar_width, color=COLOR_SSU, alpha=0.8,
           label="SSU", edgecolor="white", linewidth=0.3)
    ax.set_ylabel("Mean Projection Magnitude\nonto Safety Direction")
    ax.set_title(f"Absolute Displacement Along Safety Direction\n"
                 f"{title_model}  ·  SSS (n={n_sss}) vs SSU (n={n_ssu})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    _save(fig, "projection_magnitude.png")

    # P-values
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axvspan(5.5, 14.5, color=SAFETY_BG, alpha=0.5, zorder=0)
    p_cos_plot = np.clip(p_cos, 1e-70, 1.0)
    p_proj_plot = np.clip(p_proj, 1e-70, 1.0)
    ax.plot(x, p_cos_plot, "o-", color="#7c3aed", markersize=4, linewidth=1.5,
            label="p-value (cosine)", alpha=0.85)
    ax.plot(x, p_proj_plot, "s-", color="#0891b2", markersize=4, linewidth=1.5,
            label="p-value (projection)", alpha=0.85)
    ax.axhline(y=0.05, color="#ef4444", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(y=0.001, color="#f97316", linestyle=":", linewidth=1, alpha=0.4)
    ax.set_yscale("log")
    ax.set_ylabel("p-value (log scale)")
    ax.set_title(f"Statistical Significance of SSS vs SSU Difference (t-test)\n"
                 f"{title_model}  ·  SSS (n={n_sss}) vs SSU (n={n_ssu})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.set_ylim(1e-65, 2)
    fig.tight_layout()
    _save(fig, "p_values.png")


if __name__ == "__main__":
    main()
