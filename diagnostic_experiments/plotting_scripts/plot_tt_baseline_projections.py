"""
Plot Sanity Check: Text-Only Baseline Positions Along Safety Direction
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
    results_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME / "outputs" / "results"
    data_path = results_dir / "sanity_check_tt_baseline" / "tt_baseline_projections.json"
    out_dir = results_dir / "sanity_check_tt_baseline" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    layers = np.array([d["layer"] for d in data])
    sss_proj = np.array([d["SSS_mean_tt_projection"] for d in data])
    ssu_proj = np.array([d["SSU_mean_tt_projection"] for d in data])

    has_cosine = "SSS_mean_tt_cosine" in data[0]
    sss_cos = np.array([d["SSS_mean_tt_cosine"] for d in data]) if has_cosine else None
    ssu_cos = np.array([d["SSU_mean_tt_cosine"] for d in data]) if has_cosine else None
    gap_proj = sss_proj - ssu_proj
    gap_cos = (sss_cos - ssu_cos) if has_cosine else None

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    })
    COLOR_SSS, COLOR_SSU, COLOR_GAP = "#2563eb", "#dc2626", "#059669"
    SAFETY_BG = "#fef3c7"
    bar_width = 0.38
    x = np.arange(len(layers))
    safety_lo = np.searchsorted(layers, 6) - 0.5
    safety_hi = np.searchsorted(layers, 14) + 0.5

    def _save(fig, name):
        path = out_dir / name
        fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {path}")
        plt.close(fig)

    def _plot_bars(sss_vals, ssu_vals, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axvspan(safety_lo, safety_hi, color=SAFETY_BG, alpha=0.5, zorder=0)
        ax.bar(x - bar_width/2, sss_vals, bar_width, color=COLOR_SSS, alpha=0.8,
               label="SSS", edgecolor="white", linewidth=0.3)
        ax.bar(x + bar_width/2, ssu_vals, bar_width, color=COLOR_SSU, alpha=0.8,
               label="SSU", edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="#888", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers], fontsize=8)
        ax.set_xlabel("Transformer Layer")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        fig.tight_layout()
        _save(fig, fname)

    def _plot_gap(gap_vals, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axvspan(safety_lo, safety_hi, color=SAFETY_BG, alpha=0.5, zorder=0)
        colors = [COLOR_GAP if g > 0 else "#b91c1c" for g in gap_vals]
        ax.bar(x, gap_vals, width=0.6, color=colors, alpha=0.8,
               edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="#888", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers], fontsize=8)
        ax.set_xlabel("Transformer Layer")
        fig.tight_layout()
        _save(fig, fname)

    st = f"{model_name}"
    _plot_bars(sss_proj, ssu_proj, "Mean TT Projection onto s^l",
               f"TT Baseline Projection ({st})", "raw_projections.png")
    _plot_gap(gap_proj, "Gap (SSS - SSU)",
              f"TT Baseline Projection Gap ({st})", "projection_gap.png")
    if has_cosine:
        _plot_bars(sss_cos, ssu_cos, "Mean Cosine with s^l",
                   f"TT Baseline Cosine ({st})", "raw_cosine.png")
        _plot_gap(gap_cos, "Gap (SSS - SSU)",
                  f"TT Baseline Cosine Gap ({st})", "cosine_gap.png")


if __name__ == "__main__":
    main()
