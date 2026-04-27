#!/usr/bin/env python3
"""
Plot Compositional Safety Shift Projections
============================================
Side-by-side comparison: semantic safety direction projections vs
compositional safety direction projections across layers, for both VL shift
(from aggregate_stats.json) and TT baseline (from tt_baseline_projections.json).

Requires vl_activation_shift.py and sanity_check_tt_baseline.py to have
been run with --compositional_safety_dir.
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
    results_dir = (_DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME /
                   "outputs" / "results")
    out_dir = results_dir / "plots_compositional_safety"
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_path = results_dir / "vl_activation_shift" / "aggregate_stats.json"
    tt_path = results_dir / "sanity_check_tt_baseline" / "tt_baseline_projections.json"

    if not agg_path.exists():
        print(f"ERROR: {agg_path} not found. "
              "Run vl_activation_shift.py --compositional_safety_dir first.")
        sys.exit(1)

    with open(agg_path) as f:
        agg = json.load(f)
    agg = [d for d in agg if d.get("n_sss", 0) > 0]

    if "SSS_mean_comp_proj" not in agg[0]:
        print("ERROR: aggregate_stats.json does not contain comp_* fields.")
        print("  Re-run vl_activation_shift.py with --compositional_safety_dir")
        sys.exit(1)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
    })
    COLOR_SEMANTIC = "#2563eb"
    COLOR_COMP = "#dc2626"

    def _save(fig, name):
        path = out_dir / name
        fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {path}")
        plt.close(fig)

    # ── Plot 1: VL shift — projection gap (SSS - SSU) semantic vs compositional ─
    layers = np.array([d["layer"] for d in agg])
    x = np.arange(len(layers))

    semantic_gap = np.array([(d["SSS_mean_proj"] or 0) - (d["SSU_mean_proj"] or 0) for d in agg])
    comp_gap = np.array([(d["SSS_mean_comp_proj"] or 0) - (d["SSU_mean_comp_proj"] or 0) for d in agg])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, semantic_gap, "o-", color=COLOR_SEMANTIC,
            label="semantic safety s^l", linewidth=1.8, markersize=5)
    ax.plot(x, comp_gap, "s-", color=COLOR_COMP,
            label="compositional safety c^l", linewidth=1.8, markersize=5)
    ax.axhline(0, color="#888", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("VL-shift Projection Gap (SSS - SSU)")
    ax.set_title(f"VL Shift Projection: semantic safety vs compositional safety direction\n{model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, "vl_shift_projection_gap.png")

    # ── Plot 2: cosine means ────────────────────────────────────────────────
    sss_c = np.array([d["SSS_mean_cosine"] or 0 for d in agg])
    ssu_c = np.array([d["SSU_mean_cosine"] or 0 for d in agg])
    sss_cc = np.array([d["SSS_mean_comp_cosine"] or 0 for d in agg])
    ssu_cc = np.array([d["SSU_mean_comp_cosine"] or 0 for d in agg])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, sss_c - ssu_c, "o-", color=COLOR_SEMANTIC,
            label="semantic safety s^l (SSS-SSU)", linewidth=1.8)
    ax.plot(x, sss_cc - ssu_cc, "s-", color=COLOR_COMP,
            label="compositional safety c^l (SSS-SSU)", linewidth=1.8)
    ax.axhline(0, color="#888", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Cosine Gap (SSS - SSU)")
    ax.set_title(f"VL-Shift Cosine Alignment: semantic safety vs compositional safety\n{model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, "vl_shift_cosine_gap.png")

    # ── Plot 3: TT baseline comparison (if available) ──────────────────────
    if tt_path.exists():
        with open(tt_path) as f:
            tt = json.load(f)
        tt_layers = np.array([d["layer"] for d in tt])
        xx = np.arange(len(tt_layers))
        has_comp_tt = "SSS_mean_tt_comp_projection" in tt[0]

        tt_semantic_gap = np.array([d["SSS_mean_tt_projection"] - d["SSU_mean_tt_projection"]
                                    for d in tt])
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(xx, tt_semantic_gap, "o-", color=COLOR_SEMANTIC,
                label="semantic safety s^l", linewidth=1.8)
        if has_comp_tt:
            tt_comp_gap = np.array([d["SSS_mean_tt_comp_projection"] - d["SSU_mean_tt_comp_projection"]
                                    for d in tt])
            ax.plot(xx, tt_comp_gap, "s-", color=COLOR_COMP,
                    label="compositional safety c^l", linewidth=1.8)
        ax.axhline(0, color="#888", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("TT Baseline Projection Gap (SSS - SSU)")
        ax.set_title(f"TT Baseline Projection: semantic safety vs compositional safety\n{model_name}")
        ax.set_xticks(xx)
        ax.set_xticklabels([str(l) for l in tt_layers], fontsize=8)
        ax.set_xlabel("Transformer Layer")
        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()
        _save(fig, "tt_baseline_projection_gap.png")


if __name__ == "__main__":
    main()
