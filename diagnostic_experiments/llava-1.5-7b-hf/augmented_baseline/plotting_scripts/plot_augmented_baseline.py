#!/usr/bin/env python3
"""
Plot Augmented Baseline: TT vs CT Projection Gaps
===================================================
Generates:
  1. Multi-line plot: TT vs CT projection gaps across layers
  2. (If behavioral labels available) Distribution plot for SSU refused vs complied

Reads from: ../outputs/results/augmented_baseline_stats.json
Saves to:   ../outputs/results/plots/
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent
_RESULTS_DIR = _EXPERIMENT_DIR / "outputs" / "results"
_OUTPUT_DIR = _RESULTS_DIR / "plots"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
with open(_RESULTS_DIR / "augmented_baseline_stats.json") as f:
    data = json.load(f)

# Filter degenerate layer 0
data = [d for d in data if d["layer"] > 0]

layers = np.array([d["layer"] for d in data])
x = np.arange(len(layers))

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

COLOR_TT = "#2563eb"    # Blue
COLOR_CT = "#dc2626"    # Red
COLOR_GAP = "#059669"   # Green
SAFETY_BG = "#fef3c7"

safety_lo = np.searchsorted(layers, 6) - 0.5
safety_hi = np.searchsorted(layers, 14) + 0.5
safety_center = (np.searchsorted(layers, 6) + np.searchsorted(layers, 14)) / 2


def _save(fig, name):
    path = _OUTPUT_DIR / name
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {path}")
    plt.close(fig)


def _annotate_safety(ax, y_frac=0.95):
    ax.axvspan(safety_lo, safety_hi, color=SAFETY_BG, alpha=0.5, zorder=0)
    ylim = ax.get_ylim()
    ax.text(safety_center, ylim[0] + (ylim[1] - ylim[0]) * y_frac,
            "safety-critical layers (6-14)",
            ha="center", va="top", fontsize=8.5, color="#b45309", fontstyle="italic")


# ── Plot 1: Gap comparison (TT vs CT) ───────────────────────────────────────
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
ax.set_title("Baseline Projection Gap: TT vs CT Representations\n"
             "LLaVA-1.5-7B  ·  Safety Direction Projection (SSS - SSU)")
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
_annotate_safety(ax)
fig.tight_layout()
_save(fig, "gap_comparison_tt_ct.png")


# ── Plot 2: P-values ────────────────────────────────────────────────────────
tt_p = np.array([d.get("tt_p_mannwhitney", 1.0) or 1.0 for d in data])
tt_p = np.clip(tt_p, 1e-300, 1.0)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(x, tt_p, "o-", color=COLOR_TT, markersize=5, linewidth=1.5,
        label="TT p-value", alpha=0.85)

if has_ct:
    ct_p = np.array([d.get("ct_p_mannwhitney", 1.0) or 1.0 for d in data])
    ct_p = np.clip(ct_p, 1e-300, 1.0)
    ax.plot(x, ct_p, "s-", color=COLOR_CT, markersize=5, linewidth=1.5,
            label="CT p-value", alpha=0.85)

ax.axhline(0.05, color="#ef4444", linestyle=":", linewidth=1, alpha=0.6)
ax.text(len(x) - 0.5, 0.05, "p = 0.05", fontsize=8, color="#ef4444", va="bottom", ha="right")
ax.axhline(0.001, color="#f97316", linestyle=":", linewidth=1, alpha=0.4)
ax.text(len(x) - 0.5, 0.001, "p = 0.001", fontsize=8, color="#f97316", va="bottom", ha="right")

ax.set_yscale("log")
ax.set_ylabel("p-value (log scale)")
ax.set_title("Statistical Significance: SSS vs SSU Baseline Gap\n"
             "LLaVA-1.5-7B  ·  Mann-Whitney U Test")
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
_annotate_safety(ax)
fig.tight_layout()
_save(fig, "p_values_tt_ct.png")


# ── Plot 3: Behavioral split (if available) ─────────────────────────────────
has_behavioral = any("tt_ssu_refused_mean" in d for d in data)

if has_behavioral:
    for suffix, color, label in [("tt", COLOR_TT, "TT"), ("ct", COLOR_CT, "CT")]:
        refused_means = [d.get(f"{suffix}_ssu_refused_mean") for d in data]
        complied_means = [d.get(f"{suffix}_ssu_complied_mean") for d in data]

        if not any(v is not None for v in refused_means):
            continue

        refused_arr = np.array([v if v is not None else float("nan") for v in refused_means])
        complied_arr = np.array([v if v is not None else float("nan") for v in complied_means])

        fig, ax = plt.subplots(figsize=(14, 5))
        valid = ~(np.isnan(refused_arr) | np.isnan(complied_arr))
        ax.plot(x[valid], refused_arr[valid], "^-", color="#ef4444", markersize=6,
                linewidth=1.5, label=f"SSU Refused ({label})", alpha=0.85)
        ax.plot(x[valid], complied_arr[valid], "v-", color="#22c55e", markersize=6,
                linewidth=1.5, label=f"SSU Complied ({label})", alpha=0.85)

        ax.axhline(0, color="#888", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(f"Mean {label} Projection onto Safety Direction")
        ax.set_title(f"SSU Behavioral Split: Refused vs Complied ({label} Activations)\n"
                     f"LLaVA-1.5-7B  ·  Within SSU Samples Only")
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers], fontsize=8)
        ax.set_xlabel("Transformer Layer")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        _annotate_safety(ax)
        fig.tight_layout()
        _save(fig, f"behavioral_split_{suffix}.png")
