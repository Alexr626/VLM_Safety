#!/usr/bin/env python3
"""
Plot Direction Comparison: Combinatorial vs Content Safety Direction
=====================================================================
Generates:
  1. Layer-wise cosine similarity: combinatorial vs content safety direction
  2. Layer-wise effective rank of SSU-vs-SSS
  3. Layer-wise subspace overlap (if available)

Reads from: ../outputs/results/direction_comparison.json
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
with open(_RESULTS_DIR / "direction_comparison.json") as f:
    data = json.load(f)

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

COLOR_COS = "#7c3aed"     # Purple
COLOR_RANK = "#059669"    # Green
COLOR_OVERLAP = "#2563eb"  # Blue
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


# ── Plot 1: Cosine similarity ───────────────────────────────────────────────
cos_vals = np.array([d["cosine_sim_comb_vs_catqa"] for d in data])

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x, np.abs(cos_vals), width=0.6, color=COLOR_COS, alpha=0.8,
       edgecolor="white", linewidth=0.3)
ax.axhline(0, color="#888", linewidth=0.8, alpha=0.5)
ax.set_ylabel("|Cosine Similarity|")
ax.set_title("Combinatorial vs Content Safety Direction: Cosine Similarity\n"
             "LLaVA-1.5-7B  ·  |cos(c^l, s^l)| per Layer")
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.set_ylim(0, 1.05)
_annotate_safety(ax)

# Annotate mean in safety-critical region
safety_mask = (layers >= 6) & (layers <= 14)
mean_cos = np.abs(cos_vals[safety_mask]).mean()
ylim = ax.get_ylim()
ax.text(safety_center, ylim[1] * 0.88,
        f"mean |cos| (L6-14): {mean_cos:.3f}",
        ha="center", va="top", fontsize=9, fontweight="bold", color="#5b21b6",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLOR_COS, alpha=0.9))
fig.tight_layout()
_save(fig, "cosine_similarity.png")


# ── Plot 2: Effective rank ───────────────────────────────────────────────────
ranks = np.array([d["effective_rank"] for d in data])

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x, ranks, width=0.6, color=COLOR_RANK, alpha=0.8,
       edgecolor="white", linewidth=0.3)
ax.set_ylabel("Effective Rank (tau=0.9)")
ax.set_title("Effective Rank of SSU-vs-SSS Activation Space\n"
             "LLaVA-1.5-7B  ·  Combined (SSS_train + SSU_train) TT Activations")
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")
_annotate_safety(ax)
fig.tight_layout()
_save(fig, "effective_rank.png")


# ── Plot 3: Subspace overlap (if available) ──────────────────────────────────
has_overlap = any("subspace_overlap_top5" in d for d in data)
if has_overlap:
    overlap = np.array([d.get("subspace_overlap_top5", float("nan")) for d in data])
    valid = ~np.isnan(overlap)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x[valid], overlap[valid], "o-", color=COLOR_OVERLAP, markersize=6,
            linewidth=1.8, alpha=0.85)
    ax.set_ylabel("Subspace Overlap (top-5 PCs)")
    ax.set_title("Subspace Overlap: Combinatorial vs Content Safety Subspace\n"
                 "LLaVA-1.5-7B  ·  Mean Cosine of Principal Angles (top-5 PCs)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylim(0, 1.05)
    _annotate_safety(ax)
    fig.tight_layout()
    _save(fig, "subspace_overlap.png")
