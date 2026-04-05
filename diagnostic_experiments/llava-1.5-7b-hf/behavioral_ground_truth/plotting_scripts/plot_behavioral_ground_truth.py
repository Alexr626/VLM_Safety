#!/usr/bin/env python3
"""
Plot Behavioral Ground Truth: Refusal Rates
=============================================
Generates:
  1. Grouped bar chart: refusal rate by group (SSS/SSU) x condition (VL/TT/CT)
  2. Per-category heatmap of SSU refusal rates across conditions

Reads from: ../outputs/results/refusal_summary.json
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
with open(_RESULTS_DIR / "refusal_summary.json") as f:
    summary = json.load(f)

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

COLOR_SSS = "#2563eb"
COLOR_SSU = "#dc2626"


def _save(fig, name):
    path = _OUTPUT_DIR / name
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {path}")
    plt.close(fig)


# ── Plot 1: Grouped bar chart ───────────────────────────────────────────────
conditions = ["VL", "TT", "CT"]
x = np.arange(len(conditions))
bar_width = 0.35

sss_rates = [summary["SSS"].get(f"refusal_rate_{c.lower()}", 0) for c in conditions]
ssu_rates = [summary["SSU"].get(f"refusal_rate_{c.lower()}", 0) for c in conditions]

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - bar_width / 2, sss_rates, bar_width,
               color=COLOR_SSS, alpha=0.85, label="SSS (safe output)", edgecolor="white")
bars2 = ax.bar(x + bar_width / 2, ssu_rates, bar_width,
               color=COLOR_SSU, alpha=0.85, label="SSU (unsafe output)", edgecolor="white")

# Value labels
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.0%}", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Refusal Rate")
ax.set_title("Model Refusal Rates by Group and Input Condition\n"
             "LLaVA-1.5-7B  ·  HoliSafe-Bench SSS vs SSU")
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylim(0, max(max(sss_rates), max(ssu_rates)) * 1.2 + 0.05)
ax.legend(loc="upper right", framealpha=0.9)
ax.set_xlabel("Input Condition")
fig.tight_layout()
_save(fig, "refusal_rates_grouped.png")


# ── Plot 2: Per-category heatmap (SSU only) ─────────────────────────────────
cat_data = summary.get("ssu_per_category", {})
if cat_data:
    categories = sorted(cat_data.keys())
    matrix = np.array([
        [cat_data[c].get(f"refusal_rate_{cond}", 0) for cond in ["vl", "tt", "ct"]]
        for c in categories
    ])

    fig, ax = plt.subplots(figsize=(8, max(4, len(categories) * 0.45)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(3))
    ax.set_xticklabels(["VL", "TT", "CT"])
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=9)

    # Annotate cells
    for i in range(len(categories)):
        for j in range(3):
            val = matrix[i, j]
            color = "white" if val > 0.6 or val < 0.2 else "black"
            n = cat_data[categories[i]]["n"]
            ax.text(j, i, f"{val:.0%}\n(n={n})", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title("SSU Refusal Rate by Harm Category and Input Condition\n"
                 "LLaVA-1.5-7B  ·  HoliSafe-Bench")
    ax.set_xlabel("Input Condition")
    fig.colorbar(im, ax=ax, label="Refusal Rate", shrink=0.8)
    fig.tight_layout()
    _save(fig, "ssu_category_heatmap.png")
else:
    print("No per-category data available; skipping heatmap.")
