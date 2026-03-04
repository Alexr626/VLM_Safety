"""
Plot Sanity Check: Text-Only Baseline Positions Along Safety Direction
Generates individual plots showing whether SSU text-only activations already
sit closer to the "unsafe" side of the safety boundary than SSS text-only
activations.

Reads from: ../sanity_check_tt_baseline/tt_baseline_projections.json
            ../sanity_check_tt_baseline/tt_baseline_centered_projections.json
Saves to:   ../sanity_check_tt_baseline/plots/
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── Resolve paths relative to this script ─────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _SCRIPT_DIR.parent   # .../outputs/llava-1.5-7b-hf/
_DATA_PATH = _MODEL_DIR / "sanity_check_tt_baseline" / "tt_baseline_projections.json"
_OUTPUT_DIR = _MODEL_DIR / "sanity_check_tt_baseline" / "plots"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
with open(_DATA_PATH) as f:
    data = json.load(f)

layers   = np.array([d["layer"] for d in data])
sss_proj = np.array([d["SSS_mean_tt_projection"] for d in data])
ssu_proj = np.array([d["SSU_mean_tt_projection"] for d in data])
sss_std  = np.array([d["SSS_std"] for d in data])
ssu_std  = np.array([d["SSU_std"] for d in data])
p_vals   = np.array([d["p_value"] for d in data])
t_stats  = np.array([d["t_statistic"] for d in data])

gap = sss_proj - ssu_proj  # positive means SSU is more "unsafe"

# ── Style configuration ───────────────────────────────────────────────────
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

COLOR_SSS = "#2563eb"   # blue
COLOR_SSU = "#dc2626"   # red
COLOR_GAP = "#059669"   # emerald green
COLOR_PVAL = "#7c3aed"  # purple
SAFETY_LAYER_COLOR = "#fef3c7"  # pale yellow highlight

bar_width = 0.38
x = np.arange(len(layers))
safety_lo = np.searchsorted(layers, 6) - 0.5
safety_hi = np.searchsorted(layers, 14) + 0.5
safety_center = (np.searchsorted(layers, 6) + np.searchsorted(layers, 14)) / 2
safety_mask = (layers >= 6) & (layers <= 14)

SUBTITLE = "LLaVA-1.5-7B  ·  TT Activations Projected onto Safety Direction $s^l$"


def _save(fig, name):
    path = _OUTPUT_DIR / name
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {path}")
    plt.close(fig)


# ── Plot 1: Mean TT projection for SSS vs SSU ────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axvspan(safety_lo, safety_hi, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)
ax.bar(x - bar_width/2, sss_proj, bar_width,
       color=COLOR_SSS, alpha=0.8, label="SSS (safe output)",
       edgecolor="white", linewidth=0.3)
ax.bar(x + bar_width/2, ssu_proj, bar_width,
       color=COLOR_SSU, alpha=0.8, label="SSU (unsafe output)",
       edgecolor="white", linewidth=0.3)

ax.axhline(y=0, color="#888", linestyle="-", linewidth=0.8, alpha=0.5)
ax.set_ylabel("Mean TT Projection\nonto Safety Direction")
ax.set_title(
    "Text-Only Baseline Position: SSS vs SSU Projection onto Safety Direction\n"
    + SUBTITLE
)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax.text(safety_center, ax.get_ylim()[1] * 0.95,
        "safety-critical layers (6\u201314)",
        ha="center", va="top", fontsize=8.5, color="#b45309", fontstyle="italic")

fig.tight_layout()
_save(fig, "raw_projections.png")

# ── Plot 2: Gap (SSS − SSU) ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.axvspan(safety_lo, safety_hi, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)

bar_colors = [COLOR_GAP if g > 0 else "#b91c1c" for g in gap]
ax.bar(x, gap, width=0.6, color=bar_colors, alpha=0.8, edgecolor="white",
       linewidth=0.3)
ax.axhline(y=0, color="#888", linestyle="-", linewidth=0.8, alpha=0.5)

ax.set_ylabel("Projection Gap\n(SSS $-$ SSU)")
ax.set_title(
    "Baseline Gap: How Much More \"Unsafe\" SSU Sits vs SSS "
    "(positive = SSU more unsafe)\n" + SUBTITLE
)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")

mean_gap_safety = gap[safety_mask].mean()
ax.text(safety_center, ax.get_ylim()[1] * 0.88,
        f"mean gap (L6\u201314): {mean_gap_safety:.4f}",
        ha="center", va="top", fontsize=9, fontweight="bold", color="#065f46",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#059669", alpha=0.9))

fig.tight_layout()
_save(fig, "projection_gap.png")

# ── Plot 5: P-values (log scale) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.axvspan(safety_lo, safety_hi, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)

p_plot = np.clip(p_vals, 1e-300, 1.0)
ax.plot(x, p_plot, "o-", color=COLOR_PVAL, markersize=5, linewidth=1.5,
        label="p-value (t-test)", alpha=0.85)

ax.axhline(y=0.05, color="#ef4444", linestyle=":", linewidth=1, alpha=0.6)
ax.text(len(x) - 0.5, 0.05, "p = 0.05", fontsize=8, color="#ef4444",
        va="bottom", ha="right")
ax.axhline(y=0.001, color="#f97316", linestyle=":", linewidth=1, alpha=0.4)
ax.text(len(x) - 0.5, 0.001, "p = 0.001", fontsize=8, color="#f97316",
        va="bottom", ha="right")

ax.set_yscale("log")
ax.set_ylabel("p-value (log scale)")
ax.set_title(
    "Statistical Significance of SSS vs SSU Baseline Difference\n" + SUBTITLE
)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

min_p = max(p_plot.min(), 1e-260)
ax.set_ylim(min_p / 10, 2)

ax.text(safety_center, min_p * 10,
        "safety-critical layers (6\u201314)",
        ha="center", va="bottom", fontsize=8.5, color="#b45309",
        fontstyle="italic")

fig.tight_layout()
_save(fig, "p_values.png")
