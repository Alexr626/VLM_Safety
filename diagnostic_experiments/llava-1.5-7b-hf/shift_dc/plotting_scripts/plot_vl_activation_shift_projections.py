"""
Plot ShiftDC Diagnostic Results: SSS vs SSU Activation Shifts
Generates individual plots for cosine similarity, projection magnitude,
and statistical significance across LLaVA-1.5-7B transformer layers.

Reads from: ../outputs/results/aggregate_stats.json
Saves to:   ../outputs/results/plots/
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── Resolve paths relative to this script ─────────────────────────────────
_SCRIPT_DIR     = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent                  # shift_dc/
_RESULTS_DIR    = _EXPERIMENT_DIR / "outputs" / "results"
_STATS_PATH     = _RESULTS_DIR / "vl_activation_shift" / "aggregate_stats.json"
_OUTPUT_DIR     = _RESULTS_DIR / "vl_activation_shift" / "plots"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not _STATS_PATH.exists():
    print(
        f"ERROR: {_STATS_PATH} not found.\n"
        "Please run the ShiftDC analysis first:\n"
        "  python .../shift_dc/experiment_scripts/vl_activation_shift.py"
    )
    sys.exit(1)

# ── Load data ──────────────────────────────────────────────────────────────
with open(_STATS_PATH) as f:
    data = json.load(f)

# Drop layers with no data (e.g. layer 0 embedding)
data = [d for d in data if d["n_sss"] > 0]

layers   = np.array([d["layer"] for d in data])
sss_cos  = np.array([d["SSS_mean_cosine"] for d in data])
ssu_cos  = np.array([d["SSU_mean_cosine"] for d in data])
sss_proj = np.array([d["SSS_mean_proj"] for d in data])
ssu_proj = np.array([d["SSU_mean_proj"] for d in data])
p_cos    = np.array([d["p_cosine"] for d in data])
p_proj   = np.array([d["p_proj"] for d in data])

n_sss = data[0]["n_sss"]
n_ssu = data[0]["n_ssu"]

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
COLOR_PCOS = "#7c3aed"  # purple
COLOR_PPROJ = "#0891b2" # teal
SAFETY_LAYER_COLOR = "#fef3c7"  # pale yellow highlight

bar_width = 0.38
x = np.arange(len(layers))

# Mark crossover — first layer >= 7 where SSU overtakes SSS
crossover_idx = None
for i in range(1, len(layers)):
    if layers[i] < 7:
        continue
    if (ssu_cos[i-1] - sss_cos[i-1]) > 0 and (ssu_cos[i] - sss_cos[i]) < 0:
        crossover_idx = i
        break


def _save(fig, name):
    path = _OUTPUT_DIR / name
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {path}")
    plt.close(fig)


# ── Plot 1: Cosine Similarity ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axvspan(5.5, 14.5, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)
ax.bar(x - bar_width/2, sss_cos, bar_width,
       color=COLOR_SSS, alpha=0.8, label="SSS (safe output)",
       edgecolor="white", linewidth=0.3)
ax.bar(x + bar_width/2, ssu_cos, bar_width,
       color=COLOR_SSU, alpha=0.8, label="SSU (unsafe output)",
       edgecolor="white", linewidth=0.3)

ax.set_ylabel("Mean Cosine Similarity\nwith Safety Direction")
ax.set_title(
    "Directional Alignment of Modality Shift with Safety Direction\n"
    f"LLaVA-1.5-7B  ·  SSS (n={n_sss}) vs SSU (n={n_ssu})"
)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

ax.text(10, ax.get_ylim()[1] * 0.95,
        "safety-critical layers (6\u201314)",
        ha="center", va="top", fontsize=8.5, color="#b45309", fontstyle="italic")

if crossover_idx is not None:
    ax.axvline(x=crossover_idx - 0.5, color="#888", linestyle="--",
               linewidth=1, alpha=0.6)
    ax.text(crossover_idx - 0.3, ax.get_ylim()[1] * 0.7,
            "crossover", fontsize=8, color="#666", va="center", rotation=90)

fig.tight_layout()
_save(fig, "cosine_similarity.png")

# ── Plot 2: Projection Magnitude ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axvspan(5.5, 14.5, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)
ax.bar(x - bar_width/2, sss_proj, bar_width,
       color=COLOR_SSS, alpha=0.8, label="SSS (safe output)",
       edgecolor="white", linewidth=0.3)
ax.bar(x + bar_width/2, ssu_proj, bar_width,
       color=COLOR_SSU, alpha=0.8, label="SSU (unsafe output)",
       edgecolor="white", linewidth=0.3)

ax.set_ylabel("Mean Projection Magnitude\nonto Safety Direction")
ax.set_title(
    "Absolute Displacement Along Safety Direction\n"
    f"LLaVA-1.5-7B  ·  SSS (n={n_sss}) vs SSU (n={n_ssu})"
)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

ax.text(10, ax.get_ylim()[1] * 0.95,
        "safety-critical layers (6\u201314)",
        ha="center", va="top", fontsize=8.5, color="#b45309", fontstyle="italic")

if crossover_idx is not None:
    ax.axvline(x=crossover_idx - 0.5, color="#888", linestyle="--",
               linewidth=1, alpha=0.6)

fig.tight_layout()
_save(fig, "projection_magnitude.png")

# ── Plot 3: P-values (log scale) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.axvspan(5.5, 14.5, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)

p_cos_plot  = np.clip(p_cos,  1e-70, 1.0)
p_proj_plot = np.clip(p_proj, 1e-70, 1.0)

ax.plot(x, p_cos_plot, "o-", color=COLOR_PCOS, markersize=4, linewidth=1.5,
        label="p-value (cosine)", alpha=0.85)
ax.plot(x, p_proj_plot, "s-", color=COLOR_PPROJ, markersize=4, linewidth=1.5,
        label="p-value (projection)", alpha=0.85)

ax.axhline(y=0.05, color="#ef4444", linestyle=":", linewidth=1, alpha=0.6)
ax.text(len(x) - 0.5, 0.05, "p = 0.05", fontsize=8, color="#ef4444",
        va="bottom", ha="right")
ax.axhline(y=0.001, color="#f97316", linestyle=":", linewidth=1, alpha=0.4)
ax.text(len(x) - 0.5, 0.001, "p = 0.001", fontsize=8, color="#f97316",
        va="bottom", ha="right")

ax.set_yscale("log")
ax.set_ylabel("p-value (log scale)")
ax.set_title(
    "Statistical Significance of SSS vs SSU Difference (t-test)\n"
    f"LLaVA-1.5-7B  ·  SSS (n={n_sss}) vs SSU (n={n_ssu})"
)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax.set_ylim(1e-65, 2)

ax.text(10, 1e-62,
        "safety-critical layers (6\u201314)",
        ha="center", va="bottom", fontsize=8.5, color="#b45309", fontstyle="italic")

fig.tight_layout()
_save(fig, "p_values.png")
