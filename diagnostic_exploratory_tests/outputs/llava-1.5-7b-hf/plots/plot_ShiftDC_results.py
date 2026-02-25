"""
Plot ShiftDC Diagnostic Results: SSS vs SSU Activation Shifts
Generates three-panel figure showing cosine similarity, projection magnitude,
and statistical significance across LLaVA-1.5-7B transformer layers.

Reads from: ../method2_activation_shift/aggregate_stats.json
Saves to:   ./shiftdc_diagnostic_results.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── Resolve paths relative to this script ─────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _SCRIPT_DIR.parent  # .../outputs/llava-1.5-7b-hf/
_STATS_PATH = _MODEL_DIR / "method2_activation_shift" / "aggregate_stats.json"
_OUTPUT_PATH = _SCRIPT_DIR / "shiftdc_diagnostic_results.png"

if not _STATS_PATH.exists():
    print(
        f"ERROR: {_STATS_PATH} not found.\n"
        "Please run Method 2 (ShiftDC activation shift analysis) first:\n"
        "  python diagnostic_exploratory_tests/shiftdc/method2_activation_shift.py "
        "--model llava-hf/llava-1.5-7b-hf"
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
sss_proj = np.array([d["SSS_mean_projection"] for d in data])
ssu_proj = np.array([d["SSU_mean_projection"] for d in data])
p_cos    = np.array([d["t_test_p_value_cosine"] for d in data])
p_proj   = np.array([d["t_test_p_value_projection"] for d in data])

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

# ── Create figure ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                         gridspec_kw={"height_ratios": [1, 1, 0.7]})
fig.suptitle(
    "Modality-Induced Activation Shift Along Safety Direction\n"
    f"LLaVA-1.5-7B  ·  HoliSafe-Bench SSS (n={n_sss}) vs SSU (n={n_ssu})",
    fontsize=14, fontweight="bold", y=0.97,
)

x = np.arange(len(layers))

# ── Panel 1: Cosine Similarity ────────────────────────────────────────────
ax1 = axes[0]
ax1.axvspan(5.5, 14.5, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)
ax1.bar(x - bar_width/2, sss_cos, bar_width,
        color=COLOR_SSS, alpha=0.8, label="SSS (safe output)",
        edgecolor="white", linewidth=0.3)
ax1.bar(x + bar_width/2, ssu_cos, bar_width,
        color=COLOR_SSU, alpha=0.8, label="SSU (unsafe output)",
        edgecolor="white", linewidth=0.3)

ax1.set_ylabel("Mean Cosine Similarity\nwith Safety Direction")
ax1.set_title("(a) Directional Alignment of Modality Shift with Safety Direction")
ax1.set_xticks(x)
ax1.set_xticklabels([str(l) for l in layers], fontsize=8)
ax1.set_xlabel("")
ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

ax1.text(10, ax1.get_ylim()[1] * 0.95,
         "safety-critical layers (6\u201314)",
         ha="center", va="top", fontsize=8.5, color="#b45309", fontstyle="italic")

# Mark crossover — first layer >= 7 where SSU overtakes SSS
crossover_idx = None
for i in range(1, len(layers)):
    if layers[i] < 7:
        continue
    if (ssu_cos[i-1] - sss_cos[i-1]) > 0 and (ssu_cos[i] - sss_cos[i]) < 0:
        crossover_idx = i
        break

if crossover_idx is not None:
    ax1.axvline(x=crossover_idx - 0.5, color="#888", linestyle="--",
                linewidth=1, alpha=0.6)
    ax1.text(crossover_idx - 0.3, ax1.get_ylim()[1] * 0.7,
             "crossover", fontsize=8, color="#666", va="center", rotation=90)

# ── Panel 2: Projection Magnitude ────────────────────────────────────────
ax2 = axes[1]
ax2.axvspan(5.5, 14.5, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)
ax2.bar(x - bar_width/2, sss_proj, bar_width,
        color=COLOR_SSS, alpha=0.8, label="SSS (safe output)",
        edgecolor="white", linewidth=0.3)
ax2.bar(x + bar_width/2, ssu_proj, bar_width,
        color=COLOR_SSU, alpha=0.8, label="SSU (unsafe output)",
        edgecolor="white", linewidth=0.3)

ax2.set_ylabel("Mean Projection Magnitude\nonto Safety Direction")
ax2.set_title("(b) Absolute Displacement Along Safety Direction")
ax2.set_xticks(x)
ax2.set_xticklabels([str(l) for l in layers], fontsize=8)
ax2.set_xlabel("")
ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

ax2.text(10, ax2.get_ylim()[1] * 0.95,
         "safety-critical layers (6\u201314)",
         ha="center", va="top", fontsize=8.5, color="#b45309", fontstyle="italic")

if crossover_idx is not None:
    ax2.axvline(x=crossover_idx - 0.5, color="#888", linestyle="--",
                linewidth=1, alpha=0.6)

# ── Panel 3: P-values (log scale) ────────────────────────────────────────
ax3 = axes[2]
ax3.axvspan(5.5, 14.5, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)

p_cos_plot  = np.clip(p_cos,  1e-70, 1.0)
p_proj_plot = np.clip(p_proj, 1e-70, 1.0)

ax3.plot(x, p_cos_plot, "o-", color=COLOR_PCOS, markersize=4, linewidth=1.5,
         label="p-value (cosine)", alpha=0.85)
ax3.plot(x, p_proj_plot, "s-", color=COLOR_PPROJ, markersize=4, linewidth=1.5,
         label="p-value (projection)", alpha=0.85)

ax3.axhline(y=0.05, color="#ef4444", linestyle=":", linewidth=1, alpha=0.6)
ax3.text(len(x) - 0.5, 0.05, "p = 0.05", fontsize=8, color="#ef4444",
         va="bottom", ha="right")
ax3.axhline(y=0.001, color="#f97316", linestyle=":", linewidth=1, alpha=0.4)
ax3.text(len(x) - 0.5, 0.001, "p = 0.001", fontsize=8, color="#f97316",
         va="bottom", ha="right")

ax3.set_yscale("log")
ax3.set_ylabel("p-value (log scale)")
ax3.set_title("(c) Statistical Significance of SSS vs SSU Difference (t-test)")
ax3.set_xticks(x)
ax3.set_xticklabels([str(l) for l in layers], fontsize=8)
ax3.set_xlabel("Transformer Layer")
ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax3.set_ylim(1e-65, 2)

ax3.text(10, 1e-62,
         "safety-critical layers (6\u201314)",
         ha="center", va="bottom", fontsize=8.5, color="#b45309", fontstyle="italic")

# ── Annotations ───────────────────────────────────────────────────────────
regime_text = (
    "Layers 4\u201314: SSU shifts MORE toward safe "
    "\u2192 safety perception distorted more for unsafe-output examples\n"
    "Layers 16\u201331: SSS shifts MORE toward safe "
    "\u2192 later layers reflect correct safety disposition"
)
fig.text(0.5, 0.015, regime_text, ha="center", va="bottom", fontsize=9,
         fontstyle="italic", color="#555",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                   edgecolor="#ddd", alpha=0.9))

plt.tight_layout(rect=[0, 0.05, 1, 0.94])

# ── Save ──────────────────────────────────────────────────────────────────
fig.savefig(str(_OUTPUT_PATH), dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {_OUTPUT_PATH}")
plt.close()
