"""
Plot Experiment C: Subspace Overlap Between Integration and Safety Subspaces
Generates individual plots for integration-safety overlap, SSS-SSU overlap,
overlap gap, and sensitivity to subspace dimensionality.

Reads from: ../outputs/results/subspace_overlap_results.json
            ../outputs/results/sensitivity_analysis.json
Saves to:   ../outputs/results/plots/
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── Resolve paths relative to this script ─────────────────────────────────
_SCRIPT_DIR     = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent                  # subspace_overlap/
_RESULTS_DIR    = _EXPERIMENT_DIR / "outputs" / "results"

_OVERLAP_PATH     = _RESULTS_DIR / "subspace_overlap_results.json"
_SENSITIVITY_PATH = _RESULTS_DIR / "sensitivity_analysis.json"
_OUTPUT_DIR       = _RESULTS_DIR / "plots"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for path, label in [(_OVERLAP_PATH, "subspace overlap results"),
                    (_SENSITIVITY_PATH, "sensitivity analysis")]:
    if not path.exists():
        print(
            f"ERROR: {path} not found ({label}).\n"
            "Please run Experiment C first:\n"
            "  python .../subspace_overlap/experiment_scripts/"
            "experiment_c_subspace_overlap.py"
        )
        sys.exit(1)

# ── Load data ──────────────────────────────────────────────────────────────
with open(_OVERLAP_PATH) as f:
    overlap_data = json.load(f)

with open(_SENSITIVITY_PATH) as f:
    sensitivity_data = json.load(f)

overlap_data = [d for d in overlap_data if d["layer"] > 0]

layers = np.array([d["layer"] for d in overlap_data])
overlap_sss_vs_safety = np.array([d["overlap_sss_vs_safety"] for d in overlap_data])
overlap_ssu_vs_safety = np.array([d["overlap_ssu_vs_safety"] for d in overlap_data])
overlap_all_vs_safety = np.array([d["overlap_all_vs_safety"] for d in overlap_data])
overlap_sss_vs_ssu = np.array([d["overlap_sss_vs_ssu"] for d in overlap_data])

overlap_gap = overlap_sss_vs_safety - overlap_ssu_vs_safety

# Sensitivity analysis
sens_layers_all = sorted(set(d["layer"] for d in sensitivity_data))
sens_by_layer = {}
for layer in sens_layers_all:
    layer_entries = sorted(
        [d for d in sensitivity_data if d["layer"] == layer],
        key=lambda d: d["k"],
    )
    sens_by_layer[layer] = {
        "k": np.array([d["k"] for d in layer_entries]),
        "sss": np.array([d["overlap_sss_vs_safety"] for d in layer_entries]),
        "ssu": np.array([d["overlap_ssu_vs_safety"] for d in layer_entries]),
    }

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

COLOR_SSS = "#2563eb"
COLOR_SSU = "#dc2626"
COLOR_ALL = "#6b7280"
COLOR_GAP_POS = "#059669"
COLOR_GAP_NEG = "#b91c1c"
SAFETY_LAYER_COLOR = "#fef3c7"

SENS_LAYER_COLORS = {
    7: "#dc2626",
    10: "#7c3aed",
    14: "#f97316",
    20: "#2563eb",
}

safety_lo = np.searchsorted(layers, 6) - 0.5
safety_hi = np.searchsorted(layers, 14) + 0.5
safety_center = (np.searchsorted(layers, 6) + np.searchsorted(layers, 14)) / 2
safety_mask = (layers >= 6) & (layers <= 14)

x = np.arange(len(layers))

SUBTITLE = (
    "LLaVA-1.5-7B  ·  Overlap = mean cosine of principal angles between "
    "top-k subspaces"
)


def _save(fig, name):
    path = _OUTPUT_DIR / name
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {path}")
    plt.close(fig)


# ── Plot 1: Integration-Safety Subspace Overlap ──────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axvspan(safety_lo, safety_hi, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)

ax.plot(x, overlap_sss_vs_safety, "o-", color=COLOR_SSS, markersize=4,
        linewidth=1.5, alpha=0.85, label="SSS vs Safety")
ax.plot(x, overlap_ssu_vs_safety, "s-", color=COLOR_SSU, markersize=4,
        linewidth=1.5, alpha=0.85, label="SSU vs Safety")
ax.plot(x, overlap_all_vs_safety, "^-", color=COLOR_ALL, markersize=4,
        linewidth=1.5, alpha=0.55, label="All vs Safety")

ax.set_ylabel("Subspace Overlap\n(mean cos of principal angles)")
ax.set_title("Integration\u2013Safety Subspace Overlap\n" + SUBTITLE)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

ax.text(safety_center, ax.get_ylim()[1] * 0.95,
        "safety-critical layers (6\u201314)",
        ha="center", va="top", fontsize=8.5, color="#b45309", fontstyle="italic")

mean_sss_safety = overlap_sss_vs_safety[safety_mask].mean()
mean_ssu_safety = overlap_ssu_vs_safety[safety_mask].mean()
ax.text(safety_center, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
        f"L6\u201314 mean: SSS={mean_sss_safety:.3f}, SSU={mean_ssu_safety:.3f}",
        ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#065f46",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#059669", alpha=0.9))

fig.tight_layout()
_save(fig, "integration_safety_overlap.png")

# ── Plot 2: SSS vs SSU Integration Subspace Overlap ──────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axvspan(safety_lo, safety_hi, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)

ax.plot(x, overlap_sss_vs_ssu, "D-", color="#7c3aed", markersize=4,
        linewidth=1.5, alpha=0.85, label="SSS vs SSU Integration")

ax.set_ylabel("SSS\u2013SSU Integration Overlap")
ax.set_title("SSS vs SSU Integration Subspace Overlap\n" + SUBTITLE)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

ax.text(safety_center, ax.get_ylim()[1] * 0.95,
        "safety-critical layers (6\u201314)",
        ha="center", va="top", fontsize=8.5, color="#b45309", fontstyle="italic")

mean_sss_ssu_safety = overlap_sss_vs_ssu[safety_mask].mean()
ax.text(safety_center, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
        f"L6\u201314 mean: {mean_sss_ssu_safety:.3f}",
        ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#065f46",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#059669", alpha=0.9))

fig.tight_layout()
_save(fig, "sss_ssu_integration_overlap.png")

# ── Plot 3: Overlap Gap (SSS - SSU) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.axvspan(safety_lo, safety_hi, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)

bar_colors = [COLOR_GAP_POS if g >= 0 else COLOR_GAP_NEG for g in overlap_gap]
ax.bar(x, overlap_gap, width=0.6, color=bar_colors, alpha=0.8,
       edgecolor="white", linewidth=0.3)
ax.axhline(y=0, color="#888", linestyle="-", linewidth=0.8, alpha=0.5)

ax.set_ylabel(r"Overlap Gap (SSS $-$ SSU)")
ax.set_title(
    r"Overlap Gap: SSS $-$ SSU (Integration vs Safety)" + "\n" + SUBTITLE
)
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_xlabel("Transformer Layer")

mean_gap_safety = overlap_gap[safety_mask].mean()
ax.text(safety_center, ax.get_ylim()[1] * 0.88,
        f"mean gap (L6\u201314): {mean_gap_safety:+.4f}",
        ha="center", va="top", fontsize=9, fontweight="bold", color="#065f46",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#059669", alpha=0.9))

ax.text(safety_center, ax.get_ylim()[1] * 0.95,
        "safety-critical layers (6\u201314)",
        ha="center", va="top", fontsize=8.5, color="#b45309", fontstyle="italic")

fig.tight_layout()
_save(fig, "overlap_gap.png")

# ── Plot 4: Sensitivity to Subspace Dimensionality k ─────────────────────
selected_sens_layers = [7, 10, 14, 20]

fig, ax = plt.subplots(figsize=(8, 5))

for layer in selected_sens_layers:
    if layer not in sens_by_layer:
        continue
    color = SENS_LAYER_COLORS[layer]
    k_vals = sens_by_layer[layer]["k"]
    sss_vals = sens_by_layer[layer]["sss"]
    ssu_vals = sens_by_layer[layer]["ssu"]

    ax.plot(k_vals, sss_vals, "o-", color=color, markersize=4, linewidth=1.5,
            alpha=0.85, label=f"L{layer} SSS")
    ax.plot(k_vals, ssu_vals, "s--", color=color, markersize=4, linewidth=1.5,
            alpha=0.55, label=f"L{layer} SSU")

ax.plot([], [], "o-", color="black", markersize=3.5, linewidth=1.3,
        label="solid = SSS vs Safety")
ax.plot([], [], "s--", color="black", markersize=3.5, linewidth=1.3,
        alpha=0.55, label="dashed = SSU vs Safety")

ax.set_xlabel("Subspace Dimensionality $k$")
ax.set_ylabel("Overlap with Safety Subspace")
ax.set_title("Sensitivity to Subspace Dimensionality $k$")
ax.legend(loc="lower right", framealpha=0.9, fontsize=8, ncol=2)

fig.tight_layout()
_save(fig, "sensitivity_dimensionality.png")
