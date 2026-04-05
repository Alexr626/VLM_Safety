"""
Plot Experiment D: Per-Category Analysis of Cross-Modal Safety Distortion
Generates individual plots examining whether the cross-modal distortion differs
across harm categories or is category-agnostic.

Reads from: ../outputs/results/per_category_projections.json
            ../outputs/results/category_subspace_overlap.json
            ../outputs/results/anova_across_categories.json
            ../outputs/results/category_sample_counts.json
Saves to:   ../outputs/results/plots/
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── Resolve paths relative to this script ─────────────────────────────────
_SCRIPT_DIR     = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent                  # category_analysis/
_RESULTS_DIR    = _EXPERIMENT_DIR / "outputs" / "results"

_PROJ_PATH    = _RESULTS_DIR / "per_category_projections.json"
_OVERLAP_PATH = _RESULTS_DIR / "category_subspace_overlap.json"
_ANOVA_PATH   = _RESULTS_DIR / "anova_across_categories.json"
_COUNTS_PATH  = _RESULTS_DIR / "category_sample_counts.json"
_OUTPUT_DIR   = _RESULTS_DIR / "plots"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for path, label in [
    (_PROJ_PATH, "per-category projections"),
    (_OVERLAP_PATH, "category subspace overlap"),
    (_ANOVA_PATH, "ANOVA results"),
    (_COUNTS_PATH, "category sample counts"),
]:
    if not path.exists():
        print(
            f"ERROR: {path} not found ({label}).\n"
            "Please run Experiment D first:\n"
            "  python .../category_analysis/experiment_scripts/"
            "experiment_d_category_analysis.py"
        )
        sys.exit(1)

# ── Load data ──────────────────────────────────────────────────────────────
with open(_PROJ_PATH) as f:
    proj_data = json.load(f)
with open(_OVERLAP_PATH) as f:
    overlap_data = json.load(f)
with open(_ANOVA_PATH) as f:
    anova_data = json.load(f)
with open(_COUNTS_PATH) as f:
    sample_counts = json.load(f)

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

SAFETY_LAYER_COLOR = "#fef3c7"

CATEGORY_COLORS = {
    "hate": "#dc2626",
    "violence": "#f97316",
    "self_harm": "#8b5cf6",
    "illegal_activity": "#059669",
    "specialized_advice": "#0891b2",
}

CATEGORY_LABELS = {
    "hate": f"hate (n={sample_counts['hate']})",
    "violence": f"violence (n={sample_counts['violence']})",
    "self_harm": f"self_harm (n={sample_counts['self_harm']})",
    "illegal_activity": f"illegal_activity (n={sample_counts['illegal_activity']})",
    "specialized_advice": f"specialized_advice (n={sample_counts['specialized_advice']})",
}

CATEGORIES = ["hate", "violence", "self_harm", "illegal_activity", "specialized_advice"]

SUBTITLE = "LLaVA-1.5-7B  ·  Do different harm categories distort differently?"


def _save(fig, name):
    path = _OUTPUT_DIR / name
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {path}")
    plt.close(fig)


# ── Prepare data ──────────────────────────────────────────────────────────
# Panel (a): per-category projection for k=0
proj_k0 = [d for d in proj_data if d["component_k"] == 0 and d["layer"] > 0]
proj_layers = sorted(set(d["layer"] for d in proj_k0))

proj_by_cat = {}
for cat in CATEGORIES:
    cat_entries = sorted(
        [d for d in proj_k0 if d["category"] == cat],
        key=lambda d: d["layer"],
    )
    proj_by_cat[cat] = {
        "layers": np.array([d["layer"] for d in cat_entries]),
        "mean": np.array([d["mean_projection"] for d in cat_entries]),
        "std": np.array([d["std_projection"] for d in cat_entries]),
    }

# Panel (b): overlap with safety subspace
overlap_layers_all = sorted(set(d["layer"] for d in overlap_data))
bar_layers = [l for l in [7, 10, 14, 20, 25, 28] if l in overlap_layers_all]

overlap_by_cat = {}
for cat in CATEGORIES:
    cat_entries = {d["layer"]: d for d in overlap_data if d["category"] == cat}
    overlap_by_cat[cat] = cat_entries

# Panel (c): ANOVA heatmap
anova_layers = sorted(set(d["layer"] for d in anova_data if d["layer"] > 0))
anova_components = list(range(5))

neg_log_p = np.full((len(anova_components), len(anova_layers)), np.nan)
for d in anova_data:
    if d["layer"] == 0:
        continue
    layer_idx = anova_layers.index(d["layer"])
    comp_idx = d["component_k"]
    if comp_idx < 5:
        p = d["p_value"]
        if p is not None and not (isinstance(p, float) and np.isnan(p)):
            neg_log_p[comp_idx, layer_idx] = -np.log10(max(p, 1e-300))

# Panel (d): effective rank
rank_by_cat = {}
for cat in CATEGORIES:
    cat_entries = sorted(
        [d for d in overlap_data if d["category"] == cat],
        key=lambda d: d["layer"],
    )
    rank_by_cat[cat] = {
        "layers": np.array([d["layer"] for d in cat_entries]),
        "rank": np.array([d["effective_rank_tau90"] for d in cat_entries]),
    }

# Safety-critical layer helpers for panel (a)
layers_a = np.array(proj_layers)
x_a = np.arange(len(layers_a))
safety_lo_a = np.searchsorted(layers_a, 6) - 0.5
safety_hi_a = np.searchsorted(layers_a, 14) + 0.5
safety_center_a = (np.searchsorted(layers_a, 6) + np.searchsorted(layers_a, 14)) / 2

# ── Plot 1: Per-category projection onto dominant safety component ───────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axvspan(safety_lo_a, safety_hi_a, color=SAFETY_LAYER_COLOR, alpha=0.5, zorder=0)

for cat in CATEGORIES:
    cat_data = proj_by_cat[cat]
    x_pos = np.array([np.where(layers_a == l)[0][0] for l in cat_data["layers"]])
    ax.plot(
        x_pos, cat_data["mean"],
        "o-", color=CATEGORY_COLORS[cat], markersize=3.5, linewidth=1.3,
        alpha=0.85, label=CATEGORY_LABELS[cat],
    )

ax.axhline(y=0, color="#888", linestyle="-", linewidth=0.8, alpha=0.5)
ax.set_ylabel("Mean Projection onto\nSafety Component k=0")
ax.set_title(
    "Per-Category Projection onto Dominant Safety Component (k=0)\n" + SUBTITLE
)
ax.set_xticks(x_a)
ax.set_xticklabels([str(l) for l in layers_a], fontsize=8)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="best", framealpha=0.9, fontsize=8)

ax.text(safety_center_a, ax.get_ylim()[1] * 0.95,
        "safety-critical layers (6\u201314)",
        ha="center", va="top", fontsize=8.5, color="#b45309", fontstyle="italic")

fig.tight_layout()
_save(fig, "per_category_projections.png")

# ── Plot 2: Grouped bar chart of overlap with safety subspace ────────────
n_cats = len(CATEGORIES)
bar_width = 0.15
x_b = np.arange(len(bar_layers))

fig, ax = plt.subplots(figsize=(12, 5))

for i, cat in enumerate(CATEGORIES):
    overlaps = []
    for layer in bar_layers:
        if layer in overlap_by_cat[cat]:
            overlaps.append(overlap_by_cat[cat][layer]["overlap_with_safety"])
        else:
            overlaps.append(0.0)
    offset = (i - n_cats / 2 + 0.5) * bar_width
    ax.bar(
        x_b + offset, overlaps, bar_width,
        color=CATEGORY_COLORS[cat], alpha=0.8,
        label=CATEGORY_LABELS[cat],
        edgecolor="white", linewidth=0.3,
    )

ax.set_ylabel("Overlap with Safety Subspace")
ax.set_title("Category-Specific Overlap with Safety Subspace\n" + SUBTITLE)
ax.set_xticks(x_b)
ax.set_xticklabels([f"L{l}" for l in bar_layers], fontsize=9)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="best", framealpha=0.9, fontsize=7.5)

for idx, layer in enumerate(bar_layers):
    if 6 <= layer <= 14:
        ax.axvspan(idx - 0.5, idx + 0.5, color=SAFETY_LAYER_COLOR,
                   alpha=0.3, zorder=0)

fig.tight_layout()
_save(fig, "category_overlap_safety.png")

# ── Plot 3: ANOVA heatmap (-log10 p-value) ───────────────────────────────
sig_threshold = -np.log10(0.05)
vmax_display = min(np.nanmax(neg_log_p), 100)

fig, ax = plt.subplots(figsize=(14, 4))
ax.grid(False)

im = ax.imshow(
    np.clip(neg_log_p, 0, vmax_display),
    aspect="auto", cmap=plt.cm.YlOrRd, interpolation="nearest",
    vmin=0, vmax=vmax_display,
)

for comp_idx in range(len(anova_components)):
    for layer_idx in range(len(anova_layers)):
        val = neg_log_p[comp_idx, layer_idx]
        if not np.isnan(val) and val < sig_threshold:
            ax.text(layer_idx, comp_idx, "ns", ha="center", va="center",
                    fontsize=6, color="black", alpha=0.6)

ax.contour(
    neg_log_p, levels=[sig_threshold], colors=["black"],
    linewidths=[1.5], linestyles=["--"],
)

ax.set_yticks(range(len(anova_components)))
ax.set_yticklabels([f"k={k}" for k in anova_components], fontsize=9)
ax.set_xticks(range(len(anova_layers)))
ax.set_xticklabels([str(l) for l in anova_layers], fontsize=7)
ax.set_xlabel("Transformer Layer")
ax.set_ylabel("Safety Component")
ax.set_title(
    r"ANOVA: Do Categories Differ on Safety Components? ($-\log_{10}$ p-value)"
    + "\n" + SUBTITLE
)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"$-\log_{10}$(p-value)", fontsize=9)
cbar.ax.axhline(y=sig_threshold, color="black", linestyle="--", linewidth=1)
cbar.ax.text(1.5, sig_threshold, "p=0.05", fontsize=7, va="center")

for layer_idx, layer in enumerate(anova_layers):
    if 6 <= layer <= 14:
        ax.axvspan(layer_idx - 0.5, layer_idx + 0.5,
                   color=SAFETY_LAYER_COLOR, alpha=0.15, zorder=0)

fig.tight_layout()
_save(fig, "anova_heatmap.png")

# ── Plot 4: Per-category effective rank at tau=0.9 ───────────────────────
rank_layers = sorted(set(d["layer"] for d in overlap_data))
x_d = np.arange(len(rank_layers))
rank_layers_arr = np.array(rank_layers)

fig, ax = plt.subplots(figsize=(12, 5))

for cat in CATEGORIES:
    cat_data = rank_by_cat[cat]
    x_pos = np.array([np.where(rank_layers_arr == l)[0][0] for l in cat_data["layers"]])
    ax.plot(
        x_pos, cat_data["rank"],
        "o-", color=CATEGORY_COLORS[cat], markersize=5, linewidth=1.5,
        alpha=0.85, label=CATEGORY_LABELS[cat],
    )

ax.set_ylabel(r"Effective Rank ($\tau$ = 0.9)")
ax.set_title(
    r"Per-Category Effective Rank at $\tau$ = 0.9" + "\n" + SUBTITLE
)
ax.set_xticks(x_d)
ax.set_xticklabels([f"L{l}" for l in rank_layers], fontsize=9)
ax.set_xlabel("Transformer Layer")
ax.legend(loc="best", framealpha=0.9, fontsize=8)

for idx, layer in enumerate(rank_layers):
    if 6 <= layer <= 14:
        ax.axvspan(idx - 0.5, idx + 0.5, color=SAFETY_LAYER_COLOR,
                   alpha=0.3, zorder=0)

fig.tight_layout()
_save(fig, "per_category_effective_rank.png")
