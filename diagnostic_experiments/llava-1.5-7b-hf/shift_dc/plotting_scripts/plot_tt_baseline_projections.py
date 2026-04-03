"""
Plot Sanity Check: Text-Only Baseline Positions Along Safety Direction
Generates plots for both projection and cosine similarity metrics,
showing whether SSU text-only activations already sit closer to the
"unsafe" side of the safety boundary than SSS text-only activations.

Reads from: ../outputs/results/tt_baseline_projections.json
Saves to:   ../outputs/results/plots/
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── Resolve paths relative to this script ─────────────────────────────────
_SCRIPT_DIR     = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent                  # shift_dc/
_RESULTS_DIR    = _EXPERIMENT_DIR / "outputs" / "results"
_DATA_PATH      = _RESULTS_DIR / "sanity_check_tt_baseline" / "tt_baseline_projections.json"
_OUTPUT_DIR     = _RESULTS_DIR / "sanity_check_tt_baseline" / "plots"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
with open(_DATA_PATH) as f:
    data = json.load(f)

layers = np.array([d["layer"] for d in data])

# Projection fields
sss_proj = np.array([d["SSS_mean_tt_projection"] for d in data])
ssu_proj = np.array([d["SSU_mean_tt_projection"] for d in data])
sss_std_proj = np.array([d["SSS_std_projection"] for d in data])
ssu_std_proj = np.array([d["SSU_std_projection"] for d in data])
p_vals_proj  = np.array([d["p_value_projection"] for d in data])

# Cosine similarity fields
has_cosine   = "SSS_mean_tt_cosine" in data[0]
sss_cos      = np.array([d["SSS_mean_tt_cosine"] for d in data]) if has_cosine else None
ssu_cos      = np.array([d["SSU_mean_tt_cosine"] for d in data]) if has_cosine else None
sss_std_cos  = np.array([d["SSS_std_cosine"]     for d in data]) if has_cosine else None
ssu_std_cos  = np.array([d["SSU_std_cosine"]     for d in data]) if has_cosine else None
p_vals_cos   = np.array([d["p_value_cosine"]     for d in data]) if has_cosine else None

gap_proj = sss_proj - ssu_proj
gap_cos  = (sss_cos - ssu_cos) if has_cosine else None

# ── Style ─────────────────────────────────────────────────────────────────
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
COLOR_GAP = "#059669"
COLOR_PVAL_PROJ = "#7c3aed"
COLOR_PVAL_COS  = "#0891b2"
SAFETY_BG = "#fef3c7"

bar_width = 0.38
x = np.arange(len(layers))
safety_lo     = np.searchsorted(layers, 6)  - 0.5
safety_hi     = np.searchsorted(layers, 14) + 0.5
safety_center = (np.searchsorted(layers, 6) + np.searchsorted(layers, 14)) / 2
safety_mask   = (layers >= 6) & (layers <= 14)

SUBTITLE_PROJ = "LLaVA-1.5-7B  ·  TT Activations Projected onto Safety Direction $s^l$"
SUBTITLE_COS  = "LLaVA-1.5-7B  ·  Cosine Similarity of TT Activations with Safety Direction $s^l$"


def _save(fig, name):
    path = _OUTPUT_DIR / name
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved → {path}")
    plt.close(fig)


def _annotate_safety(ax, y_frac=0.95):
    ax.axvspan(safety_lo, safety_hi, color=SAFETY_BG, alpha=0.5, zorder=0)
    ylim = ax.get_ylim()
    ax.text(safety_center, ylim[0] + (ylim[1] - ylim[0]) * y_frac,
            "safety-critical layers (6–14)",
            ha="center", va="top", fontsize=8.5, color="#b45309", fontstyle="italic")


def _plot_bars(sss_vals, ssu_vals, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - bar_width/2, sss_vals, bar_width,
           color=COLOR_SSS, alpha=0.8, label="SSS (safe output)",
           edgecolor="white", linewidth=0.3)
    ax.bar(x + bar_width/2, ssu_vals, bar_width,
           color=COLOR_SSU, alpha=0.8, label="SSU (unsafe output)",
           edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="#888", linewidth=0.8, alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x); ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    _annotate_safety(ax)
    fig.tight_layout()
    _save(fig, fname)


def _plot_gap(gap_vals, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(14, 4))
    bar_colors = [COLOR_GAP if g > 0 else "#b91c1c" for g in gap_vals]
    ax.bar(x, gap_vals, width=0.6, color=bar_colors, alpha=0.8,
           edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="#888", linewidth=0.8, alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x); ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    mean_gap = gap_vals[safety_mask].mean()
    ylim = ax.get_ylim()
    ax.text(safety_center, ylim[0] + (ylim[1] - ylim[0]) * 0.88,
            f"mean gap (L6–14): {mean_gap:.4f}",
            ha="center", va="top", fontsize=9, fontweight="bold", color="#065f46",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#059669", alpha=0.9))
    _annotate_safety(ax, y_frac=0.12)
    fig.tight_layout()
    _save(fig, fname)


def _plot_pvals(p_proj, p_cos, fname):
    fig, ax = plt.subplots(figsize=(14, 4))
    _annotate_safety(ax)
    # NaN p-values (e.g. layer 0 degenerate case) → treat as non-significant
    p_proj_clean = np.where(np.isnan(p_proj), 1.0, p_proj)
    p_proj_plot  = np.clip(p_proj_clean, 1e-300, 1.0)
    ax.plot(x, p_proj_plot, "o-", color=COLOR_PVAL_PROJ, markersize=5,
            linewidth=1.5, label="p-value (projection)", alpha=0.85)
    p_cos_plot = None
    if p_cos is not None:
        p_cos_clean = np.where(np.isnan(p_cos), 1.0, p_cos)
        p_cos_plot  = np.clip(p_cos_clean, 1e-300, 1.0)
        ax.plot(x, p_cos_plot, "s--", color=COLOR_PVAL_COS, markersize=5,
                linewidth=1.5, label="p-value (cosine)", alpha=0.85)
    ax.axhline(0.05,  color="#ef4444", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(len(x) - 0.5, 0.05,  "p = 0.05",  fontsize=8, color="#ef4444", va="bottom", ha="right")
    ax.axhline(0.001, color="#f97316", linestyle=":", linewidth=1, alpha=0.4)
    ax.text(len(x) - 0.5, 0.001, "p = 0.001", fontsize=8, color="#f97316", va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("p-value (log scale)")
    ax.set_title("Statistical Significance: SSS vs SSU Baseline Difference\n"
                 "LLaVA-1.5-7B  ·  Projection & Cosine (t-test, Welch)")
    ax.set_xticks(x); ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    all_p = np.concatenate([p_proj_plot, p_cos_plot if p_cos_plot is not None else p_proj_plot])
    min_p = max(float(np.nanmin(all_p)), 1e-260)
    ax.set_ylim(min_p / 10, 2)
    fig.tight_layout()
    _save(fig, fname)


# ── Plot 1: Projection bars ───────────────────────────────────────────────
_plot_bars(sss_proj, ssu_proj,
           ylabel="Mean TT Projection\nonto Safety Direction",
           title="Text-Only Baseline: SSS vs SSU Projection onto Safety Direction\n" + SUBTITLE_PROJ,
           fname="raw_projections.png")

# ── Plot 2: Projection gap ────────────────────────────────────────────────
_plot_gap(gap_proj,
          ylabel="Projection Gap (SSS − SSU)",
          title="Baseline Gap: How Much More \"Unsafe\" SSU Sits (positive = SSU more unsafe)\n" + SUBTITLE_PROJ,
          fname="projection_gap.png")

# ── Plot 3: Cosine bars ───────────────────────────────────────────────────
if has_cosine:
    _plot_bars(sss_cos, ssu_cos,
               ylabel="Mean Cosine Similarity\nwith Safety Direction",
               title="Text-Only Baseline: SSS vs SSU Cosine Similarity with Safety Direction\n" + SUBTITLE_COS,
               fname="raw_cosine.png")

# ── Plot 4: Cosine gap ────────────────────────────────────────────────────
if has_cosine:
    _plot_gap(gap_cos,
              ylabel="Cosine Gap (SSS − SSU)",
              title="Baseline Gap (Cosine): How Much More \"Unsafe\" SSU Sits\n" + SUBTITLE_COS,
              fname="cosine_gap.png")

# ── Plot 5: P-values (both metrics) ──────────────────────────────────────
_plot_pvals(p_vals_proj, p_vals_cos if has_cosine else None,
            fname="p_values.png")
