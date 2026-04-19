#!/usr/bin/env python3
"""
Plot Intervention Results: ShiftDC vs Spherical ShiftDC
=========================================================
Reads summary JSON files from intervention/llava-1.5-7b-hf/outputs/results/
and produces comparison figures:

  1. ASR bar chart       — SSU Attack Success Rate across methods
  2. Helpfulness chart   — SSS false-refusal rate (helpfulness cost)
  3. Norm preservation   — mean ||x_hat|| / ||x_vl|| ratio per method
  4. Per-category ASR    — SSU ASR breakdown by HoliSafe harm category

Usage
-----
  cd intervention/llava-1.5-7b-hf/plotting_scripts/
  python plot_intervention_results.py

  # Custom results directory
  python plot_intervention_results.py --results_dir /path/to/results
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_SCRIPT_DIR   = Path(__file__).resolve().parent
_MODEL_DIR    = _SCRIPT_DIR.parent
_PROJECT_ROOT = _MODEL_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Style constants ───────────────────────────────────────────────────────────

# Okabe-Ito colorblind-safe palette
_COLORS = {
    "none":       "#E69F00",   # orange — baseline
    "original":   "#56B4E9",   # sky blue — original ShiftDC
    "spherical":  "#009E73",   # green — Spherical ShiftDC
    "other":      "#CC79A7",   # pink — additional variants
}

plt.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       120,
})

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_summaries(results_dir: Path) -> dict:
    """Load all *_summary.json files. Returns {method_tag: summary_dict}."""
    summaries = {}
    for path in sorted(results_dir.glob("*_summary.json")):
        with open(path) as f:
            s = json.load(f)
        tag = s.get("method", path.stem.replace("_summary", ""))
        summaries[tag] = s
    return summaries


def _method_color(tag: str) -> str:
    if tag == "none":
        return _COLORS["none"]
    if tag.startswith("original"):
        return _COLORS["original"]
    if tag.startswith("spherical"):
        return _COLORS["spherical"]
    return _COLORS["other"]


def _method_label(tag: str) -> str:
    if tag == "none":
        return "Baseline\n(no interv.)"
    if tag == "original":
        return "ShiftDC\n(original)"
    if tag.startswith("spherical_t"):
        t_val = tag.replace("spherical_t", "").replace("_gated", "")
        suffix = " (gated)" if "gated" in tag else ""
        return f"Spherical\nShiftDC (t={t_val}){suffix}"
    return tag


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_asr_bar(summaries: dict, out_path: Path):
    """Bar chart: SSU Attack Success Rate per method. Lower = better."""
    tags   = list(summaries.keys())
    values = [summaries[t].get("SSU", {}).get("asr", None) for t in tags]
    valid  = [(t, v) for t, v in zip(tags, values) if v is not None]
    if not valid:
        print("  Skipping ASR plot — no SSU data found")
        return
    tags_v, vals_v = zip(*valid)

    fig, ax = plt.subplots(figsize=(max(5, len(tags_v) * 1.4), 4))
    x = np.arange(len(tags_v))
    bars = ax.bar(x, vals_v, color=[_method_color(t) for t in tags_v],
                  width=0.6, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.1f%%", padding=3,
                 labels=[f"{v:.1%}" for v in vals_v])
    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(t) for t in tags_v], ha="center")
    ax.set_ylabel("Attack Success Rate (SSU) ↓")
    ax.set_ylim(0, 1.05)
    ax.set_title("SSU Attack Success Rate by Intervention Method")
    ax.axhline(vals_v[0], color="gray", linestyle="--", linewidth=0.8,
               label="Baseline")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_helpfulness_bar(summaries: dict, out_path: Path):
    """Bar chart: SSS false-refusal rate. Lower = better helpfulness."""
    tags   = list(summaries.keys())
    values = [summaries[t].get("SSS", {}).get("refusal_rate", None) for t in tags]
    valid  = [(t, v) for t, v in zip(tags, values) if v is not None]
    if not valid:
        print("  Skipping helpfulness plot — no SSS data found")
        return
    tags_v, vals_v = zip(*valid)

    fig, ax = plt.subplots(figsize=(max(5, len(tags_v) * 1.4), 4))
    x = np.arange(len(tags_v))
    bars = ax.bar(x, vals_v, color=[_method_color(t) for t in tags_v],
                  width=0.6, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.1f%%", padding=3,
                 labels=[f"{v:.1%}" for v in vals_v])
    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(t) for t in tags_v], ha="center")
    ax.set_ylabel("False Refusal Rate (SSS) ↓")
    ax.set_ylim(0, max(vals_v) * 1.25 + 0.02)
    ax.set_title("Helpfulness Cost: SSS False-Refusal Rate")
    ax.axhline(vals_v[0], color="gray", linestyle="--", linewidth=0.8,
               label="Baseline")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_norm_preservation(summaries: dict, out_path: Path):
    """Bar chart: mean ||x_hat|| / ||x_vl|| ratio. Closer to 1.0 = better."""
    tags = list(summaries.keys())
    means, stds, valid_tags = [], [], []
    for t in tags:
        ssu = summaries[t].get("SSU", {})
        m = ssu.get("mean_norm_ratio")
        s = ssu.get("std_norm_ratio", 0.0)
        if m is not None:
            means.append(m)
            stds.append(s or 0.0)
            valid_tags.append(t)
    if not valid_tags:
        print("  Skipping norm plot — no norm_ratio data found")
        return

    fig, ax = plt.subplots(figsize=(max(5, len(valid_tags) * 1.4), 4))
    x = np.arange(len(valid_tags))
    bars = ax.bar(x, means, yerr=stds, color=[_method_color(t) for t in valid_tags],
                  width=0.6, edgecolor="black", linewidth=0.7,
                  capsize=4, error_kw={"elinewidth": 1.2})
    ax.axhline(1.0, color="black", linestyle="-", linewidth=1.2,
               label="Perfect norm preservation (ratio=1)")
    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(t) for t in valid_tags], ha="center")
    ax.set_ylabel("||x_hat|| / ||x_vl|| (mean ± std)")
    ax.set_title("Norm Preservation: ||x_hat|| / ||x_vl||")
    ax.legend()
    ax.set_ylim(min(0.9, min(m - s for m, s in zip(means, stds))) - 0.01,
                max(1.1, max(m + s for m, s in zip(means, stds))) + 0.01)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_category_asr(summaries: dict, out_path: Path):
    """Grouped bar chart: per-category SSU ASR across methods."""
    # Collect all categories
    categories = set()
    for s in summaries.values():
        categories.update(s.get("SSU_per_category", {}).keys())
    if not categories:
        print("  Skipping category plot — no per-category data")
        return
    categories = sorted(categories)
    tags = list(summaries.keys())

    n_cats = len(categories)
    n_methods = len(tags)
    bar_w = 0.8 / n_methods
    x = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(max(8, n_cats * 1.5), 5))
    for i, tag in enumerate(tags):
        cat_data = summaries[tag].get("SSU_per_category", {})
        vals = [cat_data.get(c, {}).get("asr", np.nan) for c in categories]
        offset = (i - n_methods / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, width=bar_w * 0.9,
               color=_method_color(tag), label=_method_label(tag).replace("\n", " "),
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("Attack Success Rate ↓")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-Category SSU Attack Success Rate")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_combined_summary(summaries: dict, out_path: Path):
    """2×2 summary figure combining ASR, helpfulness, norm, and category ASR."""
    tags       = list(summaries.keys())
    asr_vals   = [summaries[t].get("SSU", {}).get("asr",           np.nan) for t in tags]
    help_vals  = [summaries[t].get("SSS", {}).get("refusal_rate",  np.nan) for t in tags]
    norm_means = [summaries[t].get("SSU", {}).get("mean_norm_ratio", np.nan) for t in tags]
    norm_stds  = [summaries[t].get("SSU", {}).get("std_norm_ratio",  0.0)   for t in tags]
    colors     = [_method_color(t) for t in tags]
    labels     = [_method_label(t).replace("\n", " ") for t in tags]
    x          = np.arange(len(tags))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Intervention Comparison: ShiftDC vs Spherical ShiftDC", fontsize=14)

    # (0,0) ASR
    ax = axes[0, 0]
    b = ax.bar(x, asr_vals, color=colors, edgecolor="black", linewidth=0.7)
    ax.bar_label(b, labels=[f"{v:.1%}" if not np.isnan(v) else "N/A"
                             for v in asr_vals], padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("ASR (SSU) ↓"); ax.set_ylim(0, 1.05)
    ax.set_title("Attack Success Rate (SSU)")

    # (0,1) Helpfulness
    ax = axes[0, 1]
    b = ax.bar(x, help_vals, color=colors, edgecolor="black", linewidth=0.7)
    ax.bar_label(b, labels=[f"{v:.1%}" if not np.isnan(v) else "N/A"
                             for v in help_vals], padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("False Refusal Rate (SSS) ↓")
    ax.set_title("Helpfulness Cost")

    # (1,0) Norm preservation
    ax = axes[1, 0]
    valid_norm = [(i, m, s) for i, (m, s) in enumerate(zip(norm_means, norm_stds))
                  if not np.isnan(m)]
    if valid_norm:
        idx, m_vals, s_vals = zip(*valid_norm)
        ax.bar(idx, m_vals, yerr=s_vals,
               color=[colors[i] for i in idx],
               edgecolor="black", linewidth=0.7, capsize=4)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("||x_hat|| / ||x_vl||")
    ax.set_title("Norm Preservation (1.0 = perfect)")

    # (1,1) Legend + numeric summary table
    ax = axes[1, 1]
    ax.axis("off")
    rows = [["Method", "ASR (SSU)↓", "Refusal (SSS)↓", "Norm Ratio"]]
    for t, a, h, n in zip(tags, asr_vals, help_vals, norm_means):
        rows.append([
            _method_label(t).replace("\n", " "),
            f"{a:.1%}" if not np.isnan(a) else "—",
            f"{h:.1%}" if not np.isnan(h) else "—",
            f"{n:.4f}" if not np.isnan(n) else "—",
        ])
    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(colors, labels)]
    ax.legend(handles=patches, loc="lower center", ncol=2)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=None,
                   help="Path to results directory (default: auto-resolved)")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = (Path(args.results_dir) if args.results_dir
                   else _MODEL_DIR / "outputs" / "results")
    plots_dir   = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading summaries from: {results_dir}")
    summaries = _load_summaries(results_dir)
    if not summaries:
        print("No *_summary.json files found. Run run_intervention.py first.")
        return
    print(f"  Found {len(summaries)} method(s): {list(summaries.keys())}")

    plot_asr_bar(         summaries, plots_dir / "asr_comparison.png")
    plot_helpfulness_bar( summaries, plots_dir / "helpfulness_comparison.png")
    plot_norm_preservation(summaries, plots_dir / "norm_preservation.png")
    plot_category_asr(    summaries, plots_dir / "category_asr.png")
    plot_combined_summary(summaries, plots_dir / "combined_summary.png")

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
