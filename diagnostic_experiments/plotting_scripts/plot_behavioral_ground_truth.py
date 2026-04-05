#!/usr/bin/env python3
"""
Plot Behavioral Ground Truth: Two-Axis LLM Classification Results
Generates:
  1. Grouped bar chart: ASR (harmful-content rate) by group x condition
  2. Stacked bar chart: SAFETY_AWARENESS distribution by group x condition
  3. Per-category heatmap of SSU ASR
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_DIAGNOSTIC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DIAGNOSTIC_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from src.model import _normalize_model_name

_EXPERIMENT_NAME = "behavioral_ground_truth"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    results_dir = (_DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME /
                   "outputs" / "results")
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "refusal_summary.json") as f:
        summary = json.load(f)

    # Handle both legacy (refusal_rate_*) and twoaxis formats
    conditions = summary.get("conditions") or ["vl", "tt", "ct"]
    legacy = "refusal_rate_vl" in summary.get("SSS", {})

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
    })
    COLOR_SSS, COLOR_SSU = "#2563eb", "#dc2626"

    def _save(fig, name):
        path = out_dir / name
        fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {path}")
        plt.close(fig)

    # ── Plot 1: Grouped bar: ASR (harmful content) ─────────────────────────
    x = np.arange(len(conditions))
    bw = 0.35
    if legacy:
        sss_rates = [summary["SSS"].get(f"refusal_rate_{c}", 0) for c in conditions]
        ssu_rates = [summary["SSU"].get(f"refusal_rate_{c}", 0) for c in conditions]
        ylabel = "Refusal Rate (keyword)"
    else:
        sss_rates = [summary["SSS"][c]["asr_harmful_content"] for c in conditions]
        ssu_rates = [summary["SSU"][c]["asr_harmful_content"] for c in conditions]
        ylabel = "ASR (Harmful Content Rate)"

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - bw/2, sss_rates, bw, color=COLOR_SSS, alpha=0.85, label="SSS")
    b2 = ax.bar(x + bw/2, ssu_rates, bw, color=COLOR_SSU, alpha=0.85, label="SSU")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Behavioral Ground Truth: ASR by Group × Condition\n{model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in conditions])
    ax.set_ylim(0, max(max(sss_rates), max(ssu_rates), 0.1) * 1.2 + 0.05)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlabel("Input Condition")
    fig.tight_layout()
    _save(fig, "asr_grouped.png")

    # ── Plot 2: Stacked awareness distribution (twoaxis only) ──────────────
    if not legacy:
        levels = ["NONE", "WEAK", "STRONG"]
        level_colors = {"NONE": "#dc2626", "WEAK": "#f59e0b", "STRONG": "#059669"}

        fig, ax = plt.subplots(figsize=(10, 5))
        bars_x = np.arange(len(conditions) * 2)  # SSS, SSU per condition
        labels = []
        bottoms = np.zeros(len(bars_x))
        totals = np.zeros(len(bars_x))
        for i, cond in enumerate(conditions):
            for j, group in enumerate(["SSS", "SSU"]):
                dist = summary[group][cond]["awareness_distribution"]
                totals[i * 2 + j] = sum(dist.values())
                labels.append(f"{cond.upper()}-{group}")
        for lvl in levels:
            vals = []
            for i, cond in enumerate(conditions):
                for group in ["SSS", "SSU"]:
                    dist = summary[group][cond]["awareness_distribution"]
                    tot = sum(dist.values()) or 1
                    vals.append(dist.get(lvl, 0) / tot)
            vals = np.array(vals)
            ax.bar(bars_x, vals, bottom=bottoms, color=level_colors[lvl],
                   label=lvl, edgecolor="white", linewidth=0.3)
            bottoms += vals
        ax.set_xticks(bars_x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Proportion")
        ax.set_title(f"Safety Awareness Distribution by Group × Condition\n{model_name}")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        _save(fig, "awareness_stacked.png")

    # ── Plot 3: Per-category heatmap for SSU ───────────────────────────────
    cat_data = summary.get("ssu_per_category", {})
    if cat_data:
        categories = sorted(cat_data.keys())
        if legacy:
            matrix = np.array([
                [cat_data[c].get(f"refusal_rate_{cond}", 0) for cond in conditions]
                for c in categories
            ])
        else:
            matrix = np.array([
                [cat_data[c][cond]["asr_harmful_content"] for cond in conditions]
                for c in categories
            ])
        fig, ax = plt.subplots(figsize=(8, max(4, len(categories) * 0.45)))
        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([c.upper() for c in conditions])
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories, fontsize=9)
        for i in range(len(categories)):
            for j in range(len(conditions)):
                v = matrix[i, j]
                color = "white" if v > 0.6 or v < 0.2 else "black"
                n = cat_data[categories[i]]["n"]
                ax.text(j, i, f"{v:.0%}\n(n={n})", ha="center", va="center",
                        fontsize=8, color=color)
        ax.set_title(f"SSU ASR by Harm Category × Condition · {model_name}")
        ax.set_xlabel("Input Condition")
        fig.colorbar(im, ax=ax, label="ASR", shrink=0.8)
        fig.tight_layout()
        _save(fig, "ssu_category_heatmap.png")


if __name__ == "__main__":
    main()
