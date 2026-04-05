#!/usr/bin/env python3
"""
Plot Probe Results: Cross-Evaluation Matrix and Layer-wise Curves
==================================================================
Generates:
  1. Cross-evaluation heatmap: probe x test set, at best layer
  2. Layer-wise accuracy curves for each probe on each test set
  3. ROC curves for behavioral prediction (if available)

Reads from: ../outputs/results/probe_results.json
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
with open(_RESULTS_DIR / "probe_results.json") as f:
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

SAFETY_BG = "#fef3c7"
safety_lo = np.searchsorted(layers, 6) - 0.5
safety_hi = np.searchsorted(layers, 14) + 0.5
safety_center = (np.searchsorted(layers, 6) + np.searchsorted(layers, 14)) / 2

PROBES = ["content_probe", "combinatorial_probe"]
PROBE_LABELS = {"content_probe": "Content (CatQA)", "combinatorial_probe": "Combinatorial (SSU-vs-SSS)"}
TESTS = ["holisafe_eval_tt", "holisafe_eval_vl", "catqa_full", "ssu_behavioral"]
TEST_LABELS = {
    "holisafe_eval_tt": "HoliSafe Eval (TT)",
    "holisafe_eval_vl": "HoliSafe Eval (VL)",
    "catqa_full": "CatQA (full)",
    "ssu_behavioral": "SSU Behavioral",
}

PROBE_COLORS = {"content_probe": "#2563eb", "combinatorial_probe": "#dc2626"}
TEST_STYLES = {
    "holisafe_eval_tt": ("o", "-"),
    "holisafe_eval_vl": ("s", "--"),
    "catqa_full": ("^", ":"),
    "ssu_behavioral": ("D", "-."),
}


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


# ── Plot 1: Cross-evaluation heatmap at best layer ──────────────────────────
# Find best layer by max combinatorial probe accuracy on HoliSafe eval TT
best_key = "combinatorial_probe__holisafe_eval_tt__accuracy"
accs = [(i, d.get(best_key)) for i, d in enumerate(data) if d.get(best_key) is not None]
if accs:
    best_idx = max(accs, key=lambda x: x[1])[0]
    best_row = data[best_idx]
    best_layer = best_row["layer"]

    matrix = np.full((len(PROBES), len(TESTS)), float("nan"))
    for i, probe in enumerate(PROBES):
        for j, test in enumerate(TESTS):
            key = f"{probe}__{test}__accuracy"
            val = best_row.get(key)
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=1.0)

    ax.set_xticks(range(len(TESTS)))
    ax.set_xticklabels([TEST_LABELS[t] for t in TESTS], fontsize=9, rotation=15, ha="right")
    ax.set_yticks(range(len(PROBES)))
    ax.set_yticklabels([PROBE_LABELS[p] for p in PROBES], fontsize=9)

    for i in range(len(PROBES)):
        for j in range(len(TESTS)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.75 or val < 0.45 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, color=color)

    ax.set_title(f"Cross-Evaluation: Probe Accuracy at Best Layer ({best_layer})\n"
                 f"LLaVA-1.5-7B  ·  Logistic Regression Probes")
    fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
    fig.tight_layout()
    _save(fig, "cross_evaluation_heatmap.png")


# ── Plot 2: Layer-wise accuracy curves ───────────────────────────────────────
for probe in PROBES:
    fig, ax = plt.subplots(figsize=(14, 5))

    for test in TESTS:
        key = f"{probe}__{test}__accuracy"
        vals = [d.get(key) for d in data]
        valid_mask = np.array([v is not None for v in vals])
        if not any(valid_mask):
            continue
        vals_arr = np.array([v if v is not None else float("nan") for v in vals])
        marker, linestyle = TEST_STYLES[test]
        ax.plot(x[valid_mask], vals_arr[valid_mask], marker=marker, linestyle=linestyle,
                markersize=5, linewidth=1.5, label=TEST_LABELS[test], alpha=0.85)

    ax.axhline(0.5, color="#888", linewidth=1, linestyle=":", alpha=0.5, label="Chance")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Layer-wise Probe Accuracy: {PROBE_LABELS[probe]}\n"
                 f"LLaVA-1.5-7B  ·  Cross-Evaluation on All Test Sets")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylim(0.35, 1.05)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    _annotate_safety(ax)
    fig.tight_layout()
    _save(fig, f"accuracy_curves_{probe}.png")


# ── Plot 3: AUC-ROC curves for behavioral prediction ────────────────────────
has_behavioral = any(d.get("content_probe__ssu_behavioral__auc_roc") is not None for d in data)

if has_behavioral:
    fig, ax = plt.subplots(figsize=(14, 5))

    for probe in PROBES:
        key = f"{probe}__ssu_behavioral__auc_roc"
        vals = [d.get(key) for d in data]
        valid_mask = np.array([v is not None for v in vals])
        if not any(valid_mask):
            continue
        vals_arr = np.array([v if v is not None else float("nan") for v in vals])
        ax.plot(x[valid_mask], vals_arr[valid_mask], "o-",
                color=PROBE_COLORS[probe], markersize=5, linewidth=1.8,
                label=PROBE_LABELS[probe], alpha=0.85)

    ax.axhline(0.5, color="#888", linewidth=1, linestyle=":", alpha=0.5, label="Chance")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Behavioral Prediction: AUC-ROC for Compliance vs Refusal (SSU only)\n"
                 "LLaVA-1.5-7B  ·  Probes Predicting Model Behavior on SSU Inputs")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylim(0.3, 1.05)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    _annotate_safety(ax)
    fig.tight_layout()
    _save(fig, "behavioral_auc_roc.png")
