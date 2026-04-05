#!/usr/bin/env python3
"""Plot Probe Results: Cross-Evaluation Matrix and Layer-wise Curves"""

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

_EXPERIMENT_NAME = "combinatorial_safety"
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
    "holisafe_eval_tt": ("o", "-"), "holisafe_eval_vl": ("s", "--"),
    "catqa_full": ("^", ":"), "ssu_behavioral": ("D", "-."),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    results_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME / "outputs" / "results"
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "probe_results.json") as f:
        data = json.load(f)
    data = [d for d in data if d["layer"] > 0]
    layers = np.array([d["layer"] for d in data])
    x = np.arange(len(layers))

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
    })

    def _save(fig, name):
        path = out_dir / name
        fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {path}")
        plt.close(fig)

    best_key = "combinatorial_probe__holisafe_eval_tt__accuracy"
    accs = [(i, d.get(best_key)) for i, d in enumerate(data) if d.get(best_key) is not None]
    if accs:
        best_idx = max(accs, key=lambda t: t[1])[0]
        best_row = data[best_idx]
        best_layer = best_row["layer"]
        matrix = np.full((len(PROBES), len(TESTS)), float("nan"))
        for i, probe in enumerate(PROBES):
            for j, test in enumerate(TESTS):
                v = best_row.get(f"{probe}__{test}__accuracy")
                if v is not None:
                    matrix[i, j] = v
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(len(TESTS)))
        ax.set_xticklabels([TEST_LABELS[t] for t in TESTS], fontsize=9, rotation=15, ha="right")
        ax.set_yticks(range(len(PROBES)))
        ax.set_yticklabels([PROBE_LABELS[p] for p in PROBES], fontsize=9)
        for i in range(len(PROBES)):
            for j in range(len(TESTS)):
                v = matrix[i, j]
                if not np.isnan(v):
                    color = "white" if v > 0.75 or v < 0.45 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11, color=color)
        ax.set_title(f"Cross-Evaluation at Best Layer ({best_layer}) · {model_name}")
        fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
        fig.tight_layout()
        _save(fig, "cross_evaluation_heatmap.png")

    for probe in PROBES:
        fig, ax = plt.subplots(figsize=(14, 5))
        for test in TESTS:
            key = f"{probe}__{test}__accuracy"
            vals = [d.get(key) for d in data]
            mask = np.array([v is not None for v in vals])
            if not any(mask):
                continue
            arr = np.array([v if v is not None else float("nan") for v in vals])
            marker, ls = TEST_STYLES[test]
            ax.plot(x[mask], arr[mask], marker=marker, linestyle=ls,
                    markersize=5, linewidth=1.5, label=TEST_LABELS[test], alpha=0.85)
        ax.axhline(0.5, color="#888", linewidth=1, linestyle=":", alpha=0.5, label="Chance")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Layer-wise: {PROBE_LABELS[probe]} · {model_name}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers], fontsize=8)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylim(0.35, 1.05)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
        fig.tight_layout()
        _save(fig, f"accuracy_curves_{probe}.png")


if __name__ == "__main__":
    main()
