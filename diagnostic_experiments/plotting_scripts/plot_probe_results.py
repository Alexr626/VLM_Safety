#!/usr/bin/env python3
"""Plot Probe Results: cross-evaluation heatmap and layer-wise curves.

Reads `probe_results.json` produced by the multi-source `safety_probes.py`
(5 probes × N test sets) and produces:

  - cross_evaluation_heatmap.png  — all 5 probes × all available test sets
                                    at the layer that maximizes the canonical
                                    HoliSafe-TT compositional probe.
  - accuracy_curves_{probe_id}.png — per-probe layer-wise accuracy curves
                                    across the same test sets.
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

_EXPERIMENT_NAME = "compositional_safety"

PROBES = [
    "semantic_safety_probe",
    "compositional_safety_probe_holisafe_tt",
    "compositional_safety_probe_holisafe_vl",
    "compositional_safety_probe_mssbench_tt",
    "compositional_safety_probe_mssbench_vl",
]
PROBE_LABELS = {
    "semantic_safety_probe": "Semantic (CatQA)",
    "compositional_safety_probe_holisafe_tt": "Comp. HoliSafe TT",
    "compositional_safety_probe_holisafe_vl": "Comp. HoliSafe VL",
    "compositional_safety_probe_mssbench_tt": "Comp. MSSBench TT",
    "compositional_safety_probe_mssbench_vl": "Comp. MSSBench VL",
}
PROBE_COLORS = {
    "semantic_safety_probe":                  "#2563eb",
    "compositional_safety_probe_holisafe_tt": "#dc2626",
    "compositional_safety_probe_holisafe_vl": "#ea580c",
    "compositional_safety_probe_mssbench_tt": "#059669",
    "compositional_safety_probe_mssbench_vl": "#7c3aed",
}
TESTS = [
    "holisafe_eval_tt", "holisafe_eval_vl",
    "catqa_eval", "ssu_behavioral",
    "mssbench_eval_tt", "mssbench_eval_vl",
]
TEST_LABELS = {
    "holisafe_eval_tt": "HoliSafe Eval (TT)",
    "holisafe_eval_vl": "HoliSafe Eval (VL)",
    "catqa_eval": "CatQA Eval",
    "ssu_behavioral": "SSU Behavioral",
    "mssbench_eval_tt": "MSSBench Eval (TT)",
    "mssbench_eval_vl": "MSSBench Eval (VL)",
}
TEST_STYLES = {
    "holisafe_eval_tt": ("o", "-"),
    "holisafe_eval_vl": ("s", "--"),
    "catqa_eval":       ("^", ":"),
    "ssu_behavioral":   ("D", "-."),
    "mssbench_eval_tt": ("v", "-"),
    "mssbench_eval_vl": ("P", "--"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    return p.parse_args()


def _filter_present(rows, names, key_fn):
    """Return the subset of `names` for which any row has a non-None value."""
    out = []
    for n in names:
        for r in rows:
            v = r.get(key_fn(n))
            if v is not None and not np.isnan(v):
                out.append(n)
                break
    return out


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

    # Subset PROBES/TESTS to those that actually have data (avoids empty rows
    # when MSSBench probes/tests are not yet computed for a model).
    present_probes = _filter_present(
        data, PROBES,
        lambda p: f"{p}__holisafe_eval_tt__accuracy",
    )
    if not present_probes:
        # Fallback: keep semantic if it has any test
        present_probes = _filter_present(
            data, PROBES,
            lambda p: f"{p}__catqa_eval__accuracy",
        ) or PROBES
    present_tests = _filter_present(
        data, TESTS,
        lambda t: any(f"{p}__{t}__accuracy" in r for r in data for p in present_probes)
              and f"{present_probes[0]}__{t}__accuracy",
    )

    # ── Cross-evaluation heatmap at the best HoliSafe-TT layer ─────────────
    best_key = "compositional_safety_probe_holisafe_tt__holisafe_eval_tt__accuracy"
    accs = [(i, d.get(best_key)) for i, d in enumerate(data) if d.get(best_key) is not None]
    if accs:
        best_idx = max(accs, key=lambda t: t[1])[0]
        best_row = data[best_idx]
        best_layer = best_row["layer"]
        matrix = np.full((len(present_probes), len(present_tests)), float("nan"))
        for i, probe in enumerate(present_probes):
            for j, test in enumerate(present_tests):
                v = best_row.get(f"{probe}__{test}__accuracy")
                if v is not None:
                    matrix[i, j] = v
        fig_w = max(9.0, 1.6 * len(present_tests))
        fig_h = max(4.0, 0.8 * len(present_probes) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(len(present_tests)))
        ax.set_xticklabels([TEST_LABELS[t] for t in present_tests],
                           fontsize=9, rotation=20, ha="right")
        ax.set_yticks(range(len(present_probes)))
        ax.set_yticklabels([PROBE_LABELS[p] for p in present_probes], fontsize=9)
        for i in range(len(present_probes)):
            for j in range(len(present_tests)):
                v = matrix[i, j]
                if not np.isnan(v):
                    color = "white" if v > 0.78 or v < 0.45 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=10, color=color)
        ax.set_title(
            f"Cross-Evaluation at Best Layer ({best_layer}) · {model_name}\n"
            f"(layer chosen by max accuracy of comp_holisafe_tt on holisafe_eval_tt)"
        )
        fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
        fig.tight_layout()
        _save(fig, "cross_evaluation_heatmap.png")

    # ── Per-probe layer-wise accuracy curves ───────────────────────────────
    for probe in present_probes:
        fig, ax = plt.subplots(figsize=(14, 5))
        plotted = False
        for test in present_tests:
            key = f"{probe}__{test}__accuracy"
            vals = [d.get(key) for d in data]
            mask = np.array([v is not None for v in vals])
            if not any(mask):
                continue
            arr = np.array([v if v is not None else float("nan") for v in vals])
            marker, ls = TEST_STYLES.get(test, ("o", "-"))
            ax.plot(x[mask], arr[mask], marker=marker, linestyle=ls,
                    markersize=5, linewidth=1.5, label=TEST_LABELS[test], alpha=0.85)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.axhline(0.5, color="#888", linewidth=1, linestyle=":", alpha=0.5,
                   label="Chance")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Layer-wise: {PROBE_LABELS[probe]} · {model_name}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers], fontsize=8)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylim(0.35, 1.05)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9, ncol=2)
        fig.tight_layout()
        _save(fig, f"accuracy_curves_{probe}.png")


if __name__ == "__main__":
    main()
