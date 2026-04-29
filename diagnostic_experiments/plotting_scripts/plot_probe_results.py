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
TESTS_TT = [
    "catqa_eval",
    "holisafe_eval_tt",
    "ssu_behavioral",
    "holisafe_eval_sss_vs_usu_tt",
    "holisafe_eval_sss_vs_suu_tt",
    "holisafe_eval_sss_vs_uuu_tt",
    "mssbench_eval_tt",
]
TESTS_VL = [
    "holisafe_eval_vl",
    "holisafe_eval_sss_vs_usu_vl",
    "holisafe_eval_sss_vs_suu_vl",
    "holisafe_eval_sss_vs_uuu_vl",
    "mssbench_eval_vl",
]
TESTS = TESTS_TT + TESTS_VL  # used for the cross-evaluation heatmap
TEST_LABELS = {
    "catqa_eval":                    "CatQA Eval",
    "holisafe_eval_tt":              "HoliSafe Eval SSS/SSU (TT)",
    "ssu_behavioral":                "SSU Behavioral",
    "holisafe_eval_sss_vs_usu_tt":   "HoliSafe SSS-vs-USU (TT)",
    "holisafe_eval_sss_vs_suu_tt":   "HoliSafe SSS-vs-SUU (TT)",
    "holisafe_eval_sss_vs_uuu_tt":   "HoliSafe SSS-vs-UUU (TT)",
    "mssbench_eval_tt":              "MSSBench Eval (TT)",
    "holisafe_eval_vl":              "HoliSafe Eval SSS/SSU (VL)",
    "holisafe_eval_sss_vs_usu_vl":   "HoliSafe SSS-vs-USU (VL)",
    "holisafe_eval_sss_vs_suu_vl":   "HoliSafe SSS-vs-SUU (VL)",
    "holisafe_eval_sss_vs_uuu_vl":   "HoliSafe SSS-vs-UUU (VL)",
    "mssbench_eval_vl":              "MSSBench Eval (VL)",
}
TEST_STYLES = {
    "catqa_eval":                    ("^", ":",  "#7f7f7f"),
    "holisafe_eval_tt":              ("o", "-",  "#1f77b4"),
    "ssu_behavioral":                ("D", "-.", "#9467bd"),
    "holisafe_eval_sss_vs_usu_tt":   ("v", "-",  "#ff7f0e"),
    "holisafe_eval_sss_vs_suu_tt":   ("v", "-",  "#2ca02c"),
    "holisafe_eval_sss_vs_uuu_tt":   ("v", "-",  "#d62728"),
    "mssbench_eval_tt":              ("P", "-",  "#7c3aed"),
    "holisafe_eval_vl":              ("s", "--", "#1f77b4"),
    "holisafe_eval_sss_vs_usu_vl":   ("v", "-",  "#ff7f0e"),
    "holisafe_eval_sss_vs_suu_vl":   ("v", "-",  "#2ca02c"),
    "holisafe_eval_sss_vs_uuu_vl":   ("v", "-",  "#d62728"),
    "mssbench_eval_vl":              ("P", "--", "#7c3aed"),
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

    # Subset PROBES/TESTS to those that actually have data (avoids empty rows
    # when MSSBench probes/tests are not yet computed for a model).
    def _has_any(probe_or_test_key: str) -> bool:
        return any(probe_or_test_key in r for r in data)

    present_probes = []
    for probe in PROBES:
        for test in TESTS:
            if _has_any(f"{probe}__{test}__accuracy"):
                present_probes.append(probe)
                break

    def _present_subset(test_list):
        return [t for t in test_list if any(
            _has_any(f"{p}__{t}__accuracy") for p in present_probes)]

    present_tests = _present_subset(TESTS)
    present_tests_tt = _present_subset(TESTS_TT)
    present_tests_vl = _present_subset(TESTS_VL)

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

    # ── Per-probe layer-wise accuracy curves (split by representation) ─────
    # Two folders so users can browse one modality at a time:
    #   plots/accuracy_curves_tt/{probe_id}.png — text-only test sets
    #   plots/accuracy_curves_vl/{probe_id}.png — image+text test sets
    def _plot_probe_curves(probe: str, tests: list, modality: str):
        if not tests:
            return
        fig, ax = plt.subplots(figsize=(14, 5))
        plotted = False
        for test in tests:
            key = f"{probe}__{test}__accuracy"
            vals = [d.get(key) for d in data]
            mask = np.array([v is not None for v in vals])
            if not any(mask):
                continue
            arr = np.array([v if v is not None else float("nan") for v in vals])
            style = TEST_STYLES.get(test, ("o", "-", None))
            marker, ls = style[0], style[1]
            color = style[2] if len(style) >= 3 and style[2] else None
            ax.plot(x[mask], arr[mask], marker=marker, linestyle=ls,
                    markersize=5, linewidth=1.5,
                    color=color, label=TEST_LABELS[test], alpha=0.85)
            plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.axhline(0.5, color="#888", linewidth=1, linestyle=":", alpha=0.5,
                   label="Chance")
        ax.set_ylabel("Accuracy")
        ax.set_title(
            f"Layer-wise · {PROBE_LABELS[probe]} · {modality.upper()} test sets · {model_name}"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers], fontsize=8)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylim(0.35, 1.05)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9, ncol=2)
        fig.tight_layout()
        sub = out_dir / f"accuracy_curves_{modality}"
        sub.mkdir(parents=True, exist_ok=True)
        _save(fig, str(Path(f"accuracy_curves_{modality}") / f"{probe}.png"))

    for probe in present_probes:
        _plot_probe_curves(probe, present_tests_tt, "tt")
        _plot_probe_curves(probe, present_tests_vl, "vl")


if __name__ == "__main__":
    main()
