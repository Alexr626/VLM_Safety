#!/usr/bin/env python3
"""
Plot Compositional Eval: per-layer probe accuracy on SSS-vs-{SSU,USU,SUU,UUU}
HoliSafe eval pairs, separately for TT and VL activations.

Inputs
------
  diagnostic_experiments/{model}/compositional_safety/outputs/results/probe_results.json

Outputs (to the same outputs/results/plots/ directory)
-------
  compositional_eval_tt.png   — 2 panels (one per probe), curves for the
                                4 HoliSafe eval pairs + CatQA.
  compositional_eval_vl.png   — same, VL activations, no CatQA (text-only ref).

Each panel is a per-layer accuracy curve with a 0.5 chance-line.
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
PROBE_TITLES = {
    "semantic_safety_probe":                  "Semantic (CatQA)",
    "compositional_safety_probe_holisafe_tt": "Comp. HoliSafe TT",
    "compositional_safety_probe_holisafe_vl": "Comp. HoliSafe VL",
    "compositional_safety_probe_mssbench_tt": "Comp. MSSBench TT",
    "compositional_safety_probe_mssbench_vl": "Comp. MSSBench VL",
}

# (test_key, label, color) per modality. The TT panel additionally includes
# CatQA; both panels surface MSSBench-eval when present.
PAIRS_TT = [
    ("holisafe_eval_tt",            "SSS vs SSU",      "#1f77b4"),
    ("holisafe_eval_sss_vs_usu_tt", "SSS vs USU",      "#ff7f0e"),
    ("holisafe_eval_sss_vs_suu_tt", "SSS vs SUU",      "#2ca02c"),
    ("holisafe_eval_sss_vs_uuu_tt", "SSS vs UUU",      "#d62728"),
    ("catqa_eval",                  "CatQA",           "#7f7f7f"),
    ("mssbench_eval_tt",            "MSSBench Eval",   "#7c3aed"),
]
PAIRS_VL = [
    ("holisafe_eval_vl",            "SSS vs SSU",      "#1f77b4"),
    ("holisafe_eval_sss_vs_usu_vl", "SSS vs USU",      "#ff7f0e"),
    ("holisafe_eval_sss_vs_suu_vl", "SSS vs SUU",      "#2ca02c"),
    ("holisafe_eval_sss_vs_uuu_vl", "SSS vs UUU",      "#d62728"),
    ("mssbench_eval_vl",            "MSSBench Eval",   "#7c3aed"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    return p.parse_args()


def _accuracy_series(rows, probe, test_key):
    """Return (xs, ys) for non-None per-layer accuracies, or (None, None)."""
    xs, ys = [], []
    for r in rows:
        v = r.get(f"{probe}__{test_key}__accuracy")
        if v is None:
            continue
        xs.append(r["layer"])
        ys.append(v)
    if not xs:
        return None, None
    return np.array(xs), np.array(ys)


def _plot_modality(rows, pairs, modality_label, out_path):
    """Grid of probe panels (one per available probe) showing per-test
    layer-wise accuracy curves. Probes without any data are skipped."""
    # Filter PROBES to only those with data for at least one test in `pairs`.
    present = []
    for probe in PROBES:
        for test_key, _, _ in pairs:
            if _accuracy_series(rows, probe, test_key)[0] is not None:
                present.append(probe)
                break
    if not present:
        print(f"  [warn] no curves plotted for {modality_label}")
        return

    n = len(present)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.2 * n_rows),
                             sharey=True, squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.axis("off")
    for ax, probe in zip(flat[:n], present):
        for test_key, label, color in pairs:
            xs, ys = _accuracy_series(rows, probe, test_key)
            if xs is None:
                continue
            ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=4.5,
                    color=color, label=label)
        ax.axhline(0.5, ls="--", color="gray", alpha=0.5, linewidth=1)
        ax.set_xlabel("Layer")
        ax.set_title(PROBE_TITLES[probe], fontsize=10)
        ax.set_ylim(0.35, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    fig.suptitle(f"Compositional Eval — {modality_label} activations",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    results_dir = (_DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME
                   / "outputs" / "results")
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "probe_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Run safety_probes.py --model {args.model} first."
        )
    with open(results_path) as f:
        rows = json.load(f)
    # Drop layer 0 (embedding) for visual consistency with plot_probe_results.py.
    rows = [r for r in rows if r.get("layer", 0) > 0]
    if not rows:
        raise RuntimeError(f"No usable rows in {results_path}.")

    plt.rcParams.update({"font.size": 10})
    _plot_modality(rows, PAIRS_TT, "TT", out_dir / "compositional_eval_tt.png")
    _plot_modality(rows, PAIRS_VL, "VL", out_dir / "compositional_eval_vl.png")


if __name__ == "__main__":
    main()
