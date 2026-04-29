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


def _plot_probe_modality(rows, probe, pairs, modality_label, model_name, out_path):
    """One probe × one modality → one PNG. Probes/test_keys without data are
    silently skipped; nothing is written if the probe has no curves at all."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plotted = False
    for test_key, label, color in pairs:
        xs, ys = _accuracy_series(rows, probe, test_key)
        if xs is None:
            continue
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=4.5,
                color=color, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5, linewidth=1, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"Compositional Eval · {PROBE_TITLES[probe]} · {modality_label} · {model_name}",
        fontsize=11,
    )
    ax.set_ylim(0.35, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return True


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
    # One PNG per (probe, modality) so each can be browsed in isolation.
    # Folder layout:
    #   plots/compositional_eval_tt/{probe_id}.png
    #   plots/compositional_eval_vl/{probe_id}.png
    n_written = 0
    for modality_label, pairs in (("TT", PAIRS_TT), ("VL", PAIRS_VL)):
        sub = out_dir / f"compositional_eval_{modality_label.lower()}"
        for probe in PROBES:
            out_path = sub / f"{probe}.png"
            if _plot_probe_modality(rows, probe, pairs, modality_label,
                                    model_name, out_path):
                n_written += 1
    if n_written == 0:
        print("  [warn] no compositional-eval curves were plotted "
              "(no matching probe×test data in probe_results.json)")


if __name__ == "__main__":
    main()
