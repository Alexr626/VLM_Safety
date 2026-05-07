#!/usr/bin/env python3
"""Plot Behavioral Eval — probe accuracy at predicting refusal.

Reads `probe_results.json` produced by `safety_probes.py` and produces two
figures, one per activation modality, each with **two side-by-side subplots**
(HoliSafe SSU Behavioral and MSSBench Behavioral). Every subplot draws one
layer-vs-accuracy curve per probe (up to 5).

The behavioral test sets share a target (`refused_vl`: did LLaVA refuse the
VL-image prompt?) and differ in the activation source feeding the probe:
  - `ssu_behavioral_{tt,vl}`: HoliSafe SSU eval activations.
  - `mssbench_behavioral_{tt,vl}`: MSSBench eval-split activations.

Output
------
  diagnostic_experiments/{model}/compositional_safety/outputs/results/plots/
    ├── behavioral_eval_tt.png   — left subplot ssu_behavioral_tt,
    │                               right subplot mssbench_behavioral_tt
    └── behavioral_eval_vl.png   — same shape, VL test sets

If a model's MSSBench refusal labels haven't been built yet (no
`mssbench_behavioral_*` keys in probe_results.json), the script still renders
the figure with the right subplot empty and a placeholder note.
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
    "semantic_safety_probe":                  "Semantic (CatQA)",
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

_PANELS = {
    "tt": [
        ("ssu_behavioral_tt",      "HoliSafe SSU Behavioral · TT"),
        ("mssbench_behavioral_tt", "MSSBench Behavioral · TT"),
    ],
    "vl": [
        ("ssu_behavioral_vl",      "HoliSafe SSU Behavioral · VL"),
        ("mssbench_behavioral_vl", "MSSBench Behavioral · VL"),
    ],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    return p.parse_args()


def _series(rows, probe, test):
    xs, ys = [], []
    for r in rows:
        v = r.get(f"{probe}__{test}__accuracy")
        if v is None:
            continue
        xs.append(r["layer"])
        ys.append(v)
    return (np.array(xs), np.array(ys)) if xs else (None, None)


def _plot_modality(rows, modality, model_name, out_path):
    panels = _PANELS[modality]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        f"Behavioral Eval — predict 'refused VL prompt' from activations "
        f"({modality.upper()}) · {model_name}",
        fontsize=12, fontweight="bold",
    )

    plotted_any = False
    for ax, (test_key, panel_title) in zip(axes, panels):
        n_curves = 0
        for probe in PROBES:
            xs, ys = _series(rows, probe, test_key)
            if xs is None:
                continue
            ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=4.5,
                    color=PROBE_COLORS[probe], label=PROBE_LABELS[probe],
                    alpha=0.9)
            n_curves += 1
            plotted_any = True

        ax.axhline(0.5, ls="--", color="gray", alpha=0.5, linewidth=1,
                   label="Chance")
        ax.set_xlabel("Layer")
        ax.set_title(panel_title, fontsize=11)
        ax.set_ylim(0.35, 1.05)
        ax.grid(True, alpha=0.25)
        if n_curves > 0:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
        else:
            ax.text(0.5, 0.5,
                    f"No data for\n{test_key}\n(run helper + safety_probes.py)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="#888")

    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    if not plotted_any:
        print(f"  [warn] no curves plotted for {modality}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    rows = [r for r in rows if r.get("layer", 0) > 0]
    if not rows:
        raise RuntimeError(f"No usable rows in {results_path}.")

    plt.rcParams.update({"font.size": 10})
    _plot_modality(rows, "tt", model_name, out_dir / "behavioral_eval_tt.png")
    _plot_modality(rows, "vl", model_name, out_dir / "behavioral_eval_vl.png")


if __name__ == "__main__":
    main()
