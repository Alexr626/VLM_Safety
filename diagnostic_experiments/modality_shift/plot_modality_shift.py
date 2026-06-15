#!/usr/bin/env python3
"""Plot layer-wise modality-shift norms by label."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import _normalize_model_name
from src.paths import diagnostic_results_dir, diagnostic_plots_dir

EXPERIMENT = "modality_shift"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    args = p.parse_args()
    model = _normalize_model_name(args.model)
    results_dir = diagnostic_results_dir(EXPERIMENT, model)
    plots_dir = diagnostic_plots_dir(EXPERIMENT, model)
    plots_dir.mkdir(parents=True, exist_ok=True)

    agg_path = results_dir / "aggregate_stats.json"
    if not agg_path.exists():
        raise FileNotFoundError(f"Missing {agg_path}. Run compute_modality_shift.py first.")

    with open(agg_path) as f:
        agg = json.load(f)

    plt.figure(figsize=(10, 5))
    for label, layer_stats in agg.items():
        layers = sorted(layer_stats.keys(), key=lambda x: int(x.split("_")[1]))
        ys = [layer_stats[l]["mean_norm"] for l in layers]
        xs = [int(l.split("_")[1]) for l in layers]
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Layer")
    plt.ylabel("Mean ||m^l||")
    plt.title(f"Modality shift norms — {model}")
    plt.legend()
    plt.tight_layout()
    out = plots_dir / "modality_shift_norms.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
