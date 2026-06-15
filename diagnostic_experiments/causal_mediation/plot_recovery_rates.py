#!/usr/bin/env python3
"""Plot causal mediation recovery rates per layer."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import _normalize_model_name
from src.paths import diagnostic_results_dir, diagnostic_plots_dir

EXPERIMENT = "causal_mediation"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    args = p.parse_args()
    model = _normalize_model_name(args.model)
    results_dir = diagnostic_results_dir(EXPERIMENT, model)
    plots_dir = diagnostic_plots_dir(EXPERIMENT, model)
    plots_dir.mkdir(parents=True, exist_ok=True)

    path = results_dir / "recovery_rates.json"
    with open(path) as f:
        data = json.load(f)

    plt.figure(figsize=(10, 5))
    for comp, layer_stats in data.items():
        layers = sorted(layer_stats.keys(), key=int)
        ys = [layer_stats[l]["mean"] for l in layers]
        xs = [int(l) for l in layers]
        plt.plot(xs, ys, marker="o", label=comp)

    plt.axhline(0, color="gray", linewidth=0.5)
    plt.xlabel("Layer")
    plt.ylabel("Mean recovery rate")
    plt.title(f"Causal mediation — {model}")
    plt.legend()
    plt.tight_layout()
    out = plots_dir / "recovery_rates.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
