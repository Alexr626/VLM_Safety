#!/usr/bin/env python3
"""Compute per-sample modality shifts m^l = x_vl^l - x_tt^l and aggregate stats."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import load_benchmark, DATASET_DATA_DIRS, ALL_BENCHMARKS
from src.extraction import ActivationCache, load_modality_shift_matrix, save_json
from src.model import _normalize_model_name
from src.paths import diagnostic_results_dir, experiment_artifacts_dir

EXPERIMENT = "modality_shift"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark", default="pope", choices=ALL_BENCHMARKS)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    model = _normalize_model_name(args.model)
    dataset_dir = DATASET_DATA_DIRS[args.benchmark]
    act_dir = _PROJECT_ROOT / "data" / dataset_dir / model / "activations"
    cache = ActivationCache(str(act_dir))
    samples = load_benchmark(args.benchmark, limit=args.limit)

    results_dir = diagnostic_results_dir(EXPERIMENT, model)
    results_dir.mkdir(parents=True, exist_ok=True)

    per_sample = []
    for s in tqdm(samples, desc="modality shift"):
        sid = s["id"]
        if not cache.exists(sid, "vl") or not cache.exists(sid, "tt"):
            continue
        shifts = {}
        vl = cache.load(sid, "vl")
        tt = cache.load(sid, "tt")
        for layer in vl:
            if layer in tt:
                m = vl[layer] - tt[layer]
                shifts[layer] = {
                    "norm": float(np.linalg.norm(m)),
                    "label": s["label"],
                }
        per_sample.append({"id": sid, "label": s["label"], "shifts": shifts})

    save_json(per_sample, str(results_dir / "per_sample_shifts.json"))

    # Aggregate mean norm per layer by label
    by_label: dict = {}
    for rec in per_sample:
        lbl = rec["label"]
        by_label.setdefault(lbl, {})
        for layer, stats in rec["shifts"].items():
            by_label[lbl].setdefault(layer, []).append(stats["norm"])

    aggregate = {
        lbl: {layer: {"mean_norm": float(np.mean(vals)), "n": len(vals)}
              for layer, vals in layers.items()}
        for lbl, layers in by_label.items()
    }
    save_json(aggregate, str(results_dir / "aggregate_stats.json"))
    save_json({"model": model, "benchmark": args.benchmark, "n": len(per_sample)},
              str(results_dir / "run_metadata.json"))
    print(f"Wrote results -> {results_dir}")


if __name__ == "__main__":
    main()
