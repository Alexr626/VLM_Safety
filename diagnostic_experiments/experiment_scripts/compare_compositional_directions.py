#!/usr/bin/env python3
"""
Cross-Direction Cosine Comparison
==================================
Loads the CatQA semantic safety direction `s^l` and the four compositional
safety directions `c^l_{source,representation}` for a given model, then
writes the per-layer pairwise cosine-similarity matrix between all five.

This is the figure data behind the manuscript's "how much does the
source/representation choice move the direction estimate?" claim.

Output
------
  diagnostic_experiments/{model}/compositional_safety/outputs/results/
    └── cross_direction_cosine.json

Schema (one entry per layer present in s^l):
  {
    "layer": 7,
    "available": ["semantic", "holisafe_tt", "holisafe_vl", "mssbench_tt", "mssbench_vl"],
    "cosine_matrix": {
      "semantic":     {"holisafe_tt": ..., "holisafe_vl": ..., "mssbench_tt": ..., "mssbench_vl": ...},
      "holisafe_tt":  {"holisafe_vl": ..., "mssbench_tt": ..., "mssbench_vl": ...},
      "holisafe_vl":  {"mssbench_tt": ..., "mssbench_vl": ...},
      "mssbench_tt":  {"mssbench_vl": ...}
    }
  }
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_DIAGNOSTIC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DIAGNOSTIC_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction import save_json
from src.model import _normalize_model_name


_SOURCES = ["holisafe_tt", "holisafe_vl", "mssbench_tt", "mssbench_vl"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    return p.parse_args()


def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    shared_artifacts = _PROJECT_ROOT / "experiment_artifacts" / model_name
    results_dir = (_DIAGNOSTIC_ROOT / model_name / "compositional_safety"
                   / "outputs" / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load every available direction ──────────────────────────────────────
    directions: dict[str, dict[str, np.ndarray]] = {}

    sd_path = shared_artifacts / "vl_activation_shift" / "safety_direction_vectors.npz"
    if sd_path.exists():
        directions["semantic"] = dict(np.load(sd_path))
        print(f"  loaded semantic from {sd_path} "
              f"({len(directions['semantic'])} layers)")
    else:
        print(f"  WARN: semantic direction not found at {sd_path}")

    for src in _SOURCES:
        path = (shared_artifacts / "compositional_safety" / src
                / "compositional_safety_direction_vectors.npz")
        if path.exists():
            directions[src] = dict(np.load(path))
            print(f"  loaded {src} from {path} ({len(directions[src])} layers)")
        else:
            print(f"  WARN: {src} not found at {path}; will be omitted")

    if "semantic" not in directions:
        raise FileNotFoundError(
            "No semantic direction available. Run vl_activation_shift.py first."
        )

    # Layer set = layers where the semantic direction exists.
    layers = sorted(int(k.replace("layer_", "")) for k in directions["semantic"])

    # Order columns/rows consistently: semantic first, then sources.
    name_order = ["semantic"] + [s for s in _SOURCES if s in directions]

    rows = []
    for layer in layers:
        layer_key = f"layer_{layer}"
        # Materialise float64 vectors for the names that have this layer.
        present = [n for n in name_order if layer_key in directions.get(n, {})]
        vecs = {n: directions[n][layer_key].astype(np.float64) for n in present}

        cosine_matrix: dict[str, dict[str, float]] = {}
        for i, a in enumerate(present):
            inner: dict[str, float] = {}
            for b in present[i + 1:]:
                inner[b] = _cosine_sim(vecs[a], vecs[b])
            if inner:
                cosine_matrix[a] = inner

        rows.append({
            "layer": layer,
            "available": present,
            "cosine_matrix": cosine_matrix,
        })

    out_path = results_dir / "cross_direction_cosine.json"
    save_json(rows, str(out_path))
    print(f"\nWrote {out_path}")

    # Print a small summary at the diagonal-adjacent cells (semantic vs each source).
    sem_cells = {s: [] for s in _SOURCES if s in directions}
    for row in rows:
        sem_row = row["cosine_matrix"].get("semantic", {})
        for s, vals in sem_cells.items():
            v = sem_row.get(s)
            if v is not None and not np.isnan(v):
                vals.append(v)
    if sem_cells:
        print("\nMean cos(semantic, c^l_*):")
        for s, vals in sem_cells.items():
            if vals:
                print(f"  {s}: mean={np.mean(vals):.3f}  "
                      f"min={np.min(vals):.3f}  max={np.max(vals):.3f}  "
                      f"(n={len(vals)})")


if __name__ == "__main__":
    main()
