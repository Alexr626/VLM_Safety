#!/usr/bin/env python3
"""Draw + pin the shared CHAIR / AMBER-discriminative subsets (ONCE).

Both diagnostics experiments (reproduction grid + rotation-strength sweep) and
any reruns must score the IDENTICAL items so every cell is mutually comparable
and within-item flips are analyzable. This script draws those subsets with a
fixed seed and writes them to TRACKED, stable locations (not the dated results
tree), so they survive run-date changes and reruns:

  - CHAIR : 500 random COCO val2014 image ids  -> data/chair/pinned_chair_500.json
  - AMBER : stratified by dimension, 150 each from existence / attribute /
            relation (~450 total)            -> data/amber/pinned_amber_disc_450.json

The dimension tag is the annotation-derived `category` already surfaced by the
loader (existence/attribute/relation), NOT the raw query-file name — the plan's
`query_discriminative.json` is the full discriminative file, not existence-only.

Each pinned file is shaped so it can be passed straight to
`run_eval.py --subset_ids_file`: a dict keyed by benchmark name (`chair` /
`amber`) plus an ignored `_meta` block for provenance.

Idempotent: with a fixed seed the draw is deterministic; rerunning reproduces
the same ids. Pass --force to overwrite an existing pin.

Usage:
  python evaluation/chair_amber_diagnostics/draw_subsets.py
  python evaluation/chair_amber_diagnostics/draw_subsets.py --seed 1234 --force
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import (  # noqa: E402
    combined_json_path, benchmark_data_dir,
    _load_amber_annotations, _amber_discriminative_qtype,
)

CHAIR_N = 500
AMBER_PER_DIM = 150
AMBER_DIMS = ("existence", "attribute", "relation")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--chair_n", type=int, default=CHAIR_N)
    p.add_argument("--amber_per_dim", type=int, default=AMBER_PER_DIM)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing pinned files.")
    return p.parse_args()


def _write(path: Path, payload: dict, force: bool) -> None:
    if path.exists() and not force:
        print(f"  [exists] {path} (use --force to overwrite); leaving as-is")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [wrote]  {path}")


def draw_chair(seed: int, n: int) -> dict:
    entries = json.load(open(combined_json_path("chair")))
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(entries)), n))
    ids = [entries[i]["id"] for i in idxs]
    coco_ids = [(entries[i].get("raw") or {}).get("coco_id") for i in idxs]
    return {
        "chair": ids,
        "_meta": {
            "benchmark": "chair",
            "seed": seed,
            "n": len(ids),
            "source": "data/chair/combined.json (COCO val2014)",
            "coco_ids": coco_ids,
            "drawn_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


def draw_amber(seed: int, per_dim: int) -> dict:
    root = benchmark_data_dir("amber")
    annotations = _load_amber_annotations(root)
    entries = json.load(open(combined_json_path("amber")))
    by_dim: dict[str, list] = defaultdict(list)
    for e in entries:
        if e.get("task", "discriminative") != "discriminative":
            continue
        rid = (e.get("raw") or {}).get("id")
        ann = annotations.get(rid)
        if ann is None:
            continue
        dim = _amber_discriminative_qtype(ann.get("type", ""))
        by_dim[dim].append((e["id"], ann.get("truth", "")))

    rng = random.Random(seed)
    ids: list[str] = []
    strata = {}
    gold = Counter()
    for dim in AMBER_DIMS:
        pool = sorted(by_dim.get(dim, []))  # deterministic order before sampling
        k = min(per_dim, len(pool))
        chosen = rng.sample(pool, k)
        ids.extend(cid for cid, _ in chosen)
        strata[dim] = {"available": len(pool), "drawn": k}
        for _, truth in chosen:
            gold[truth] += 1
        if k < per_dim:
            print(f"  [warn] AMBER dim '{dim}' has only {len(pool)} items "
                  f"(< {per_dim}); drew {k}")
    return {
        "amber": ids,
        "_meta": {
            "benchmark": "amber",
            "task": "discriminative",
            "seed": seed,
            "n": len(ids),
            "stratified_by": "annotation-derived dimension (existence/attribute/relation)",
            "strata": strata,
            "gold_counts": dict(gold),
            "source": "data/amber/combined.json + data/amber/data/annotations.json",
            "drawn_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


def main():
    args = parse_args()
    print(f"=== drawing pinned subsets (seed={args.seed}) ===")

    chair_payload = draw_chair(args.seed, args.chair_n)
    chair_path = benchmark_data_dir("chair") / "pinned_chair_500.json"
    print(f"CHAIR: n={chair_payload['_meta']['n']}")
    _write(chair_path, chair_payload, args.force)

    amber_payload = draw_amber(args.seed, args.amber_per_dim)
    amber_path = benchmark_data_dir("amber") / "pinned_amber_disc_450.json"
    print(f"AMBER: n={amber_payload['_meta']['n']}  "
          f"strata={ {d: s['drawn'] for d, s in amber_payload['_meta']['strata'].items()} }  "
          f"gold={amber_payload['_meta']['gold_counts']}")
    _write(amber_path, amber_payload, args.force)

    print("\nPinned subset files (pass to run_eval.py --subset_ids_file):")
    print(f"  {chair_path}")
    print(f"  {amber_path}")


if __name__ == "__main__":
    main()
