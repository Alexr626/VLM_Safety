#!/usr/bin/env python3
"""Reproducibility check: overlapping POPE-30-yes items vs old dumps.

Compares the first 10 unique yes items (new pin items 1–10 =
``pope_random_00000``…``00018`` even indices) against the old
``pope30_windowed_steering`` / ``pope30_existence_yes_baseline`` manifests.

Reports divergent (cell_id, item_id, condition_id) counts for ``response`` and
all ``score_*`` fields. Divergence is an environment-drift signal — not a gate.

Usage::

    python diagnostic_experiments/perception_diag/check_pope30_yes_reproducibility.py \\
      --model llava-1.5-7b-hf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from diagnostic_experiments.perception_diag.build_windowed_steering_summary import (  # noqa: E402
    expected_cell_ids,
    load_manifest,
)
from src.paths import perception_dump_dir, project_root  # noqa: E402

OVERLAP_IDS = [
    "pope_random_00000",
    "pope_random_00002",
    "pope_random_00004",
    "pope_random_00006",
    "pope_random_00008",
    "pope_random_00010",
    "pope_random_00012",
    "pope_random_00014",
    "pope_random_00016",
    "pope_random_00018",
]

CONDITIONS = ("neutral", "assertive_toward_no")
SCORE_PREFIX = "score_"

MODEL_LAYERS = {
    "llava-1.5-7b-hf": 32,
    "qwen2.5-vl-7b-instruct": 28,
}


def _score_keys(row: dict) -> List[str]:
    return sorted(k for k in row if k.startswith(SCORE_PREFIX))


def _vals_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:  # both NaN
            return True
        return a == b
    return a == b


def compare_row(old: dict, new: dict) -> List[str]:
    diverged: List[str] = []
    if not _vals_equal(old.get("response"), new.get("response")):
        diverged.append("response")
    keys = sorted(set(_score_keys(old)) | set(_score_keys(new)))
    for k in keys:
        if not _vals_equal(old.get(k), new.get(k)):
            diverged.append(k)
    return diverged


def check_model(model_short: str) -> dict:
    n_layers = MODEL_LAYERS[model_short]
    new_root = perception_dump_dir("pope", model_short, "pope30_yes_windowed_steering")
    old_steered = perception_dump_dir("pope", model_short, "pope30_windowed_steering")
    old_baseline = (
        perception_dump_dir("pope", model_short, "pope30_existence_yes_baseline")
        / "baseline"
    )

    cells = ["baseline"] + expected_cell_ids(n_layers, include_baseline=False)
    n_compared = 0
    n_divergent_records = 0
    divergent_cell_ids: Set[str] = set()
    examples: List[dict] = []
    missing: List[str] = []

    for cell_id in cells:
        if cell_id == "baseline":
            old_man = load_manifest(old_baseline)
            new_man = load_manifest(new_root / "baseline")
        else:
            old_man = load_manifest(old_steered / cell_id)
            new_man = load_manifest(new_root / cell_id)
        if not old_man:
            missing.append(f"old:{cell_id}")
            continue
        if not new_man:
            missing.append(f"new:{cell_id}")
            continue
        for iid in OVERLAP_IDS:
            for cond in CONDITIONS:
                old = old_man.get((iid, cond))
                new = new_man.get((iid, cond))
                if old is None or new is None:
                    missing.append(f"{cell_id}/{iid}/{cond}")
                    continue
                n_compared += 1
                diffs = compare_row(old, new)
                if diffs:
                    n_divergent_records += 1
                    divergent_cell_ids.add(cell_id)
                    if len(examples) < 20:
                        examples.append(
                            {
                                "cell_id": cell_id,
                                "item_id": iid,
                                "condition_id": cond,
                                "fields": diffs,
                            }
                        )

    return {
        "model_short": model_short,
        "n_overlap_items": len(OVERLAP_IDS),
        "n_cells_checked": len(cells),
        "n_records_compared": n_compared,
        "n_divergent_records": n_divergent_records,
        "n_divergent_cell_ids": len(divergent_cell_ids),
        "divergent_cell_ids": sorted(divergent_cell_ids),
        "n_missing": len(missing),
        "missing_sample": missing[:20],
        "divergent_examples": examples,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_LAYERS.keys()),
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional JSON output path.",
    )
    args = p.parse_args()
    report = check_model(args.model)
    print(json.dumps(report, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
