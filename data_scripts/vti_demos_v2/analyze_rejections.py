#!/usr/bin/env python3
"""Rejection diagnostics for demos_v2 stage-1 / stage-4 artifacts.

Permanent pipeline health tool. Writes ``data/vti/v2/rejection_analysis.json``
and prints summary tables. Works on v2.0 rejects (string reasons / failures
lists) and v2.1 rejects (explicit ``failing_dimensions`` when present).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG.parent))

from vti_demos_v2.io_utils import read_jsonl, write_summary  # noqa: E402
from src.paths import vti_demos_v2_dir  # noqa: E402

# v2.0 stage-4 statement map (8 statements). v2.1 stores dimension on each failure.
_V20_STAGE4_IDX_TO_DIM = {
    1: "truthful_S1",
    2: "truthful_S2",
    3: "truthful_S3",
    4: "truthful_S4_relation",
    5: "existence_false",
    6: "attribute_false",
    7: "counting_false",
    8: "relation_false",
}

_STAGE1_BUCKETS = (
    "counting_check",
    "relation_check",
    "distractor",
    "attribute",
    "unparseable",
    "other",
)


def _stage1_bucket(rec: dict) -> str:
    reason = str(rec.get("reject_reason") or "")
    if reason == "unparseable" or "unparseable" in reason:
        return "unparseable"
    for key in ("counting_check", "relation_check", "distractor", "attribute"):
        if reason.startswith(f"{key}:") or f"{key}:" in reason.split(";")[0]:
            # multi-fail: take first token
            for part in reason.split(";"):
                part = part.strip()
                for k in ("counting_check", "relation_check", "distractor", "attribute"):
                    if part.startswith(f"{k}:"):
                        return k
            return key
    # structured stage1 payload
    s1 = rec.get("stage1") or {}
    for k in ("counting_check", "relation_check", "distractor", "attribute"):
        if isinstance(s1.get(k), dict) and s1[k].get("verdict") != "pass":
            return k
    return "other"


def _stage4_dims(rec: dict) -> List[str]:
    """Map a stage-4 reject to failing dimension labels."""
    explicit = rec.get("failing_dimensions")
    if isinstance(explicit, list) and explicit:
        return [str(x) for x in explicit]
    dims: List[str] = []
    for fail in rec.get("failures") or []:
        if not isinstance(fail, dict):
            continue
        if fail.get("dimension"):
            dims.append(str(fail["dimension"]))
            continue
        idx = fail.get("idx")
        try:
            idx_i = int(idx)
        except (TypeError, ValueError):
            dims.append("unknown")
            continue
        dims.append(_V20_STAGE4_IDX_TO_DIM.get(idx_i, f"idx_{idx_i}"))
    if not dims and rec.get("reject_reason") == "unparseable":
        return ["unparseable"]
    return dims or ["unknown"]


def analyze(
    *,
    stage1_path: Path,
    stage4_path: Path,
) -> Dict[str, Any]:
    s1 = read_jsonl(stage1_path)
    s4 = read_jsonl(stage4_path)

    s1_hist: Counter = Counter()
    s1_examples: Dict[str, List[str]] = defaultdict(list)
    for r in s1:
        b = _stage1_bucket(r)
        s1_hist[b] += 1
        if len(s1_examples[b]) < 5:
            s1_examples[b].append(str(r.get("id")))

    s4_hist: Counter = Counter()
    s4_examples: Dict[str, List[str]] = defaultdict(list)
    for r in s4:
        dims = _stage4_dims(r)
        # count each failing dimension once per record (set)
        for d in sorted(set(dims)):
            s4_hist[d] += 1
            if len(s4_examples[d]) < 5:
                s4_examples[d].append(str(r.get("id")))

    return {
        "stage1_rejected_path": str(stage1_path),
        "stage4_rejected_path": str(stage4_path),
        "stage1_n": len(s1),
        "stage4_n": len(s4),
        "stage1_histogram": dict(s1_hist.most_common()),
        "stage1_examples": dict(s1_examples),
        "stage4_histogram_by_dimension": dict(s4_hist.most_common()),
        "stage4_examples": dict(s4_examples),
        "stage4_v20_idx_map": _V20_STAGE4_IDX_TO_DIM,
        "note": (
            "v2.0 stage-4 rejects are bucketed via fixed 8-statement idx map. "
            "v2.1 rejects should carry failures[].dimension / failing_dimensions."
        ),
    }


def _print_table(title: str, hist: dict, examples: dict) -> None:
    print(f"\n=== {title} ===")
    if not hist:
        print("(empty)")
        return
    width = max(len(str(k)) for k in hist) if hist else 10
    print(f"{'bucket':<{width}}  count  examples")
    for k, n in hist.items():
        ex = ", ".join(examples.get(k, [])[:3])
        print(f"{k:<{width}}  {n:5d}  {ex}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1", type=Path, default=None)
    p.add_argument("--stage4", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    v2 = vti_demos_v2_dir()
    s1 = args.stage1 or (v2 / "stage1_rejected.jsonl")
    s4 = args.stage4 or (v2 / "stage4_rejected.jsonl")
    out = args.out or (v2 / "rejection_analysis.json")
    report = analyze(stage1_path=s1, stage4_path=s4)
    write_summary(out, report)
    _print_table("Stage 1 rejects", report["stage1_histogram"], report["stage1_examples"])
    _print_table(
        "Stage 4 rejects (by dimension)",
        report["stage4_histogram_by_dimension"],
        report["stage4_examples"],
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
