#!/usr/bin/env python3
"""Assemble data/vti/demos_850.jsonl from demos_v2 + new stage-4 passes.

The first 555 lines are a byte-identical copy of demos_v2.jsonl. New rows are
appended from stage-4 passes whose ids are not in the base file, ranked by
top-up stage0 order (same convention as stage5_assemble.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG.parent))

from vti_demos_v2 import config  # noqa: E402
from vti_demos_v2.io_utils import read_jsonl, today_iso, write_summary  # noqa: E402
from src.paths import (  # noqa: E402
    vti_demos_850_partition_path,
    vti_demos_850_path,
    vti_demos_v2_dir,
    vti_demos_v2_path,
)

DIMENSIONS = ("existence", "attribute", "counting", "relation", "all")


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _admissible(row: dict) -> bool:
    if not (row.get("value") and str(row["value"]).strip()):
        return False
    hv = row.get("h_values") or {}
    for d in DIMENSIONS:
        if not (hv.get(d) and str(hv[d]).strip()):
            return False
    return True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=None)
    p.add_argument("--stage4", type=Path, default=None)
    p.add_argument(
        "--stage0-topup",
        type=Path,
        nargs="+",
        default=None,
        help="Top-up stage0 candidate file(s) for deterministic rank order",
    )
    p.add_argument("--n-new", type=int, default=295)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--summary-out", type=Path, default=None)
    p.add_argument(
        "--partition-out",
        type=Path,
        default=None,
        help="Also write demos_850_partition_s42.json (default path if flag alone)",
    )
    p.add_argument(
        "--write-partition",
        action="store_true",
        help="Build disjoint 50/100/200/500 partition after assembly",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    v2 = vti_demos_v2_dir()
    base = args.base or vti_demos_v2_path()
    stage4 = args.stage4 or (v2 / "stage4_verdicts.jsonl")
    stage0_paths = args.stage0_topup or [
        v2 / "stage0_candidates_topup_2026-07-28.jsonl",
    ]
    out = args.out or vti_demos_850_path()
    summary_out = args.summary_out or (v2 / "demos_850_assembly_summary.json")

    if not base.is_file():
        raise SystemExit(f"Base demos missing: {base}")
    if out.resolve() == base.resolve():
        raise SystemExit(
            "Refusing to write demos_850 onto demos_v2.jsonl — out must be a "
            "separate file."
        )

    base_bytes = base.read_bytes()
    if not base_bytes.endswith(b"\n"):
        raise SystemExit(f"Base file does not end with newline: {base}")

    base_rows = read_jsonl(base)
    base_ids = {r["id"] for r in base_rows}
    if len(base_rows) != 555 or len(base_ids) != 555:
        raise SystemExit(
            f"Base expected 555 unique rows; got n={len(base_rows)} "
            f"unique={len(base_ids)}"
        )

    rank: dict[str, int] = {}
    idx = 0
    for path in stage0_paths:
        if not path.is_file():
            raise SystemExit(f"Missing stage0 top-up file: {path}")
        for r in read_jsonl(path):
            rid = r["id"]
            if rid not in rank:
                rank[rid] = idx
                idx += 1

    candidates = []
    for r in read_jsonl(stage4):
        if r["id"] in base_ids:
            continue
        if not _admissible(r):
            continue
        candidates.append(r)
    candidates.sort(key=lambda r: (rank.get(r["id"], 10**9), r["id"]))

    if len(candidates) < args.n_new:
        raise SystemExit(
            f"Only {len(candidates)} admissible new stage-4 rows "
            f"(need {args.n_new}). Stop — do not reduce block sizes here."
        )
    selected = candidates[: args.n_new]

    out.parent.mkdir(parents=True, exist_ok=True)
    # Verbatim base bytes, then append new finals.
    with open(out, "wb") as f:
        f.write(base_bytes)

    new_ids = []
    with open(out, "a", encoding="utf-8") as f:
        for r in selected:
            rec = {
                "id": r["id"],
                "image": r["image"],
                "question": config.QUESTION,
                "value": r["value"],
                "h_values": r["h_values"],
                "anchors": r["anchors"],
                "provenance": {
                    "stage1_model": r.get("stage1_model"),
                    "stage2_model": r.get("stage2_model"),
                    "stage3_model": r.get("stage3_model"),
                    "stage4_model": r.get("stage4_model"),
                    "pipeline_version": config.PIPELINE_VERSION,
                    "date": today_iso(),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            new_ids.append(r["id"])

    out_bytes = out.read_bytes()
    if out_bytes[: len(base_bytes)] != base_bytes:
        raise SystemExit("Lines 1–555 of output are not byte-identical to base")

    digest = _content_hash(out)
    dim_counts = {
        d: sum(
            1
            for r in (base_rows + selected)
            if (r.get("h_values") or {}).get(d)
            and str((r.get("h_values") or {}).get(d)).strip()
        )
        for d in DIMENSIONS
    }
    # selected rows use stage4 schema (h_values present); base_rows too.
    summary = {
        "n_base": len(base_rows),
        "n_new": len(selected),
        "n_total": len(base_rows) + len(selected),
        "n_admissible_candidates": len(candidates),
        "new_ids": new_ids,
        "out": str(out),
        "content_hash_sha256_16": digest,
        "base_content_hash_sha256_16": _content_hash(base),
        "base_bytes_identical_prefix": True,
        "pipeline_version": config.PIPELINE_VERSION,
        "stage0_topup": [str(p) for p in stage0_paths],
        "per_dimension_caption_presence": dim_counts,
        "date": today_iso(),
    }
    write_summary(summary_out, summary)
    print(
        f"Wrote {len(base_rows) + len(selected)} records -> {out} "
        f"(hash={digest}, n_new={len(selected)})"
    )

    if args.write_partition or args.partition_out is not None:
        # Late import: directions_partition lives under evaluation/.
        sys.path.insert(0, str(_ROOT))
        from evaluation.interventions.vti.directions_partition import (  # noqa: E402
            build_or_load_partition,
            check_partition_integrity,
        )
        part_path = args.partition_out or vti_demos_850_partition_path()
        part = build_or_load_partition(out, part_path)
        check_partition_integrity(out, part)
        print(f"Wrote partition -> {part_path}")
        summary["partition_path"] = str(part_path)
        summary["partition_content_hash"] = part["_meta"]["content_hash_sha256_16"]
        write_summary(summary_out, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
