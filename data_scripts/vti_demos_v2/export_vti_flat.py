#!/usr/bin/env python3
"""Export one demos_v2 dimension to the flat VTI schema (value / h_value)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG.parent))

from vti_demos_v2.io_utils import read_jsonl, write_jsonl  # noqa: E402
from src.paths import vti_data_dir, vti_demos_v2_path  # noqa: E402

DIMENSIONS = ("existence", "attribute", "counting", "relation", "all")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dimension", required=True, choices=DIMENSIONS)
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--subtype", default=None,
                   help="optional selected anchor subtype, e.g. vertical or at_most")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp = args.input or vti_demos_v2_path()
    out = args.out or (vti_data_dir() / f"demos_v2_{args.dimension}.jsonl")
    rows = read_jsonl(inp)
    if not rows:
        raise SystemExit(f"No records in {inp}. Run stage5_assemble.py first.")

    flat = []
    for r in rows:
        if args.subtype:
            anchor = (r.get("anchors") or {}).get(args.dimension, {})
            subtype = anchor.get("type") or anchor.get("mode")
            if subtype != args.subtype:
                continue
        hv = (r.get("h_values") or {}).get(args.dimension)
        if not hv:
            raise SystemExit(f"Record {r.get('id')} missing h_values.{args.dimension}")
        flat.append({
            "id": r["id"],
            "image": r["image"],
            "question": r.get("question", "Describe this image in detail."),
            "value": r["value"],
            "h_value": hv,
        })
    write_jsonl(out, flat)
    print(f"Wrote {len(flat)} flat demos ({args.dimension}) -> {out}")
    print(
        "Consumable by VTITextualIntervention(demos_path=...) with no loader change."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
