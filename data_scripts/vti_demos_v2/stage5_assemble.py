#!/usr/bin/env python3
"""Stage 5 — assemble demos_v2.jsonl from stage-4 passes."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG.parent))

from vti_demos_v2 import config  # noqa: E402
from vti_demos_v2.io_utils import read_jsonl, today_iso, write_jsonl, write_summary  # noqa: E402
from src.paths import vti_demos_v2_dir, vti_demos_v2_path  # noqa: E402


def _content_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-final", type=int, default=config.N_FINAL)
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--stage0", type=Path, default=None,
                   help="stage0 candidates for rank order")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    v2 = vti_demos_v2_dir()
    inp = args.input or (v2 / "stage4_verdicts.jsonl")
    stage0 = args.stage0 or (v2 / "stage0_candidates.jsonl")
    out = args.out or vti_demos_v2_path()

    rank = {r["id"]: i for i, r in enumerate(read_jsonl(stage0))}
    rows = read_jsonl(inp)
    rows.sort(key=lambda r: (rank.get(r["id"], 10**9), r["id"]))

    if len(rows) < args.n_final:
        shortfall = args.n_final - len(rows)
        print(
            f"Only {len(rows)} stage-4 passes (need {args.n_final}; "
            f"shortfall={shortfall}).\n"
            f"Mine a top-up batch:\n"
            f"  python data_scripts/vti_demos_v2/stage0_mine_candidates.py "
            f"--emit-next-batch --n-candidates {config.TOPUP_BATCH}\n"
            f"Then re-run stages 1–4 on the new candidates and assemble again."
        )
        selected = rows
    else:
        selected = rows[: args.n_final]

    final = []
    for r in selected:
        final.append({
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
        })

    write_jsonl(out, final)
    digest = _content_hash(out)
    write_summary(v2 / "stage5_summary.json", {
        "n_final": len(final),
        "n_available": len(rows),
        "target": args.n_final,
        "out": str(out),
        "content_hash_sha256_16": digest,
        "pipeline_version": config.PIPELINE_VERSION,
    })
    print(f"Wrote {len(final)} records -> {out} (hash={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
