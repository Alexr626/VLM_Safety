#!/usr/bin/env python3
"""
List Top-X% Most-Similar MSSBench Stems (DINOv2)
================================================
Print the top X% of MSSBench `chat`-split stems with the highest DINOv2
cosine similarity between their SSS and SSU image variants. Resolves each
`rec_idx` back to its row in `data/mssbench/combined.json` so you can see
the intent / queries / image paths behind the numeric ids printed by
`causal_mediation_mssbench.py` (e.g. `rec0277_q0`).

Background:
    `combined.json["chat"]` has no `rec_idx` field — `rec_idx` is the
    positional index into that list. The DINOv2 scorer
    (`compute_image_similarity.py`) and the causal-mediation runner both
    assume that mapping; this helper just inverts it.

Tier definition (matches `causal_mediation_mssbench._select_stems_by_tier`):
    Sort train-split stems by ascending cosine similarity, then take the
    last K = round(N * pct/100) (most similar). Pass `--all_stems` to
    rank across the full 300, not just the train split.

Output:
    Pretty-printed table to stdout. Pass `--json <path>` to also dump the
    selected rows as JSON (the same fields shown in the table).

Usage
-----
    # Top 10% of train-split stems
    python helper_scripts/list_top_similar_mssbench_stems.py --pct 10

    # Top 33% across all 300 chat stems, dump as JSON
    python helper_scripts/list_top_similar_mssbench_stems.py \\
        --pct 33 --all_stems --json /tmp/top33.json

    # Bottom 10% (least-similar; same selection rule as `--tier bottom`)
    python helper_scripts/list_top_similar_mssbench_stems.py --pct 10 --bottom
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SCORES = _PROJECT_ROOT / "data" / "mssbench" / "image_similarity" / "dinov2_similarity_scores.json"
_DEFAULT_COMBINED = _PROJECT_ROOT / "data" / "mssbench" / "combined.json"


def _select(stems: List[dict], pct: float, bottom: bool) -> List[dict]:
    sorted_stems = sorted(stems, key=lambda s: s["cosine_similarity"])
    n = len(sorted_stems)
    k = max(1, int(round(n * pct / 100.0)))
    chosen = sorted_stems[:k] if bottom else sorted_stems[-k:]
    chosen.sort(key=lambda s: s["cosine_similarity"], reverse=not bottom)
    return chosen


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pct", type=float, default=10.0,
                    help="Tier percentage (default: 10).")
    ap.add_argument("--bottom", action="store_true",
                    help="Select bottom-X%% (least-similar) instead of top.")
    ap.add_argument("--all_stems", action="store_true",
                    help="Rank across all 300 chat stems instead of train-split only.")
    ap.add_argument("--scores", type=Path, default=_DEFAULT_SCORES,
                    help=f"DINOv2 scores JSON (default: {_DEFAULT_SCORES.relative_to(_PROJECT_ROOT)}).")
    ap.add_argument("--combined", type=Path, default=_DEFAULT_COMBINED,
                    help=f"MSSBench combined.json (default: {_DEFAULT_COMBINED.relative_to(_PROJECT_ROOT)}).")
    ap.add_argument("--json", dest="json_out", type=Path, default=None,
                    help="Optional path to dump the selected rows as JSON.")
    ap.add_argument("--query_chars", type=int, default=80,
                    help="Truncate each query to this many characters in the table (default: 80).")
    args = ap.parse_args()

    scores_data = json.loads(args.scores.read_text())
    combined = json.loads(args.combined.read_text())
    chat_records = combined["chat"]

    stems = scores_data["stems"]
    if not args.all_stems:
        stems = [s for s in stems if s.get("in_train_split")]
    pool_label = "all chat stems" if args.all_stems else "train-split stems"
    chosen = _select(stems, args.pct, args.bottom)

    tier_word = "bottom" if args.bottom else "top"
    print(f"\n{tier_word.title()} {args.pct:g}% of {len(stems)} {pool_label} "
          f"→ {len(chosen)} stems (DINOv2 cosine similarity).\n")

    rows = []
    for rank, stem in enumerate(chosen, 1):
        rec_idx = stem["rec_idx"]
        rec = chat_records[rec_idx] if rec_idx < len(chat_records) else {}
        queries = rec.get("queries", []) or []
        rows.append({
            "rank": rank,
            "rec_label": f"rec{rec_idx:04d}",
            "rec_idx": rec_idx,
            "cosine_similarity": stem["cosine_similarity"],
            "type": stem.get("type"),
            "in_train_split": stem.get("in_train_split"),
            "safe_image_path": stem.get("safe_image_path"),
            "unsafe_image_path": stem.get("unsafe_image_path"),
            "intent": rec.get("intent", ""),
            "unsafe_image_caption": rec.get("unsafe_image", ""),
            "queries": queries,
        })

    header = f"{'#':>3}  {'rec':>8}  {'sim':>6}  {'type':<10}  {'safe→unsafe':<24}  intent / queries"
    print(header)
    print("-" * len(header))
    for r in rows:
        safe = Path(r["safe_image_path"]).name if r["safe_image_path"] else "?"
        unsafe = Path(r["unsafe_image_path"]).name if r["unsafe_image_path"] else "?"
        pair_str = f"{safe}→{unsafe}"
        intent = _truncate(r["intent"], args.query_chars)
        print(f"{r['rank']:>3}  {r['rec_label']:>8}  {r['cosine_similarity']:>6.3f}  "
              f"{(r['type'] or ''):<10}  {pair_str:<24}  {intent}")
        for qi, q in enumerate(r["queries"]):
            print(f"{'':>3}  {'':>8}  {'':>6}  {'':<10}  {'':<24}    q{qi}: {_truncate(q, args.query_chars)}")
    print()

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "tier": tier_word,
            "pct": args.pct,
            "all_stems": args.all_stems,
            "n_pool": len(stems),
            "n_chosen": len(chosen),
            "rows": rows,
        }, indent=2))
        print(f"Wrote {len(rows)} rows → {args.json_out}")


if __name__ == "__main__":
    main()
