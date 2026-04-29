#!/usr/bin/env python3
"""
Recompute MSSBench eval-only ASR from existing responses
=========================================================
Walks `evaluation/results/{model}/mssbench/*/responses.json`, filters each
to records whose id is in `data/mssbench/train_eval_split.json["eval_sample_ids"]`,
and writes `asr_summary_eval.json` next to `responses.json`.

This is the post-hoc path for fair comparison: vanilla and adashield_s runs
that generated responses on the FULL MSSBench (1200 samples) get an eval-only
summary derived from those existing responses, so they can be compared head-
to-head with comp_safety_shift_mssbench_vl (which trains on the same 75%
record-level split that's held out from this 25% eval subset). No model load
or generation involved — the operation is a pure filter on responses.json.

Usage
-----
    # Single model
    python evaluation/scripts/recompute_mssbench_eval_asr.py \
        --model llava-hf/llava-1.5-7b-hf

    # Every model with results under evaluation/results/
    python evaluation/scripts/recompute_mssbench_eval_asr.py --all_models

When `responses.json` covers fewer samples than the full benchmark (e.g. it
was generated with --mssbench_eval_only), the eval summary still gets
written; it'll just be ~identical to asr_summary.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import _normalize_model_name  # noqa: E402
from evaluation.runners.eval_runner import (  # noqa: E402
    write_eval_only_asr_summary, _load_mssbench_eval_ids, _load_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", default=None,
                   help="HuggingFace model ID (e.g. llava-hf/llava-1.5-7b-hf).")
    g.add_argument("--all_models", action="store_true",
                   help="Walk every {model_short}/ under --results_root.")
    p.add_argument("--results_root", default="evaluation/results",
                   help="Root directory containing per-model result trees.")
    return p.parse_args()


def _process_model(results_root: Path, model_short: str, eval_ids: set) -> int:
    """Process every intervention dir under {results_root}/{model_short}/mssbench/.
    Returns the number of asr_summary_eval.json files written."""
    mssbench_dir = results_root / model_short / "mssbench"
    if not mssbench_dir.exists():
        print(f"  [{model_short}] no mssbench/ dir under {results_root}; skipping")
        return 0

    n_written = 0
    for iv_dir in sorted(p for p in mssbench_dir.iterdir() if p.is_dir()):
        responses_path = iv_dir / "responses.json"
        if not responses_path.exists():
            continue
        base_summary = None
        full_summary_path = iv_dir / "asr_summary.json"
        if full_summary_path.exists():
            try:
                base_summary = _load_json(full_summary_path)
            except Exception:
                pass
        result = write_eval_only_asr_summary(iv_dir, eval_ids, base_summary)
        if result is None:
            continue
        asr = result.get("asr_overall")
        n_in = result.get("n_responses_in_eval", 0)
        n_total = result.get("n_responses_total", 0)
        asr_str = f"{asr * 100:.1f}%" if asr is not None else "--"
        print(f"  [{model_short}] {iv_dir.name:40s}  "
              f"asr_eval={asr_str}  "
              f"({n_in} of {n_total} responses kept)")
        n_written += 1
    return n_written


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = _PROJECT_ROOT / results_root

    eval_ids = _load_mssbench_eval_ids(_PROJECT_ROOT)
    if eval_ids is None:
        print("ERROR: data/mssbench/train_eval_split.json not found.\n"
              "Run: python -m src.dataset --mssbench_split")
        sys.exit(1)
    print(f"Loaded {len(eval_ids)} eval-split sample ids "
          f"from data/mssbench/train_eval_split.json")

    if args.all_models:
        if not results_root.exists():
            print(f"ERROR: {results_root} not found.")
            sys.exit(1)
        model_shorts = sorted(p.name for p in results_root.iterdir() if p.is_dir())
        print(f"Walking {len(model_shorts)} model dirs under {results_root}")
    else:
        model_shorts = [_normalize_model_name(args.model)]

    total = 0
    for ms in model_shorts:
        n = _process_model(results_root, ms, eval_ids)
        total += n
    print(f"\nWrote {total} asr_summary_eval.json file(s).")


if __name__ == "__main__":
    main()
