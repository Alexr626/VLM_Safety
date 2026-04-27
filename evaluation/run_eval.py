#!/usr/bin/env python3
"""
CLI entry point for the VLM jailbreak-defense evaluation framework.

Usage
-----
    python evaluation/run_eval.py \
        --model llava-hf/llava-1.5-7b-hf \
        --interventions vanilla comp_safety_shift adashield_s \
        --benchmarks mm_safetybench figstep mssbench \
        --output_dir evaluation/results \
        --max_new_tokens 256 \
        --skip_if_exists
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.interventions import ALL_INTERVENTIONS  # noqa: E402
from evaluation.runners import run_evaluation, print_comparison_table  # noqa: E402


_DEFAULT_BENCHMARKS = ["mm_safetybench", "figstep", "mssbench"]


def _split_csv_int(s: str | None) -> list[int] | None:
    if s is None or not s.strip():
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def _split_csv(s: str | None) -> list[str] | None:
    if s is None or not s.strip():
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM jailbreak-defense evaluation.")
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID (e.g. llava-hf/llava-1.5-7b-hf).")
    p.add_argument("--interventions", nargs="+",
                   default=list(ALL_INTERVENTIONS),
                   help="Subset of interventions to run.")
    p.add_argument("--benchmarks", nargs="+", default=list(_DEFAULT_BENCHMARKS),
                   help="Subset of benchmarks to run.")
    p.add_argument("--output_dir", default="evaluation/results")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of samples per benchmark (for debugging).")
    p.add_argument("--skip_if_exists", action="store_true",
                   help="Skip benchmark×intervention combos with existing outputs.")
    p.add_argument("--mm_scenarios", default=None,
                   help="Comma-separated MM-SafetyBench scenario IDs, e.g. 1,2,3.")
    p.add_argument("--mm_image_types", default=None,
                   help="Comma-separated MM-SafetyBench image types, "
                        "e.g. SD,OCR,SD_TYPO.")
    p.add_argument("--mssbench_labels", default=None,
                   help="Comma-separated MSSBench safety labels, e.g. SSU.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_evaluation(
        model_id=args.model,
        interventions=args.interventions,
        benchmarks=args.benchmarks,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        skip_if_exists=args.skip_if_exists,
        mm_safetybench_scenarios=_split_csv_int(args.mm_scenarios),
        mm_safetybench_image_types=_split_csv(args.mm_image_types),
        mssbench_safety_labels=_split_csv(args.mssbench_labels),
    )
    print_comparison_table(out["model_short"], args.output_dir)


if __name__ == "__main__":
    main()
