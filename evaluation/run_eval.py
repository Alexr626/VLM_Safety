#!/usr/bin/env python3
"""CLI entry point for VLM hallucination evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.interventions import ALL_INTERVENTIONS  # noqa: E402
from evaluation.runners import run_evaluation, print_comparison_table  # noqa: E402


_DEFAULT_BENCHMARKS = ["pope", "amber", "chair", "hallusionbench", "mmhal_bench"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM hallucination evaluation.")
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID (e.g. llava-hf/llava-1.5-7b-hf).")
    p.add_argument("--interventions", nargs="+",
                   default=list(ALL_INTERVENTIONS))
    p.add_argument("--benchmarks", nargs="+", default=list(_DEFAULT_BENCHMARKS))
    p.add_argument("--output_dir", default="evaluation/results")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap samples per benchmark (debugging).")
    p.add_argument("--skip_if_exists", action="store_true")
    p.add_argument("--pope_split", default="random",
                   help="POPE split name (random, popular, adversarial).")
    p.add_argument("--amber_task", default=None,
                   choices=["discriminative", "generative"],
                   help="Restrict AMBER to one task split.")
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
        pope_split=args.pope_split,
        amber_task=args.amber_task,
    )
    print_comparison_table(out["model_short"], args.output_dir)


if __name__ == "__main__":
    main()
