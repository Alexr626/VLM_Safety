#!/usr/bin/env python3
"""CLI entry point for VLM hallucination evaluation.

Example:
    python evaluation/run_eval.py --model llava-hf/llava-1.5-7b-hf \\
        --benchmarks chair --limit 20 --pad_to_square
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import CAPTION_MAX_NEW_TOKENS  # noqa: E402
from evaluation.interventions import ALL_INTERVENTIONS  # noqa: E402
from evaluation.runners import run_evaluation, print_comparison_table  # noqa: E402


_DEFAULT_BENCHMARKS = ["pope", "amber", "chair"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM hallucination evaluation.")
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID (llava-hf/llava-1.5-7b-hf or "
                        "Qwen/Qwen2.5-VL-7B-Instruct).")
    p.add_argument("--interventions", nargs="+",
                   default=list(ALL_INTERVENTIONS))
    p.add_argument("--benchmarks", nargs="+", default=list(_DEFAULT_BENCHMARKS),
                   choices=_DEFAULT_BENCHMARKS)
    p.add_argument("--output_dir", default="evaluation/results")
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="Generation budget for POPE / AMBER.")
    p.add_argument("--chair_max_new_tokens", type=int, default=CAPTION_MAX_NEW_TOKENS,
                   help="Frozen caption length for CHAIR (OPERA protocol: 512). "
                        "Kept constant across conditions because caption length "
                        "confounds CHAIR.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap samples per benchmark (debugging).")
    p.add_argument("--skip_if_exists", action="store_true")
    p.add_argument("--pope_split", default="random",
                   choices=["random", "popular", "adversarial"])
    p.add_argument("--amber_task", default=None,
                   choices=["discriminative", "generative"],
                   help="Restrict AMBER to one task split.")
    p.add_argument("--run_date", default=None,
                   help="Run date subdir (YYYY-MM-DD); defaults to today.")
    p.add_argument("--subset_ids_file", default=None,
                   help="JSON of pinned sample ids to score (overrides --limit). "
                        "Either {benchmark: [ids]} or a flat [ids] list.")
    p.add_argument("--chair_prompt", default=None,
                   help="Verbatim CHAIR caption prompt; replaces the stored prompt.")
    p.add_argument("--max_pixels", type=int, default=None,
                   help="Qwen2.5-VL vision pixel budget. Ignored for LLaVA.")
    p.add_argument("--pad_to_square", action="store_true",
                   help="LLaVA: pad images to square (original LLaVA preprocessing) "
                        "instead of HF centre crop. Ignored for Qwen.")
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
        run_date=args.run_date,
        chair_max_new_tokens=args.chair_max_new_tokens,
        subset_ids_file=args.subset_ids_file,
        chair_prompt=args.chair_prompt,
        max_pixels=args.max_pixels,
        pad_to_square=args.pad_to_square,
    )
    print_comparison_table(out["model_short"], args.output_dir, run_date=out["run_date"])


if __name__ == "__main__":
    main()
