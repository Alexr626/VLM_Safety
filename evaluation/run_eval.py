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


def _parse_layer_set(raw: str | None):
    """Return (layer_indices, layer_set_label).

    ``all`` → (None, ``\"all\"``); ``A-B`` inclusive → (range list, ``\"A_B\"``).
    """
    if raw is None:
        return None, None
    s = raw.strip().lower()
    if s == "all":
        return None, "all"
    if "-" not in s:
        raise argparse.ArgumentTypeError(
            f"--layer_set must be 'all' or inclusive 'A-B'; got {raw!r}"
        )
    a_str, b_str = s.split("-", 1)
    try:
        a, b = int(a_str), int(b_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--layer_set must be 'all' or inclusive 'A-B'; got {raw!r}"
        ) from e
    if b < a:
        raise argparse.ArgumentTypeError(
            f"--layer_set range must satisfy A <= B; got {raw!r}"
        )
    return list(range(a, b + 1)), f"{a}_{b}"


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
    p.add_argument("--run_date", default=None,
                   help="Run date subdir (YYYY-MM-DD); defaults to today.")
    p.add_argument("--beta", type=float, default=None,
                   help="Textual steering coefficient for VTI interventions. "
                        "When set, results are written under '{iv}__b{beta}'.")
    p.add_argument("--alpha", type=float, default=None,
                   help="Vision steering coefficient for vti_visual_* interventions. "
                        "When set, results are written under '{iv}__a{alpha}'.")
    p.add_argument("--amber_task", default=None,
                   choices=["discriminative", "generative"],
                   help="Restrict AMBER to one task split.")
    p.add_argument("--judge", default="mock",
                   help="MMHal-Bench judge: mock (default, offline) | "
                        "openai[:model] | anthropic[:model] | gemini[:model]. "
                        "Only used for the mmhal_bench benchmark.")
    p.add_argument("--chair_max_new_tokens", type=int, default=256,
                   help="Frozen caption length for CHAIR generation (default 256). "
                        "Kept constant across baseline and interventions because "
                        "caption length confounds CHAIR; independent of "
                        "--max_new_tokens (used by other benchmarks).")
    p.add_argument("--subset_ids_file", default=None,
                   help="JSON of pinned sample ids to score (overrides --limit). "
                        "Either {benchmark: [ids]} or a flat [ids] list applied "
                        "to every benchmark in the run.")
    p.add_argument("--chair_prompt", default=None,
                   help="Verbatim CHAIR caption prompt; replaces the stored "
                        "prompt for every CHAIR sample (e.g. the exact VTI "
                        "prompt 'Please Describe this image in detail.').")
    p.add_argument("--demos_path", default=None,
                   help="Paired-caption demos JSONL for textual VTI directions. "
                        "Use data/vti/demos_v2.jsonl with --vector_dimension.")
    p.add_argument("--vector_dimension", default=None,
                   choices=["existence", "attribute", "counting", "relation", "all"],
                   help="demos_v2 h_values dimension for textual VTI extraction.")
    p.add_argument("--num_demos", type=int, default=None,
                   help="Number of demos for textual VTI direction PCA.")
    p.add_argument("--rank", type=int, default=None,
                   help="PCA rank for textual VTI (demos_v2 caches rank=2; "
                        "steering uses PC1+mean live recon).")
    p.add_argument("--max_pixels", type=int, default=None,
                   help="Qwen2/2.5-VL vision pixel budget (e.g. 1003520). "
                        "Ignored for non-Qwen2 models.")
    p.add_argument("--directions_dir", default=None,
                   help="Precomputed textual_v2-format direction directory "
                        "(directions.npz + metadata.json). Takes precedence "
                        "over demos_v2 / legacy extraction.")
    p.add_argument("--layer_set", default=None,
                   help="Decoder layers to steer: 'all' or inclusive 'A-B' "
                        "(e.g. 5-14). Absolute layer indices.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    layer_indices, layer_set_label = _parse_layer_set(args.layer_set)
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
        beta=args.beta,
        alpha=args.alpha,
        judge=args.judge,
        chair_max_new_tokens=args.chair_max_new_tokens,
        subset_ids_file=args.subset_ids_file,
        chair_prompt=args.chair_prompt,
        demos_path=args.demos_path,
        vector_dimension=args.vector_dimension,
        num_demos=args.num_demos,
        rank=args.rank,
        max_pixels=args.max_pixels,
        directions_dir=args.directions_dir,
        layer_indices=layer_indices,
        layer_set_label=layer_set_label,
    )
    print_comparison_table(out["model_short"], args.output_dir, run_date=out["run_date"])


if __name__ == "__main__":
    main()
