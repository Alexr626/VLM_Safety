#!/usr/bin/env python3
"""Extract demos_850 shuffled-image control directions (all / nd∈{50,100,200,500}).

Same ids, order, and captions as each deployed
``demos850_{h8}_all_nd{N}_s42_r2_partition`` cell; images deranged within the
block (seed 1234). Writes under ``shuffled_control_demos850/`` with gating
sanity reports. No cosine / behavioral comparison.

Example::

    CUDA_VISIBLE_DEVICES=0 python evaluation/run_scripts/extract_demos850_shuffled_control_directions.py \\
      --model llava-hf/llava-1.5-7b-hf

    CUDA_VISIBLE_DEVICES=0 python evaluation/run_scripts/extract_demos850_shuffled_control_directions.py \\
      --model Qwen/Qwen2.5-VL-7B-Instruct --max_pixels 1003520
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import create_wrapper, _normalize_model_name  # noqa: E402
from src.paths import (  # noqa: E402
    vti_demos_850_partition_path,
    vti_demos_850_path,
)
from evaluation.interventions.vti.directions_partition import (  # noqa: E402
    PARTITION_SIZES,
    QWEN_ACT_CACHE_MAX_PIXELS,
    build_or_load_partition,
    check_partition_integrity,
    demos_content_hash,
)
from evaluation.interventions.vti.shuffled_control_partition import (  # noqa: E402
    DERANGEMENT_SEED,
    EXPECTED_DEMOS_HASH_PREFIX,
    extract_shuffled_control_partition_direction,
    write_sanity_report,
    write_sanity_rollup,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True)
    p.add_argument("--demos_path", default=str(vti_demos_850_path()))
    p.add_argument("--partition_path", default=str(vti_demos_850_partition_path()))
    p.add_argument(
        "--num_demos",
        nargs="+",
        type=int,
        default=list(PARTITION_SIZES),
    )
    p.add_argument("--derangement_seed", type=int, default=DERANGEMENT_SEED)
    p.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="Qwen2/2.5-VL only; must be 1003520.",
    )
    p.add_argument("--force_recompute", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_short = _normalize_model_name(args.model)
    demos_path = Path(args.demos_path)
    partition_path = Path(args.partition_path)
    is_qwen2 = "qwen2" in args.model.lower()

    if is_qwen2 and args.max_pixels != QWEN_ACT_CACHE_MAX_PIXELS:
        raise SystemExit(
            f"Qwen2 models require --max_pixels {QWEN_ACT_CACHE_MAX_PIXELS} "
            f"(got {args.max_pixels})"
        )
    if not demos_path.is_file():
        raise SystemExit(f"Missing demos file: {demos_path}")

    demos_hash = demos_content_hash(demos_path)
    if demos_hash != EXPECTED_DEMOS_HASH_PREFIX:
        raise SystemExit(
            f"demos_850 hash {demos_hash} != expected {EXPECTED_DEMOS_HASH_PREFIX}"
        )

    part = build_or_load_partition(demos_path, partition_path)
    check_partition_integrity(demos_path, part, sizes=args.num_demos)

    print("=== demos_850 shuffled-control extraction ===")
    print(f"  model           : {args.model} ({model_short})")
    print(f"  demos           : {demos_path}  hash={demos_hash}")
    print(f"  partition       : {partition_path}")
    print(f"  num_demos       : {args.num_demos}")
    print(f"  derangement_seed: {args.derangement_seed}")
    if args.max_pixels is not None:
        print(f"  max_pixels      : {args.max_pixels}")

    wrapper_kwargs = {}
    if args.max_pixels is not None and is_qwen2:
        wrapper_kwargs["max_pixels"] = args.max_pixels
    wrapper = create_wrapper(args.model, **wrapper_kwargs).load()
    print(f"  layers={wrapper.num_layers}  hidden_dim={wrapper.hidden_dim}")

    cell_results = []
    any_fail = False
    for n in args.num_demos:
        print(f"\n--- all / nd={n} ---")
        directions, meta, checks = extract_shuffled_control_partition_direction(
            wrapper,
            model_short,
            num_demos=n,
            demos_path=demos_path,
            partition_path=partition_path,
            derangement_seed=args.derangement_seed,
            max_pixels=args.max_pixels,
            force_recompute=args.force_recompute,
        )
        # lineage: control_for must equal computed slug from hash
        checks["lineage_slug_ok"] = (
            meta.get("control_for_deployed_slug")
            == f"demos850_{demos_hash[:8]}_all_nd{n}_s42_r2_partition"
        )
        report = write_sanity_report(model_short, n, checks, meta)
        gate = bool(checks.get("all_gating_pass"))
        print(
            f"  shape={tuple(directions.shape)}  "
            f"forwards={checks.get('forwards_executed')}  "
            f"hits={checks.get('cache_hits')}  "
            f"gating={'PASS' if gate else 'FAIL'}  "
            f"report={report}"
        )
        if not gate and not checks.get("skipped_extraction"):
            any_fail = True
        cell_results.append({
            "num_demos": n,
            "all_gating_pass": gate,
            "forwards_executed": checks.get("forwards_executed"),
            "cache_hits": checks.get("cache_hits"),
            "control_for_deployed_slug": meta.get("control_for_deployed_slug"),
        })

    rollup = write_sanity_rollup(model_short, cell_results)
    print(f"\nRollup: {rollup}")
    wrapper.cleanup()
    if any_fail:
        print("One or more cells failed gating.", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
