#!/usr/bin/env python3
"""Extract demos_850 disjoint-partition textual VTI directions.

Runs check 0.4 (``--check_cache_fidelity``) before any cache write. Qwen2
models require ``--max_pixels 1003520`` to match the existing shared cache.

Example::

    CUDA_VISIBLE_DEVICES=0 python evaluation/run_scripts/extract_demos850_partition_directions.py \\
      --model llava-hf/llava-1.5-7b-hf \\
      --demos_path data/vti/demos_850.jsonl \\
      --partition_path data/vti/demos_850_partition_s42.json \\
      --dimensions all existence attribute counting relation \\
      --num_demos 50 100 200 500 \\
      --check_cache_fidelity
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import create_wrapper, _normalize_model_name  # noqa: E402
from src.paths import (  # noqa: E402
    experiment_artifacts_dir,
    vti_demos_850_partition_path,
    vti_demos_850_path,
)
from evaluation.interventions.vti.directions_partition import (  # noqa: E402
    DIMENSIONS,
    PARTITION_SIZES,
    QWEN_ACT_CACHE_MAX_PIXELS,
    build_or_load_partition,
    check_cache_fidelity,
    check_partition_integrity,
    demos_content_hash,
    extract_partition_grid,
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
        "--dimensions",
        nargs="+",
        default=["all", "existence", "attribute", "counting", "relation"],
        choices=list(DIMENSIONS),
    )
    p.add_argument("--num_demos", nargs="+", type=int, default=list(PARTITION_SIZES))
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="Qwen2/2.5-VL only; must be 1003520 for shared-cache reuse.",
    )
    p.add_argument(
        "--check_cache_fidelity",
        action="store_true",
        help="Run check 0.4 before any write; on fail use isolated act cache.",
    )
    p.add_argument(
        "--act_cache_override",
        type=Path,
        default=None,
        help="Force a non-shared activation cache directory.",
    )
    p.add_argument("--force_recompute", action="store_true")
    return p.parse_args()


def _acquire_extract_lock(model_short: str):
    """Exclusive lock so concurrent/sequential extractors for one model serialize."""
    lock_dir = experiment_artifacts_dir("vti", model_short) / "textual_v2"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".extract_demos850_lock_{model_short}"
    fh = open(lock_path, "a+", encoding="utf-8")
    print(f"  acquiring extract lock {lock_path} ...", flush=True)
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} acquired={datetime.now().isoformat()}\n")
    fh.flush()
    print(f"  extract lock acquired", flush=True)
    return fh


def main() -> None:
    args = parse_args()
    demos_path = Path(args.demos_path)
    partition_path = Path(args.partition_path)
    model_short = _normalize_model_name(args.model)
    is_qwen2 = "qwen2" in args.model.lower()

    if is_qwen2 and args.max_pixels != QWEN_ACT_CACHE_MAX_PIXELS:
        raise SystemExit(
            f"Qwen2 models require --max_pixels {QWEN_ACT_CACHE_MAX_PIXELS} "
            f"(got {args.max_pixels}). A different visual budget needs its own "
            "cache namespace and plan."
        )

    if not demos_path.is_file():
        raise SystemExit(f"Missing demos file: {demos_path}")

    lock_fh = _acquire_extract_lock(model_short)
    try:
        _run_extract(args, demos_path, partition_path, model_short, is_qwen2)
    finally:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        lock_fh.close()


def _run_extract(args, demos_path, partition_path, model_short, is_qwen2) -> None:
    h = demos_content_hash(demos_path)
    print("=== demos_850 disjoint-partition extraction ===")
    print(f"  model      : {args.model} ({model_short})")
    print(f"  demos      : {demos_path}  hash={h}")
    print(f"  partition  : {partition_path}")
    print(f"  dimensions : {args.dimensions}")
    print(f"  num_demos  : {args.num_demos}")
    print(f"  rank       : {args.rank}")
    if args.max_pixels is not None:
        print(f"  max_pixels : {args.max_pixels}")

    part = build_or_load_partition(demos_path, partition_path, seed=args.seed)
    check_partition_integrity(demos_path, part, sizes=args.num_demos)
    print("  partition integrity: OK")

    wrapper_kwargs = {}
    if args.max_pixels is not None and is_qwen2:
        wrapper_kwargs["max_pixels"] = args.max_pixels
    wrapper = create_wrapper(args.model, **wrapper_kwargs).load()
    print(f"  layers={wrapper.num_layers}  hidden_dim={wrapper.hidden_dim}")

    act_override = args.act_cache_override
    fidelity_report = None
    if args.check_cache_fidelity:
        print("\n--- check 0.4 cache fidelity ---")
        fidelity_report = check_cache_fidelity(
            wrapper,
            model_short,
            demos_path,
            act_cache_override=act_override,
        )
        report_path = (
            experiment_artifacts_dir("vti", model_short)
            / "textual_v2"
            / f"demos850_cache_fidelity_{model_short}_2026-07-28.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(fidelity_report, indent=2) + "\n")
        print(f"  fidelity pass={fidelity_report['pass']}  wrote {report_path}")
        if not fidelity_report["pass"]:
            if act_override is None:
                act_override = (
                    experiment_artifacts_dir("vti", model_short)
                    / "textual_v2"
                    / "_act_cache_demos850_2026-07-28"
                )
                print(
                    "  fidelity FAILED — isolating activations under "
                    f"{act_override} (shared _act_cache untouched)"
                )
            else:
                print(f"  fidelity FAILED — using override {act_override}")

    t0 = time.time()
    out_paths = {}
    for dim in args.dimensions:
        print(f"\n--- dimension={dim} ---")
        paths = extract_partition_grid(
            wrapper,
            model_short,
            dimension=dim,
            sizes=args.num_demos,
            demos_path=demos_path,
            partition_path=partition_path,
            rank=args.rank,
            seed=args.seed,
            act_cache_override=act_override,
            max_pixels=args.max_pixels,
            force_recompute=args.force_recompute,
        )
        out_paths[dim] = {str(n): str(p) for n, p in paths.items()}
        for n, p in paths.items():
            print(f"  nd={n} -> {p}")

    wall = time.time() - t0
    wrapper.cleanup()

    run_meta = {
        "model": args.model,
        "model_short": model_short,
        "demos_path": str(demos_path),
        "content_hash_sha256_16": h,
        "partition_path": str(partition_path),
        "dimensions": args.dimensions,
        "num_demos": args.num_demos,
        "max_pixels": args.max_pixels,
        "act_cache_dir": str(act_override) if act_override else "textual_v2/_act_cache",
        "cache_fidelity": fidelity_report,
        "outputs": out_paths,
        "wall_clock_sec": wall,
        "date": datetime.now().strftime("%Y-%m-%d"),
    }
    meta_path = (
        experiment_artifacts_dir("vti", model_short)
        / "textual_v2"
        / f"demos850_partition_run_meta_{model_short}_2026-07-28.json"
    )
    meta_path.write_text(json.dumps(run_meta, indent=2) + "\n")
    print(f"\nDone. wall_clock_sec={wall:.1f}  run_meta={meta_path}")


if __name__ == "__main__":
    main()
