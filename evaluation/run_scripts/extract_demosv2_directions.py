#!/usr/bin/env python3
"""Extract demos_v2 textual VTI directions (live PCA path + shuffled_prefix).

Ordering (per plan): extract ``all`` first (value + h_values.all over the
500-prefix), then remaining dimensions. Nested N∈{50,100,200,500} PCA runs
reuse the activation cache — no re-forward per N.

Example::

    CUDA_VISIBLE_DEVICES=0 python evaluation/run_scripts/extract_demosv2_directions.py \\
      --model llava-hf/llava-1.5-7b-hf --dimensions all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import create_wrapper, _normalize_model_name  # noqa: E402
from src.paths import vti_demos_v2_path  # noqa: E402
from evaluation.interventions.vti.directions_v2 import (  # noqa: E402
    DIMENSIONS,
    NUM_DEMOS_GRID,
    demos_content_hash,
    extract_dimension_grid,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--demos_path", default=str(vti_demos_v2_path()))
    p.add_argument("--dimensions", nargs="+", default=["all"],
                   choices=list(DIMENSIONS))
    p.add_argument("--num_demos", nargs="+", type=int, default=list(NUM_DEMOS_GRID))
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_pixels", type=int, default=None,
                   help="Qwen2/2.5-VL only (e.g. 1003520).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    demos_path = Path(args.demos_path)
    h = demos_content_hash(demos_path)
    model_short = _normalize_model_name(args.model)
    print(f"=== demos_v2 textual extraction ===")
    print(f"  model      : {args.model} ({model_short})")
    print(f"  demos      : {demos_path}  hash={h}")
    print(f"  dimensions : {args.dimensions}")
    print(f"  num_demos  : {args.num_demos}")
    print(f"  rank       : {args.rank}")

    wrapper_kwargs = {}
    if args.max_pixels is not None and "qwen2" in args.model.lower():
        wrapper_kwargs["max_pixels"] = args.max_pixels
        print(f"  max_pixels : {args.max_pixels}")
    wrapper = create_wrapper(args.model, **wrapper_kwargs).load()
    print(f"  layers={wrapper.num_layers}  hidden_dim={wrapper.hidden_dim}")

    for dim in args.dimensions:
        print(f"\n--- dimension={dim} ---")
        paths = extract_dimension_grid(
            wrapper,
            model_short,
            dimension=dim,
            num_demos_list=args.num_demos,
            rank=args.rank,
            seed=args.seed,
            demos_path=demos_path,
        )
        for n, p in paths.items():
            print(f"  nd={n} -> {p}")

    wrapper.cleanup()
    print("\nDone.")


if __name__ == "__main__":
    main()
