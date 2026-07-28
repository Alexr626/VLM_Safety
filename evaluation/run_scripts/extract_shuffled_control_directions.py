#!/usr/bin/env python3
"""Build and verify the shuffled-control demos_v2 ``all``/nd200 direction.

Same 200 demo ids / captions / order as the deployed
``demosv2_9a44f4af_all_nd200_s42_r2_prefix`` direction; images swapped via a
seed-1234 derangement. Writes directions under
``experiment_artifacts/vti/{model_short}/shuffled_control/all_nd200/`` and a
sanity report. No cosine / behavioral comparison.

Example::

    CUDA_VISIBLE_DEVICES=0 python evaluation/run_scripts/extract_shuffled_control_directions.py \\
      --model llava-hf/llava-1.5-7b-hf

    CUDA_VISIBLE_DEVICES=0 python evaluation/run_scripts/extract_shuffled_control_directions.py \\
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
from src.paths import vti_demos_v2_path  # noqa: E402
from evaluation.interventions.vti.directions_v2 import (  # noqa: E402
    load_textual_v2_directions,
)
from evaluation.interventions.vti.shuffled_control import (  # noqa: E402
    DEPLOYED_SLUG,
    DERANGEMENT_SEED,
    deployed_direction_dir,
    extract_shuffled_control_direction,
    load_or_write_derangement,
    write_sanity_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True)
    p.add_argument("--demos_path", default=str(vti_demos_v2_path()))
    p.add_argument("--deployed_slug", default=DEPLOYED_SLUG)
    p.add_argument("--derangement_seed", type=int, default=DERANGEMENT_SEED)
    p.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="Qwen2/2.5-VL only; use 1003520 to match deployed demos_v2 extract.",
    )
    p.add_argument(
        "--force_recompute",
        action="store_true",
        help="Wipe shuffled act cache and rebuild direction.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_short = _normalize_model_name(args.model)
    demos_path = Path(args.demos_path)

    deployed_dir = deployed_direction_dir(model_short, args.deployed_slug)
    if not (deployed_dir / "metadata.json").exists():
        raise SystemExit(f"Deployed metadata missing: {deployed_dir}")
    _, deployed_meta = load_textual_v2_directions(deployed_dir)
    ids_used = list(deployed_meta["ids_used"])

    print("=== shuffled-control direction extraction ===")
    print(f"  model           : {args.model} ({model_short})")
    print(f"  control_for     : {args.deployed_slug}")
    print(f"  demos           : {demos_path}")
    print(f"  n_ids           : {len(ids_used)}")
    print(f"  derangement_seed: {args.derangement_seed}")

    derangement, der_path = load_or_write_derangement(
        ids_used, seed=args.derangement_seed,
    )
    print(f"  derangement     : {der_path}")

    wrapper_kwargs = {}
    if args.max_pixels is not None and "qwen2" in args.model.lower():
        wrapper_kwargs["max_pixels"] = args.max_pixels
        print(f"  max_pixels      : {args.max_pixels}")

    wrapper = create_wrapper(args.model, **wrapper_kwargs).load()
    print(f"  layers={wrapper.num_layers}  hidden_dim={wrapper.hidden_dim}")

    directions, meta, checks = extract_shuffled_control_direction(
        wrapper,
        model_short,
        derangement=derangement,
        derangement_file=der_path,
        demos_path=demos_path,
        deployed_slug=args.deployed_slug,
        force_recompute=args.force_recompute,
    )
    report = write_sanity_report(model_short, checks, meta)
    wrapper.cleanup()

    print(f"  direction_shape : {tuple(directions.shape)}")
    print(f"  forwards        : {checks.get('forwards_executed')}")
    print(f"  report          : {report}")
    print("Done.")


if __name__ == "__main__":
    main()
