#!/usr/bin/env python3
"""Extract demos_850 per-layer-PCA directions (deployed + shuffled-image control).

CPU-only. Does not import ``src.model`` and does not construct a wrapper.
Reads activation caches with ``ActivationCache.load`` (raises on miss).

Example::

    python evaluation/run_scripts/extract_demos850_perlayer_pca_directions.py \\
      --models llava-1.5-7b-hf qwen2.5-vl-7b-instruct \\
      --arms deployed control \\
      --num_demos 50 100 200 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import (  # noqa: E402
    vti_demos_850_partition_path,
    vti_demos_850_path,
)
from evaluation.interventions.vti.directions_partition import (  # noqa: E402
    PARTITION_SIZES,
)
from evaluation.interventions.vti.perlayer_pca import (  # noqa: E402
    EXPECTED_DECODER_SHAPES,
    compute_perlayer_direction_for_cell,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(EXPECTED_DECODER_SHAPES.keys()),
        choices=list(EXPECTED_DECODER_SHAPES.keys()),
    )
    p.add_argument(
        "--arms",
        nargs="+",
        default=["deployed", "control"],
        choices=["deployed", "control"],
    )
    p.add_argument(
        "--num_demos",
        nargs="+",
        type=int,
        default=list(PARTITION_SIZES),
    )
    p.add_argument("--demos_path", default=str(vti_demos_850_path()))
    p.add_argument("--partition_path", default=str(vti_demos_850_partition_path()))
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument(
        "--out_root",
        type=Path,
        default=None,
        help="Override experiment_artifacts root (default: project experiment_artifacts/).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for n in args.num_demos:
        if n not in PARTITION_SIZES:
            print(f"ERROR: num_demos {n} not in {PARTITION_SIZES}", file=sys.stderr)
            return 1

    # Guard: this script must not import src.model.
    if "src.model" in sys.modules:
        print("ERROR: src.model is imported; refusing to run", file=sys.stderr)
        return 1

    demos_path = Path(args.demos_path)
    partition_path = Path(args.partition_path)
    artifacts_root = Path(args.out_root) if args.out_root is not None else None

    failures = 0
    for model_short in args.models:
        for arm in args.arms:
            for n in args.num_demos:
                print(f"=== {model_short}  arm={arm}  nd={n}")
                try:
                    directions, meta, checks = compute_perlayer_direction_for_cell(
                        model_short,
                        arm,
                        n,
                        demos_path=demos_path,
                        partition_path=partition_path,
                        force_recompute=args.force_recompute,
                        artifacts_root=artifacts_root,
                    )
                except Exception as e:
                    print(f"  FAIL: {e}", file=sys.stderr)
                    failures += 1
                    continue

                pc1_evr = meta.get("pc1_explained_variance_per_layer") or []
                if pc1_evr:
                    evr_msg = f"PC1_evr=[{min(pc1_evr):.4f}, {max(pc1_evr):.4f}]"
                else:
                    evr_msg = "PC1_evr=n/a"
                print(
                    f"  shape={tuple(directions.shape)}  n_pairs={meta.get('n_pairs')}  "
                    f"{evr_msg}  cache_hits={meta.get('cache_hits')}  "
                    f"forwards={meta.get('forwards_executed')}  "
                    f"out={checks.get('out_dir')}"
                    + ("  [skipped]" if checks.get("skipped_extraction") else "")
                )
                if int(meta.get("forwards_executed", -1)) != 0:
                    print("  FAIL: forwards_executed != 0", file=sys.stderr)
                    failures += 1
                if not meta.get("act_cache_read_only", False):
                    print("  FAIL: act_cache_read_only is not true", file=sys.stderr)
                    failures += 1

    if failures:
        print(f"Done with {failures} failure(s).", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
