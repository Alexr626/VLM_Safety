#!/usr/bin/env python3
"""Extract demos_850 raw mean-difference textual VTI directions (CPU only).

Zero forward passes. Reads ``textual_v2/_act_cache`` only; never constructs a
model wrapper. Writes eight direction directories (2 models × 4 block sizes)
plus a run manifest.

Example::

    python evaluation/run_scripts/extract_demos850_meandiff_directions.py \\
      --models llava-hf/llava-1.5-7b-hf Qwen/Qwen2.5-VL-7B-Instruct \\
      --dimension all --sizes 50 100 200 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import _normalize_model_name  # noqa: E402
from src.paths import (  # noqa: E402
    experiment_artifacts_dir,
    project_root,
    vti_demos_850_partition_path,
    vti_demos_850_path,
)
from evaluation.interventions.vti.directions_meandiff import (  # noqa: E402
    compute_or_load_meandiff_directions,
    meandiff_cache_dir,
    meandiff_partition_slug,
)
from evaluation.interventions.vti.directions_partition import (  # noqa: E402
    PARTITION_SIZES,
    build_or_load_partition,
    check_partition_integrity,
    resolve_act_cache_dir,
)
from evaluation.interventions.vti.directions_v2 import (  # noqa: E402
    demos_content_hash,
)


MANIFEST_NAME = "demos850_meandiff_extraction_manifest_2026-07-30.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "llava-hf/llava-1.5-7b-hf",
            "Qwen/Qwen2.5-VL-7B-Instruct",
        ],
    )
    p.add_argument("--demos_path", default=str(vti_demos_850_path()))
    p.add_argument("--partition_path", default=str(vti_demos_850_partition_path()))
    p.add_argument("--dimension", default="all")
    p.add_argument("--sizes", nargs="+", type=int, default=list(PARTITION_SIZES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_recompute", action="store_true")
    return p.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _act_cache_digest(act_dir: Path) -> dict:
    files = sorted(act_dir.glob("*.npz")) if act_dir.is_dir() else []
    mtimes = [f.stat().st_mtime_ns for f in files]
    return {
        "path": str(act_dir),
        "n_files": len(files),
        "mtime_digest_sha256_16": hashlib.sha256(
            ("|".join(f"{f.name}:{m}" for f, m in zip(files, mtimes))).encode()
        ).hexdigest()[:16]
        if files
        else None,
    }


def main() -> None:
    args = parse_args()
    demos_path = Path(args.demos_path)
    partition_path = Path(args.partition_path)
    if not demos_path.is_file():
        raise SystemExit(f"Missing demos file: {demos_path}")

    demos_hash = demos_content_hash(demos_path)
    part = build_or_load_partition(demos_path, partition_path, seed=args.seed)
    check_partition_integrity(demos_path, part, sizes=args.sizes)

    print("=== demos_850 mean-difference extraction (CPU, 0 forwards) ===")
    print(f"  demos      : {demos_path}  hash={demos_hash}")
    print(f"  partition  : {partition_path}")
    print(f"  dimension  : {args.dimension}")
    print(f"  sizes      : {args.sizes}")
    print(f"  models     : {args.models}")

    cells: List[dict] = []
    act_before: Dict[str, dict] = {}
    act_after: Dict[str, dict] = {}

    for model_id in args.models:
        model_short = _normalize_model_name(model_id)
        act_dir = resolve_act_cache_dir(model_short)
        act_before[model_short] = _act_cache_digest(act_dir)
        print(f"\n--- model={model_id} ({model_short}) ---")
        print(f"  act_cache before: {act_before[model_short]}")

        for n in args.sizes:
            directions = compute_or_load_meandiff_directions(
                model_short,
                dimension=args.dimension,
                num_demos=n,
                seed=args.seed,
                demos_path=demos_path,
                partition_path=partition_path,
                force_recompute=args.force_recompute,
            )
            slug = meandiff_partition_slug(
                demos_hash, args.dimension, n, seed=args.seed,
            )
            cdir = meandiff_cache_dir(model_short, slug)
            d_path = cdir / "directions.npz"
            m_path = cdir / "metadata.json"
            meta = json.loads(m_path.read_text())
            cells.append({
                "model": model_id,
                "model_short": model_short,
                "slug": slug,
                "cache_dir": str(cdir),
                "num_demos": n,
                "directions_shape": list(directions.shape),
                "n_pairs": meta.get("n_pairs"),
                "forwards_executed": meta.get("forwards_executed", 0),
                "directions_npz_sha256": _sha256(d_path),
                "metadata_json_sha256": _sha256(m_path),
            })

        act_after[model_short] = _act_cache_digest(act_dir)
        print(f"  act_cache after : {act_after[model_short]}")
        if act_before[model_short] != act_after[model_short]:
            raise SystemExit(
                f"Activation cache changed under {act_dir} during meandiff "
                "extraction (must be read-only)."
            )

    manifest = {
        "date": "2026-07-30",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "demos_path": str(demos_path),
        "demos_content_hash_sha256_16": demos_hash,
        "partition_path": str(partition_path),
        "dimension": args.dimension,
        "sizes": list(args.sizes),
        "steer_reconstruction": "raw_mean_difference",
        "forwards_executed": 0,
        "act_cache_before": act_before,
        "act_cache_after": act_after,
        "cells": cells,
    }
    out_path = project_root() / "experiment_artifacts" / "vti" / MANIFEST_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote manifest {out_path}  cells={len(cells)}")


if __name__ == "__main__":
    main()
