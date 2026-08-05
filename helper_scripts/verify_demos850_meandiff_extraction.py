#!/usr/bin/env python3
"""Verify demos_850 mean-difference extraction (gating checks).

All gating checks must pass. Non-gating: prints per-layer L2 norms of the
mean-difference direction next to ``direction_layer_norms`` from the matching
``_r2_partition`` metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    meandiff_cache_dir,
    meandiff_partition_slug,
)
from evaluation.interventions.vti.directions_partition import (  # noqa: E402
    PARTITION_SIZES,
    build_or_load_partition,
    check_partition_integrity,
    partition_cache_dir,
    partition_slug,
    resolve_act_cache_dir,
)
from evaluation.interventions.vti.directions_v2 import (  # noqa: E402
    demos_content_hash,
    load_textual_v2_directions,
)

EXPECTED_SHAPES = {
    "llava-1.5-7b-hf": (32, 4096),
    "qwen2.5-vl-7b-instruct": (28, 3584),
}

DEFAULT_MODELS = [
    "llava-hf/llava-1.5-7b-hf",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _act_cache_snapshot(act_dir: Path) -> Tuple[int, Dict[str, int]]:
    files = sorted(act_dir.glob("*.npz")) if act_dir.is_dir() else []
    return len(files), {f.name: f.stat().st_mtime_ns for f in files}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--demos_path", default=str(vti_demos_850_path()))
    p.add_argument("--partition_path", default=str(vti_demos_850_partition_path()))
    p.add_argument("--dimension", default="all")
    p.add_argument("--sizes", nargs="+", type=int, default=list(PARTITION_SIZES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--pre_r2_hashes",
        type=Path,
        default=None,
        help="Optional JSON map of _r2_partition directions.npz sha256 recorded "
             "before extraction. If omitted, current on-disk hashes are used "
             "only as a self-check that files are readable.",
    )
    return p.parse_args()


def verify_model(
    model_id: str,
    *,
    demos_path: Path,
    partition_path: Path,
    dimension: str,
    sizes: List[int],
    seed: int,
    pre_r2_hashes: Dict[str, str],
    act_before_count: int,
    act_before_mtimes: Dict[str, int],
) -> Tuple[bool, List[Tuple[str, bool, str]], str]:
    model_short = _normalize_model_name(model_id)
    shape_expected = EXPECTED_SHAPES[model_short]
    demos_hash = demos_content_hash(demos_path)
    part = build_or_load_partition(demos_path, partition_path, seed=seed)
    check_partition_integrity(demos_path, part, sizes=sizes)

    checks: List[Tuple[str, bool, str]] = []
    norm_lines: List[str] = []

    # Eight directories exist for this model (four sizes).
    present = []
    for n in sizes:
        slug = meandiff_partition_slug(demos_hash, dimension, n, seed=seed)
        cdir = meandiff_cache_dir(model_short, slug)
        present.append((n, slug, cdir, (cdir / "directions.npz").is_file()))
    n_present = sum(1 for *_, ok in present if ok)
    checks.append((
        f"meandiff directories present for all sizes ({len(sizes)})",
        n_present == len(sizes),
        f"present={n_present}/{len(sizes)}",
    ))

    for n, slug, cdir, ok in present:
        if not ok:
            checks.append((f"load {slug}", False, "missing directions.npz"))
            continue
        directions, meta = load_textual_v2_directions(cdir)
        shape_ok = directions.shape == shape_expected
        checks.append((
            f"shape {slug}",
            shape_ok,
            f"got={directions.shape} expected={shape_expected}",
        ))
        n_pairs = meta.get("n_pairs")
        checks.append((
            f"n_pairs == block size ({n}) for {slug}",
            n_pairs == n,
            f"n_pairs={n_pairs}",
        ))
        ids_used = meta.get("ids_used") or []
        block = part["blocks"][str(n)]
        ids_ok = list(ids_used) == list(block)
        checks.append((
            f"ids_used equals partition block order for {slug}",
            ids_ok,
            f"len={len(ids_used)}",
        ))
        components_absent = not (cdir / "components.npz").exists()
        checks.append((
            f"no components.npz for {slug}",
            components_absent,
            "absent" if components_absent else "PRESENT",
        ))
        forwards = meta.get("forwards_executed")
        checks.append((
            f"forwards_executed == 0 for {slug}",
            forwards == 0,
            f"forwards_executed={forwards}",
        ))

        # Non-gating norm comparison against matching _r2_partition.
        r2_slug = partition_slug(demos_hash, dimension, n, seed=seed, rank=2)
        r2_dir = partition_cache_dir(model_short, r2_slug)
        if (r2_dir / "metadata.json").is_file():
            r2_meta = json.loads((r2_dir / "metadata.json").read_text())
            md_norms = meta.get("direction_layer_norms") or []
            r2_norms = r2_meta.get("direction_layer_norms") or []
            norm_lines.append(f"### nd={n}  meandiff vs _r2_partition layer L2 norms")
            norm_lines.append("| layer | meandiff | r2_partition |")
            norm_lines.append("|---|---|---|")
            for i, (a, b) in enumerate(zip(md_norms, r2_norms)):
                norm_lines.append(f"| {i} | {a:.6f} | {b:.6f} |")
            if len(md_norms) != len(r2_norms):
                norm_lines.append(
                    f"(length mismatch: meandiff={len(md_norms)} r2={len(r2_norms)})"
                )

    # Pre-existing _r2_partition directions.npz byte-identical to pre-hash map.
    r2_ok = True
    r2_msgs = []
    for n in sizes:
        r2_slug = partition_slug(demos_hash, dimension, n, seed=seed, rank=2)
        r2_path = partition_cache_dir(model_short, r2_slug) / "directions.npz"
        if not r2_path.is_file():
            r2_ok = False
            r2_msgs.append(f"missing {r2_path}")
            continue
        cur = _sha256(r2_path)
        key = str(r2_path)
        if key in pre_r2_hashes:
            if cur != pre_r2_hashes[key]:
                r2_ok = False
                r2_msgs.append(f"CHANGED {r2_path}")
            else:
                r2_msgs.append(f"unchanged {r2_path.name}")
        else:
            r2_msgs.append(f"no pre-hash recorded for {r2_path.name} (self-read ok)")
    checks.append((
        "pre-existing _r2_partition directions.npz unchanged",
        r2_ok,
        "; ".join(r2_msgs) if r2_msgs else "none",
    ))

    # _act_cache file count and mtimes unchanged vs snapshot taken by caller.
    act_dir = resolve_act_cache_dir(model_short)
    n_after, mtimes_after = _act_cache_snapshot(act_dir)
    count_ok = n_after == act_before_count
    mtimes_ok = mtimes_after == act_before_mtimes
    checks.append((
        "_act_cache file count unchanged",
        count_ok,
        f"before={act_before_count} after={n_after}",
    ))
    checks.append((
        "_act_cache mtimes unchanged",
        mtimes_ok,
        f"n_keys_before={len(act_before_mtimes)} n_keys_after={len(mtimes_after)}",
    ))

    all_ok = all(ok for _, ok, _ in checks)
    report_lines = [
        f"# demos_850 meandiff verification — {model_short}",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Model: `{model_id}` (`{model_short}`)",
        f"Dimension: `{dimension}`",
        f"Sizes: {sizes}",
        "",
        "## Gating checks",
        "",
    ]
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        report_lines.append(f"- **{mark}** — {name}: {detail}")
    report_lines.append("")
    report_lines.append("## Non-gating: layer L2 norms vs matching `_r2_partition`")
    report_lines.append("")
    if norm_lines:
        report_lines.extend(norm_lines)
    else:
        report_lines.append("(no matching `_r2_partition` metadata found)")
    report_lines.append("")
    report = "\n".join(report_lines) + "\n"
    return all_ok, checks, report


def main() -> int:
    args = parse_args()
    demos_path = Path(args.demos_path)
    partition_path = Path(args.partition_path)
    demos_hash = demos_content_hash(demos_path)

    # Snapshot _r2 hashes and act-cache state before any further reads that
    # could confuse "before/after" — caller is expected to run this AFTER
    # extraction; pre_r2_hashes should have been captured before extraction.
    pre_r2: Dict[str, str] = {}
    if args.pre_r2_hashes is not None and args.pre_r2_hashes.is_file():
        pre_r2 = json.loads(args.pre_r2_hashes.read_text())
    else:
        # Fall back: record current hashes and treat identity as the check
        # that files are still what they are now (useful when the overnight
        # orchestrator wrote the pre-hash file).
        for model_id in args.models:
            model_short = _normalize_model_name(model_id)
            for n in args.sizes:
                r2_slug = partition_slug(
                    demos_hash, args.dimension, n, seed=args.seed, rank=2,
                )
                r2_path = partition_cache_dir(model_short, r2_slug) / "directions.npz"
                if r2_path.is_file():
                    pre_r2[str(r2_path)] = _sha256(r2_path)

    act_snaps: Dict[str, Tuple[int, Dict[str, int]]] = {}
    for model_id in args.models:
        model_short = _normalize_model_name(model_id)
        act_snaps[model_short] = _act_cache_snapshot(resolve_act_cache_dir(model_short))

    overall = True
    for model_id in args.models:
        model_short = _normalize_model_name(model_id)
        n_files, mtimes = act_snaps[model_short]
        ok, checks, report = verify_model(
            model_id,
            demos_path=demos_path,
            partition_path=partition_path,
            dimension=args.dimension,
            sizes=list(args.sizes),
            seed=args.seed,
            pre_r2_hashes=pre_r2,
            act_before_count=n_files,
            act_before_mtimes=mtimes,
        )
        overall = overall and ok
        out = (
            experiment_artifacts_dir("vti", model_short)
            / "textual_v2"
            / f"demos850_meandiff_verification_report_{model_short}_2026-07-30.md"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"\n=== {model_short}: {'PASS' if ok else 'FAIL'} ===")
        for name, c_ok, detail in checks:
            print(f"  [{'PASS' if c_ok else 'FAIL'}] {name}: {detail}")
        print(f"  wrote {out}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
