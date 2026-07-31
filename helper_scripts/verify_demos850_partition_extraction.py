#!/usr/bin/env python3
"""Verify demos_850 disjoint-partition extraction artifacts (pass/fail only).

Writes a markdown report and a machine-readable manifest under
``experiment_artifacts/vti/{model_short}/textual_v2/``.
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
    vti_demos_v2_path,
)
from evaluation.interventions.vti.directions_partition import (  # noqa: E402
    DIMENSIONS,
    PARTITION_SIZES,
    build_or_load_partition,
    check_partition_integrity,
    demos_content_hash,
    partition_cache_dir,
    partition_slug,
    resolve_act_cache_dir,
    variant_suffix,
)
from evaluation.interventions.vti.directions_v2 import (  # noqa: E402
    load_demos_v2_rows,
    load_textual_v2_directions,
)
from src.extraction import ActivationCache  # noqa: E402

EXPECTED_SHAPES = {
    "llava-1.5-7b-hf": {
        "stack": (33, 4096),
        "directions": (32, 4096),
    },
    "qwen2.5-vl-7b-instruct": {
        "stack": (29, 3584),
        "directions": (28, 3584),
    },
}

SNAPSHOT = (
    project_root()
    / "data"
    / "vti"
    / "v2"
    / "_summaries_snapshot_555_2026-07-28"
    / "pre_topup_checksums.txt"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_snapshot_hashes(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        sha, _bytes, _lines, rel = parts[0], parts[1], parts[2], parts[3]
        out[rel] = sha
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model id or short name")
    p.add_argument("--demos_path", default=str(vti_demos_850_path()))
    p.add_argument("--partition_path", default=str(vti_demos_850_partition_path()))
    p.add_argument("--act_cache_override", type=Path, default=None)
    p.add_argument("--date_tag", default="2026-07-28")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_short = _normalize_model_name(args.model)
    if model_short not in EXPECTED_SHAPES:
        raise SystemExit(f"Unsupported model_short for verify: {model_short}")
    shapes = EXPECTED_SHAPES[model_short]
    demos_path = Path(args.demos_path)
    partition_path = Path(args.partition_path)
    base_path = vti_demos_v2_path()
    order_path = project_root() / "data" / "vti" / "demos_v2_order_s42.json"

    checks: List[Tuple[str, bool, str]] = []
    cells_manifest: List[dict] = []

    # 1. demos_850 hash + base prefix
    demos_hash = demos_content_hash(demos_path)
    base_bytes = base_path.read_bytes()
    out_bytes = demos_path.read_bytes()
    prefix_ok = out_bytes[: len(base_bytes)] == base_bytes
    checks.append((
        "demos_850 first 555 lines byte-identical to demos_v2",
        prefix_ok,
        f"demos_hash={demos_hash}",
    ))
    n_rows = len(load_demos_v2_rows(demos_path))
    checks.append(("demos_850 has 850 rows", n_rows == 850, f"n={n_rows}"))

    # 2. partition integrity
    part = build_or_load_partition(demos_path, partition_path)
    try:
        check_partition_integrity(demos_path, part)
        part_ok = True
        part_msg = "ok"
    except Exception as e:
        part_ok = False
        part_msg = str(e)
    checks.append(("partition integrity (check 0.5)", part_ok, part_msg))

    v2_ids = {r["id"] for r in load_demos_v2_rows(base_path)}
    all_ids = {r["id"] for r in load_demos_v2_rows(demos_path)}
    new_ids = all_ids - v2_ids
    checks.append((
        "555 v2 ids and 295 new ids disjoint; |new|=295",
        len(v2_ids & new_ids) == 0 and len(new_ids) == 295 and len(v2_ids) == 555,
        f"|v2|={len(v2_ids)} |new|={len(new_ids)} overlap={len(v2_ids & new_ids)}",
    ))

    # 3. activation cache
    act_dir = resolve_act_cache_dir(model_short, args.act_cache_override)
    cache = ActivationCache(str(act_dir))
    variants = ("value",) + DIMENSIONS
    missing = []
    bad_shape = []
    bad_finite = []
    n_present = 0
    for rid in sorted(all_ids):
        for v in variants:
            suffix = variant_suffix(v)
            if not cache.exists(rid, suffix=suffix):
                missing.append(f"{rid}/{v}")
                continue
            act = cache.load(rid, suffix=suffix)
            stack = np.stack(
                [np.asarray(act[i], dtype=np.float32) for i in range(max(act) + 1)],
                axis=0,
            )
            n_present += 1
            if stack.shape != shapes["stack"]:
                bad_shape.append(f"{rid}/{v}:{stack.shape}")
            if not np.isfinite(stack).all() or np.all(stack == 0):
                bad_finite.append(f"{rid}/{v}")
    checks.append((
        f"activation cache complete (expected 5100 under {act_dir})",
        len(missing) == 0 and n_present == 5100,
        f"present={n_present} missing={len(missing)}",
    ))
    checks.append((
        f"activation stacks shape {shapes['stack']}",
        len(bad_shape) == 0,
        f"bad={len(bad_shape)}",
    ))
    checks.append((
        "activation stacks finite and non-zero",
        len(bad_finite) == 0,
        f"bad={len(bad_finite)}",
    ))

    # 4. direction cells
    cell_failures = []
    for dim in DIMENSIONS:
        for n in PARTITION_SIZES:
            slug = partition_slug(demos_hash, dim, n)
            cdir = partition_cache_dir(model_short, slug)
            try:
                directions, meta = load_textual_v2_directions(cdir)
            except Exception as e:
                cell_failures.append(f"{slug}: load failed ({e})")
                continue
            block = part["blocks"][str(n)]
            ok = True
            reasons = []
            if directions.shape != shapes["directions"]:
                ok = False
                reasons.append(f"shape {directions.shape}")
            if not np.isfinite(directions).all():
                ok = False
                reasons.append("non-finite")
            norms = np.linalg.norm(directions, axis=1)
            if not np.all(norms > 0):
                ok = False
                reasons.append("zero-norm layer")
            if meta.get("n_pairs") != n:
                ok = False
                reasons.append(f"n_pairs={meta.get('n_pairs')}")
            if meta.get("ids_used") != block:
                ok = False
                reasons.append("ids_used != block")
            if meta.get("skipped_ids") not in ([], None):
                ok = False
                reasons.append(f"skipped_ids={meta.get('skipped_ids')}")
            if meta.get("selection_policy") != "disjoint_partition":
                ok = False
                reasons.append(f"policy={meta.get('selection_policy')}")
            if meta.get("content_hash_sha256_16") != demos_hash:
                ok = False
                reasons.append("demos hash mismatch")
            if "max_pixels" not in meta:
                ok = False
                reasons.append("max_pixels missing")
            if not ok:
                cell_failures.append(f"{slug}: {', '.join(reasons)}")
            cells_manifest.append({
                "slug": slug,
                "model_short": model_short,
                "dimension": dim,
                "block_size": n,
                "n_pairs": meta.get("n_pairs"),
                "output_path": str(cdir),
                "demos_hash": demos_hash,
                "partition_hash": meta.get("partition_content_hash"),
                "max_pixels": meta.get("max_pixels"),
                "act_cache_dir": meta.get("act_cache_dir"),
                "git_commit": meta.get("git_commit"),
                "pass": ok,
            })
    checks.append((
        "all 20 direction cells pass schema checks",
        len(cell_failures) == 0,
        f"failures={len(cell_failures)}",
    ))

    # 5. pre-existing artifacts unchanged
    snap = _load_snapshot_hashes(SNAPSHOT)
    unchanged = []
    changed = []
    for rel, expected in snap.items():
        path = project_root() / rel
        if not path.is_file():
            changed.append(f"missing {rel}")
            continue
        got = _sha256(path)
        if got == expected:
            unchanged.append(rel)
        else:
            # Only fail for demos_v2 / order / demosv2 directions for this model
            if (
                rel.endswith("demos_v2.jsonl")
                or rel.endswith("demos_v2_order_s42.json")
                or (
                    f"experiment_artifacts/vti/{model_short}/textual_v2/demosv2_9a44f4af_"
                    in rel
                    and rel.endswith("directions.npz")
                )
            ):
                changed.append(rel)
    checks.append((
        "pre-existing demos_v2 / order / demosv2 directions unchanged",
        len(changed) == 0,
        f"changed={len(changed)} checked_snap={len(snap)}",
    ))

    all_pass = all(ok for _, ok, _ in checks)
    out_dir = experiment_artifacts_dir("vti", model_short) / "textual_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / (
        f"demos850_partition_verification_report_{model_short}_{args.date_tag}.md"
    )
    manifest_path = out_dir / (
        f"demos850_partition_extraction_manifest_{args.date_tag}.json"
    )

    lines = [
        f"# demos_850 partition verification — {model_short}",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Overall: {'PASS' if all_pass else 'FAIL'}",
        "",
        "## Checks",
        "",
    ]
    for name, ok, detail in checks:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    if cell_failures:
        lines.extend(["", "## Cell failures", ""])
        for f in cell_failures:
            lines.append(f"- {f}")
    if missing[:20]:
        lines.extend(["", "## Missing activations (sample)", ""])
        for m in missing[:20]:
            lines.append(f"- {m}")
    if changed:
        lines.extend(["", "## Changed pre-existing artifacts", ""])
        for c in changed:
            lines.append(f"- {c}")
    report_path.write_text("\n".join(lines) + "\n")

    manifest = {
        "model": args.model,
        "model_short": model_short,
        "date": args.date_tag,
        "overall_pass": all_pass,
        "demos_path": str(demos_path),
        "demos_hash": demos_hash,
        "partition_path": str(partition_path),
        "act_cache_dir": str(act_dir),
        "checks": [
            {"name": n, "pass": ok, "detail": d} for n, ok, d in checks
        ],
        "cells": cells_manifest,
        "n_activation_files_present": n_present,
        "n_activation_files_missing": len(missing),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"overall={'PASS' if all_pass else 'FAIL'}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
