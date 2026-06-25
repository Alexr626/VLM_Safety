#!/usr/bin/env python3
"""Print rotation-strength sweep progress from a .progress.json sidecar or checkpoint.

The full checkpoint is one minified JSON line with every caption/response embedded,
which most editors cannot render. Prefer the sidecar written alongside it:

  sweep_uniform_rotation_layer_n500.progress.json   # small, pretty-printed
  sweep_uniform_rotation_layer_n500.checkpoint.json   # full resume payload

Usage:
  python evaluation/chair_amber_diagnostics/inspect_rotation_progress.py \\
      evaluation/results/2026-06-22/qwen2.5-vl-7b-instruct/chair_rotation_strength/
  python evaluation/chair_amber_diagnostics/inspect_rotation_progress.py \\
      evaluation/results/2026-06-22/qwen2.5-vl-7b-instruct/chair_rotation_strength/sweep_uniform_rotation_layer_n500.checkpoint.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_artifacts(path: Path) -> tuple[Path | None, Path | None]:
    if path.is_file():
        if path.name.endswith(".progress.json"):
            return path, path.parent / path.name.replace(
                ".progress.json", ".checkpoint.json")
        if path.name.endswith(".checkpoint.json"):
            return path.parent / path.name.replace(
                ".checkpoint.json", ".progress.json"), path
        raise SystemExit(f"Not a rotation-strength checkpoint/progress file: {path}")
    progress = sorted(path.glob("sweep_*_n*.progress.json"))
    checkpoint = sorted(path.glob("sweep_*_n*.checkpoint.json"))
    return (progress[-1] if progress else None, checkpoint[-1] if checkpoint else None)


def _from_progress(p: Path) -> dict:
    return json.loads(p.read_text())


def _from_checkpoint(p: Path) -> dict:
    ck = json.loads(p.read_text())
    ps = ck.get("per_sample") or []
    betas = ck.get("betas") or []
    # Infer n_total from filename sweep_*_n{N}.checkpoint.json when possible.
    n_total = None
    stem = p.stem  # sweep_uniform_rotation_layer_n500
    if "_n" in stem:
        try:
            n_total = int(stem.rsplit("_n", 1)[-1])
        except ValueError:
            pass
    n_total = n_total or len(ps)
    gens = 1 + len(betas) + 2
    return {
        "benchmark": ck.get("benchmark"),
        "model": ck.get("model"),
        "n_total": n_total,
        "n_completed": len(ps),
        "n_failed_oom": len(ck.get("failed_ids") or []),
        "pct_complete": round(100 * len(ps) / n_total, 1) if n_total else 0.0,
        "last_id": ps[-1]["id"] if ps else None,
        "betas": betas,
        "max_pixels": ck.get("max_pixels"),
        "generations_per_sample": gens,
        "generations_done_approx": len(ps) * gens,
        "generations_total_approx": n_total * gens,
        "failed_ids": ck.get("failed_ids") or [],
        "checkpoint_file": p.name,
        "source": "checkpoint (no .progress.json sidecar yet)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="Sweep output dir or .checkpoint.json / .progress.json")
    args = ap.parse_args()
    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    progress_path, checkpoint_path = _find_artifacts(root)
    if progress_path and progress_path.exists():
        info = _from_progress(progress_path)
        info.setdefault("source", str(progress_path.name))
    elif checkpoint_path and checkpoint_path.exists():
        info = _from_checkpoint(checkpoint_path)
    else:
        raise SystemExit(f"No sweep checkpoint/progress under {root}")

    print(f"benchmark           : {info.get('benchmark')}")
    print(f"model               : {info.get('model')}")
    print(f"completed           : {info.get('n_completed')} / {info.get('n_total')}"
          f"  ({info.get('pct_complete')}%)")
    print(f"failed (OOM)        : {info.get('n_failed_oom')}")
    print(f"last id             : {info.get('last_id')}")
    print(f"betas               : {info.get('betas')}")
    print(f"max_pixels          : {info.get('max_pixels')}")
    if info.get("generations_per_sample"):
        print(f"generations/sample  : {info['generations_per_sample']}")
        print(f"generations done    : ~{info.get('generations_done_approx')}"
              f" / ~{info.get('generations_total_approx')}")
    if info.get("updated_at"):
        print(f"updated_at          : {info['updated_at']}")
    print(f"source              : {info.get('source', progress_path or checkpoint_path)}")


if __name__ == "__main__":
    main()
