#!/usr/bin/env python3
"""Post-grid verification for the AMBER expanded steering direction grid.

Writes run_verification.json under the analysis directory. Safe to run on a
partial grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

MODELS = {
    "llava-1.5-7b-hf": "llava-hf/llava-1.5-7b-hf",
    "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
}

DIRECTION_STEMS = {
    "raw_mean_difference": "meandiff",
    "live_pc1_plus_mean": "r2",
}

ERROR_RE = re.compile(r"\[error\] sample")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _act_cache_digest(act_dir: Path) -> Dict[str, Any]:
    files = sorted(act_dir.glob("*.npz")) if act_dir.is_dir() else []
    return {
        "n_files": len(files),
        "mtime_digest": hashlib.sha256(
            "|".join(f"{f.name}:{f.stat().st_mtime_ns}" for f in files).encode()
        ).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_date", default="2026-08-05")
    p.add_argument("--output_dir", default="evaluation/results")
    p.add_argument(
        "--pre_hashes",
        type=Path,
        default=None,
        help="Optional JSON of direction/act-cache hashes taken before launch.",
    )
    p.add_argument(
        "--llava_log",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--qwen_log",
        type=Path,
        default=None,
    )
    return p.parse_args()


def _current_direction_hashes() -> Dict[str, Any]:
    out: Dict[str, Any] = {"directions": {}, "act_caches": {}}
    for model_short in MODELS:
        for recon, stem in DIRECTION_STEMS.items():
            d = (
                _PROJECT_ROOT
                / "experiment_artifacts/vti"
                / model_short
                / "textual_v2"
                / f"demos850_ba05bd96_all_nd500_s42_{stem}_partition"
            )
            key = f"{model_short}/{recon}"
            out["directions"][key] = {
                "directions_npz_sha256": _sha256(d / "directions.npz"),
                "metadata_json_sha256": _sha256(d / "metadata.json"),
                "path": str(d),
            }
        act = (
            _PROJECT_ROOT
            / "experiment_artifacts/vti"
            / model_short
            / "textual_v2"
            / "_act_cache"
        )
        out["act_caches"][model_short] = {
            "path": str(act),
            **_act_cache_digest(act),
        }
    return out


def _count_error_lines(log_path: Optional[Path]) -> int:
    if log_path is None or not log_path.is_file():
        return -1
    return sum(1 for line in log_path.read_text(errors="replace").splitlines()
               if ERROR_RE.search(line))


def main() -> int:
    args = parse_args()
    results_root = Path(args.output_dir) / args.run_date
    analysis = results_root / "_analysis_steering_vector_validation_continuation"
    analysis.mkdir(parents=True, exist_ok=True)

    pre = {}
    if args.pre_hashes and args.pre_hashes.is_file():
        pre = json.loads(args.pre_hashes.read_text())
    post = _current_direction_hashes()

    cells: List[Dict[str, Any]] = []
    id_sets: List[set] = []
    n_empty_by_cell: Dict[str, int] = {}

    for model_short in MODELS:
        amber = results_root / model_short / "amber"
        if not amber.is_dir():
            continue
        for cell_dir in sorted(p for p in amber.iterdir() if p.is_dir()):
            summary_path = cell_dir / "metric_summary.json"
            responses_path = cell_dir / "responses.json"
            if not summary_path.is_file() or not responses_path.is_file():
                continue
            summary = json.loads(summary_path.read_text())
            records = json.loads(responses_path.read_text())
            ids = {r["id"] for r in records}
            id_sets.append(ids)
            n_total = summary.get("n_total") or summary.get("total") or len(records)
            cfg = summary.get("intervention_config") or {}
            n_empty = sum(
                1 for r in records if (r.get("response") or "").strip() == ""
            )
            cell_key = f"{model_short}/{cell_dir.name}"
            n_empty_by_cell[cell_key] = n_empty
            cells.append({
                "model": model_short,
                "cell_dir": cell_dir.name,
                "n_total": n_total,
                "n_responses": len(records),
                "has_responses_and_summary": True,
                "steer_reconstruction": cfg.get("steer_reconstruction"),
                "direction_slug": cfg.get("direction_slug"),
                "directions_dir": cfg.get("directions_dir"),
                "layer_indices": cfg.get("layer_indices"),
                "layer_set_label": cfg.get("layer_set_label"),
                "beta": cfg.get("beta"),
                "num_demos": cfg.get("num_demos"),
                "n_empty_response": n_empty,
            })

    id_sets_identical = (
        all(s == id_sets[0] for s in id_sets) if id_sets else True
    )
    all_n_total_1500 = all(c["n_total"] == 1500 for c in cells) if cells else False

    # Direction hash stability.
    direction_unchanged = True
    direction_diffs: List[str] = []
    if pre.get("directions"):
        for key, before in pre["directions"].items():
            after = post["directions"].get(key, {})
            for field in ("directions_npz_sha256", "metadata_json_sha256"):
                if before.get(field) != after.get(field):
                    direction_unchanged = False
                    direction_diffs.append(f"{key}.{field}")

    act_unchanged = True
    act_diffs: List[str] = []
    if pre.get("act_caches"):
        for model_short, before in pre["act_caches"].items():
            after = post["act_caches"].get(model_short, {})
            for field in ("n_files", "mtime_digest"):
                if before.get(field) != after.get(field):
                    act_unchanged = False
                    act_diffs.append(f"{model_short}.{field}")

    llava_log = args.llava_log or Path(
        f"logs/amber_expanded_steering_direction_grid_llava_{args.run_date}.log"
    )
    qwen_log = args.qwen_log or Path(
        f"logs/amber_expanded_steering_direction_grid_qwen25_{args.run_date}.log"
    )
    error_counts = {
        "llava_log": str(llava_log),
        "qwen_log": str(qwen_log),
        "llava_error_sample_lines": _count_error_lines(llava_log),
        "qwen_error_sample_lines": _count_error_lines(qwen_log),
        "n_empty_response_by_cell": n_empty_by_cell,
    }

    report = {
        "run_date": args.run_date,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "n_complete_cells": len(cells),
        "n_expected_cells": 38,
        "all_present_cells_n_total_1500": all_n_total_1500,
        "id_sets_identical_across_present_cells": id_sets_identical,
        "cells": cells,
        "direction_hashes_before": pre.get("directions"),
        "direction_hashes_after": post["directions"],
        "direction_hashes_unchanged": direction_unchanged,
        "direction_hash_diffs": direction_diffs,
        "act_cache_before": pre.get("act_caches"),
        "act_cache_after": post["act_caches"],
        "act_cache_unchanged": act_unchanged,
        "act_cache_diffs": act_diffs,
        "error_line_reconciliation": error_counts,
    }
    out = analysis / "run_verification.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[wrote] {out}")
    print(f"  n_complete={len(cells)}/38")
    print(f"  all_n_total_1500={all_n_total_1500}")
    print(f"  id_sets_identical={id_sets_identical}")
    print(f"  direction_hashes_unchanged={direction_unchanged}")
    print(f"  act_cache_unchanged={act_unchanged}")

    gating_ok = (
        (len(cells) == 0 or (all_n_total_1500 and id_sets_identical))
        and direction_unchanged
        and act_unchanged
    )
    return 0 if gating_ok else 1


if __name__ == "__main__":
    # Snapshot helper for pre-launch hashes when --pre_hashes not given:
    # callers can dump _current_direction_hashes() themselves.
    raise SystemExit(main())
