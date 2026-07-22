#!/usr/bin/env python3
"""Build consolidated human-readable JSON for layer-windowed steering dumps.

Supports PARTIAL builds: aggregates whatever manifests exist on disk and records
completeness under ``_completeness``. Re-running overwrites the same output path.

Usage::

    python diagnostic_experiments/perception_diag/build_windowed_steering_summary.py
    python diagnostic_experiments/perception_diag/build_windowed_steering_summary.py \\
      --models llava-1.5-7b-hf
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import perception_dump_dir, project_root  # noqa: E402
from src.prompt_spans import layer_windows  # noqa: E402

# (model_short, hf_id, num_layers)
MODELS = [
    ("llava-1.5-7b-hf", "llava-hf/llava-1.5-7b-hf", 32),
    ("qwen2.5-vl-7b-instruct", "Qwen/Qwen2.5-VL-7B-Instruct", 28),
]

BENCHMARKS = [
    {
        "dataset": "pope30_yes",
        "name": "pope",
        "run_tag": "pope30_yes_windowed_steering",
        "baseline_in_run_tag": True,
        "augmented_jsonl": "data/pope/augmented_pope30_yes.jsonl",
        "gold_filter": "yes",
    },
    {
        "dataset": "pope30_no",
        "name": "pope",
        "run_tag": "pope30_no_windowed_steering",
        "baseline_in_run_tag": True,
        "augmented_jsonl": "data/pope/augmented_pope30_no.jsonl",
        "gold_filter": "no",
    },
    {
        "dataset": "pope30_legacy",
        "name": "pope",
        "run_tag": "pope30_windowed_steering",
        "baseline_run_tag": "pope30_existence_yes_baseline",
        "baseline_in_run_tag": False,
        "augmented_jsonl": "data/pope/augmented_pope30.jsonl",
        "gold_filter": "yes",
    },
    {
        "dataset": "amber100",
        "name": "amber",
        "run_tag": "amber100_windowed_steering",
        "baseline_run_tag": "amber100_baseline",
        "baseline_in_run_tag": False,
        "augmented_jsonl": "data/amber/augmented_amber100.jsonl",
        "gold_filter": None,
    },
]

WINDOWED_CONFIGS = [
    ("additive_mlp", "additive", "mlp"),
    ("rotation_mlp", "uniform_rotation", "mlp"),
    ("rotation_layer", "uniform_rotation", "layer"),
]
STRENGTHS = (0.2, 0.5, 0.9)

SCORE_KEYS = (
    "p_yes_raw",
    "p_no_raw",
    "p_yes_norm",
    "logit_margin_yes_minus_no",
    "first_token_pred",
)


def _gold_opposing(gold: str) -> str:
    g = (gold or "").strip().lower()
    if g == "no":
        return "assertive_toward_yes"
    if g == "yes":
        return "assertive_toward_no"
    raise ValueError(f"Unexpected gold: {gold!r}")


def expected_cell_ids(num_layers: int, *, include_baseline: bool = False) -> List[str]:
    """Steered windowed-grid cell ids; optionally prepend in-grid ``baseline``."""
    windows = layer_windows(num_layers, width=10, stride=5)
    win_labels = [f"layers_{s}_{e}" for s, e in windows] + ["layers_all"]
    out = []
    for prefix, _m, _s in WINDOWED_CONFIGS:
        for strength in STRENGTHS:
            for wl in win_labels:
                out.append(f"{prefix}_{strength}_{wl}")
    if include_baseline:
        return ["baseline"] + out
    return out


def parse_cell_id(cell_id: str) -> Optional[Dict[str, Any]]:
    """Parse ``{prefix}_{beta}_layers_{start}_{end|all}`` into method/site/beta/layers."""
    if cell_id == "baseline":
        return {
            "method": "none",
            "site": None,
            "beta": 0.0,
            "layers": "all",
            "layer_indices": None,
        }
    for prefix, method, site in WINDOWED_CONFIGS:
        for strength in STRENGTHS:
            head = f"{prefix}_{strength}_"
            if not cell_id.startswith(head):
                continue
            rest = cell_id[len(head) :]
            if rest == "layers_all":
                return {
                    "method": method,
                    "site": site,
                    "beta": float(strength),
                    "layers": "all",
                    "layer_indices": None,
                }
            if rest.startswith("layers_"):
                parts = rest[len("layers_") :].split("_")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    s, e = int(parts[0]), int(parts[1])
                    return {
                        "method": method,
                        "site": site,
                        "beta": float(strength),
                        "layers": f"{s}-{e}",
                        "layer_indices": list(range(s, e + 1)),
                    }
    return None


def load_augmented(path: Path) -> Dict[str, dict]:
    items: Dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            items[row["item_id"]] = row
    return items


def iter_json_records(path: Path):
    """Yield dict records from JSONL or concatenated pretty-printed JSON objects.

    Some baseline manifests (e.g. LLaVA ``amber100_baseline``) were written as
    one pretty-printed object per record rather than single-line JSONL.
    """
    text = path.read_text()
    dec = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = dec.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object in {path}, got {type(obj)}")
        yield obj
        idx = end


def load_manifest(cell_dir: Path) -> Dict[Tuple[str, str], dict]:
    """Map (item_id, condition_id) → manifest row."""
    path = cell_dir / "manifest.jsonl"
    out: Dict[Tuple[str, str], dict] = {}
    if not path.exists():
        return out
    for row in iter_json_records(path):
        out[(row["item_id"], row["condition_id"])] = row
    return out


def _prompt_for(item: dict, condition_id: str) -> Optional[str]:
    for v in item.get("variants", []):
        if v["condition_id"] == condition_id:
            return v["prompt"]
    return None


def _record_from_manifest(
    row: dict,
    *,
    cell_name: str,
    cell_info: dict,
    prompt: Optional[str],
) -> dict:
    rec = {
        "condition_id": row["condition_id"],
        "prompt": prompt,
        "cell": cell_name,
        "method": cell_info.get("method"),
        "site": cell_info.get("site"),
        "beta": cell_info.get("beta"),
        "layers": cell_info.get("layers"),
        "response": row.get("response"),
        "parsed_outcome": row.get("parsed_outcome"),
        "first_token_pred": row.get("score_first_token_pred"),
        "p_yes_raw": row.get("score_p_yes_raw"),
        "p_no_raw": row.get("score_p_no_raw"),
        "p_yes_norm": row.get("score_p_yes_norm"),
        "logit_margin_yes_minus_no": row.get("score_logit_margin_yes_minus_no"),
        "degeneracy_flag": row.get("degeneracy_flag"),
        "truncated": row.get("truncated"),
        "status": row.get("status", "ok"),
    }
    return rec


def build_summary(
    *,
    models: Optional[List[str]] = None,
    root: Optional[Path] = None,
) -> dict:
    root = root or project_root()
    model_filter = set(models) if models else None

    out: Dict[str, Any] = {}
    warnings: List[str] = []
    completeness: Dict[str, Any] = {
        "models_requested": [],
        "models_included": [],
        "run_tags": {},
        "cells_found": {},
        "cells_expected_but_missing": {},
    }

    for model_short, _hf, n_layers in MODELS:
        if model_filter is not None and model_short not in model_filter:
            continue
        completeness["models_requested"].append(model_short)
        model_has_any = False
        model_block: Dict[str, Any] = {}

        for bench in BENCHMARKS:
            bname = bench["name"]
            dataset = bench["dataset"]
            aug_path = root / bench["augmented_jsonl"]
            if not aug_path.exists():
                warnings.append(f"missing augmented jsonl: {aug_path}")
                continue
            items = load_augmented(aug_path)

            steered_dir = perception_dump_dir(bname, model_short, bench["run_tag"])
            if bench.get("baseline_in_run_tag"):
                baseline_dir = steered_dir
            else:
                baseline_dir = perception_dump_dir(
                    bname, model_short, bench["baseline_run_tag"]
                )

            tag_key = f"{model_short}/{dataset}"
            completeness["run_tags"][tag_key] = {
                "dataset": dataset,
                "steered": str(steered_dir),
                "steered_exists": steered_dir.is_dir(),
                "baseline": str(baseline_dir),
                "baseline_exists": (
                    (baseline_dir / "baseline").is_dir()
                    if bench.get("baseline_in_run_tag")
                    else baseline_dir.is_dir()
                ),
                "baseline_in_run_tag": bool(bench.get("baseline_in_run_tag")),
            }

            expected_here = expected_cell_ids(
                n_layers, include_baseline=bool(bench.get("baseline_in_run_tag"))
            )
            found_cells: List[str] = []
            missing_cells: List[str] = []
            cell_manifests: Dict[str, Dict[Tuple[str, str], dict]] = {}

            if steered_dir.is_dir():
                for cell_id in expected_here:
                    if cell_id == "baseline":
                        continue  # loaded separately below
                    cell_dir = steered_dir / cell_id
                    man = load_manifest(cell_dir)
                    if man:
                        cell_manifests[cell_id] = man
                        found_cells.append(cell_id)
                    else:
                        missing_cells.append(cell_id)
                if bench.get("baseline_in_run_tag"):
                    if load_manifest(steered_dir / "baseline"):
                        found_cells = ["baseline"] + found_cells
                    else:
                        missing_cells = ["baseline"] + missing_cells
            else:
                missing_cells = list(expected_here)

            completeness["cells_found"][tag_key] = found_cells
            completeness["cells_expected_but_missing"][tag_key] = missing_cells

            baseline_man = load_manifest(baseline_dir / "baseline")
            if not baseline_man and (
                baseline_dir.is_dir()
                or (bench.get("baseline_in_run_tag") and steered_dir.is_dir())
            ):
                warnings.append(
                    f"baseline manifest missing/empty: {baseline_dir / 'baseline'}"
                )

            if not cell_manifests and not baseline_man:
                continue

            model_has_any = True
            bench_items: Dict[str, Any] = {}

            for item_id, item in items.items():
                gold = item.get("gold")
                try:
                    opposing = _gold_opposing(str(gold))
                except ValueError as exc:
                    warnings.append(f"{item_id}: {exc}")
                    continue
                keep_conds = ("neutral", opposing)
                runs: List[dict] = []

                # Baseline rows (filtered to gold-conditional conditions)
                for cond in keep_conds:
                    brow = baseline_man.get((item_id, cond))
                    if brow is None:
                        continue
                    prompt = _prompt_for(item, cond)
                    for cell_id, man in cell_manifests.items():
                        srow = man.get((item_id, cond))
                        if srow is None:
                            continue
                        pass
                    runs.append(
                        _record_from_manifest(
                            brow,
                            cell_name="baseline",
                            cell_info=parse_cell_id("baseline") or {},
                            prompt=prompt,
                        )
                    )

                for cell_id, man in cell_manifests.items():
                    cell_info = parse_cell_id(cell_id) or {
                        "method": None,
                        "site": None,
                        "beta": None,
                        "layers": None,
                    }
                    # Prefer cell metadata.json if present for layer_indices
                    meta_path = steered_dir / cell_id / "metadata.json"
                    if meta_path.exists():
                        try:
                            meta = json.loads(meta_path.read_text())
                            cell_spec = meta.get("cell") or {}
                            if "layer_indices" in cell_spec:
                                idxs = cell_spec["layer_indices"]
                                if idxs is None:
                                    cell_info["layers"] = "all"
                                    cell_info["layer_indices"] = None
                                else:
                                    cell_info["layer_indices"] = idxs
                                    cell_info["layers"] = f"{min(idxs)}-{max(idxs)}"
                        except Exception as exc:  # noqa: BLE001
                            warnings.append(
                                f"failed reading {meta_path}: {exc}"
                            )
                    for cond in keep_conds:
                        row = man.get((item_id, cond))
                        if row is None:
                            continue
                        prompt = _prompt_for(item, cond)
                        runs.append(
                            _record_from_manifest(
                                row,
                                cell_name=cell_id,
                                cell_info=cell_info,
                                prompt=prompt,
                            )
                        )

                if not runs:
                    continue

                bench_items[item_id] = {
                    "item_id": item_id,
                    "dataset": dataset,
                    "qtype": item.get("qtype"),
                    "gold": gold,
                    "image_path": item.get("image_path"),
                    "question_neutral": item.get("question_neutral"),
                    "runs": runs,
                }

            if bench_items:
                model_block[dataset] = {
                    "dataset": dataset,
                    "benchmark": bname,
                    "gold_filter": bench.get("gold_filter"),
                    "run_tag": bench["run_tag"],
                    "items": bench_items,
                }

        if model_has_any:
            completeness["models_included"].append(model_short)
            out[model_short] = model_block

    # Baseline prompt-string mismatch check: compare prompts we attach from the
    # augmented JSONL against any prompt field stored on baseline metadata /
    # run_metadata (none today). Also scan steered cell metadata for
    # augmented_jsonl path divergence.
    for model_short, _hf, _nl in MODELS:
        if model_filter is not None and model_short not in model_filter:
            continue
        for bench in BENCHMARKS:
            steered_dir = perception_dump_dir(
                bench["name"], model_short, bench["run_tag"]
            )
            if bench.get("baseline_in_run_tag"):
                baseline_dir = steered_dir
            else:
                baseline_dir = perception_dump_dir(
                    bench["name"], model_short, bench["baseline_run_tag"]
                )
            aug = str(root / bench["augmented_jsonl"])
            for label, d in (("steered", steered_dir), ("baseline", baseline_dir)):
                meta_p = d / "run_metadata.json"
                if not meta_p.exists():
                    continue
                try:
                    meta = json.loads(meta_p.read_text())
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"bad run_metadata {meta_p}: {exc}")
                    continue
                recorded = meta.get("augmented_jsonl")
                if recorded and Path(recorded).resolve() != Path(aug).resolve():
                    # Resolve may fail across hosts; also compare basenames
                    if Path(recorded).name != Path(aug).name:
                        warnings.append(
                            f"{label} {model_short}/{bench['dataset']}: "
                            f"augmented_jsonl mismatch "
                            f"recorded={recorded!r} expected={aug!r}"
                        )

    # Per-item prompt consistency: for each (item, condition), all runs should
    # share the same prompt string (from augmented JSONL). If a future dump
    # embeds a different prompt in the manifest, flag it.
    for model_short, datasets in out.items():
        for dataset, block in datasets.items():
            items = block.get("items") or {}
            for item_id, item_block in items.items():
                by_cond: Dict[str, Set[str]] = defaultdict(set)
                for run in item_block["runs"]:
                    if run.get("prompt") is not None:
                        by_cond[run["condition_id"]].add(run["prompt"])
                for cond, prompts in by_cond.items():
                    if len(prompts) > 1:
                        warnings.append(
                            f"prompt mismatch {model_short}/{dataset}/{item_id}/{cond}: "
                            f"{len(prompts)} distinct strings"
                        )

    return {
        "_completeness": completeness,
        "_warnings": warnings,
        "results": out,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        default=None,
        help="Comma-separated model_short filter (default: all known models).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: "
        "diagnostic_experiments/perception_diag/windowed_steering_summary/"
        "windowed_steering_consolidated_results.json).",
    )
    args = p.parse_args()
    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else None
    )
    summary = build_summary(models=models)
    out_path = (
        Path(args.out)
        if args.out
        else (
            project_root()
            / "diagnostic_experiments"
            / "perception_diag"
            / "windowed_steering_summary"
            / "windowed_steering_consolidated_results.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    n_models = len(summary.get("results", {}))
    n_warn = len(summary.get("_warnings", []))
    print(f"Wrote {out_path} (models={n_models}, warnings={n_warn})")


if __name__ == "__main__":
    main()
