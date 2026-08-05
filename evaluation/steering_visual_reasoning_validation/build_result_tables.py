#!/usr/bin/env python3
"""Build result tables for the steering visual reasoning validation run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.classifiers.metrics import _normalize_yes_no  # noqa: E402

MODELS = ("llava-1.5-7b-hf", "qwen2.5-vl-7b-instruct")
BENCHMARKS = ("amber", "chair", "pope_random", "pope_popular", "pope_adversarial")
DISCRIMINATIVE_BENCHMARKS = ("amber", "pope_random", "pope_popular", "pope_adversarial")
LAYER_SETS_BY_MODEL = {
    "llava-1.5-7b-hf": ("all", "5-14", "20-29"),
    "qwen2.5-vl-7b-instruct": ("all", "5-14", "15-24"),
}
DIRECTION_SAMPLE_SIZES = (50, 100, 200, 500)
BETAS = (0.2, 0.5, 0.9)
EXPECTED_CELL_COUNT = 370

ANALYSIS_DIR_NAME = "_analysis_steering_visual_reasoning_validation"
STEERED_RE = re.compile(
    r"^vti_textual_additive_mlp__b(?P<beta>[^_]+)__d(?P<dim>[^_]+)__"
    r"nd(?P<nd>\d+)__meandiff__layers_(?P<layers>.+)$"
)

DISCRIMINATIVE_COLUMNS = [
    "model",
    "benchmark",
    "layer_set",
    "direction_sample_size",
    "beta",
    "n_total",
    "n_unparsed",
    "tp",
    "fp",
    "tn",
    "fn",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "yes_rate_over_parsed_answers",
    "yes_ratio_over_all_items",
    "accuracy_gold_no",
    "n_gold_no",
    "accuracy_gold_yes",
    "n_gold_yes",
]
FLIP_COLUMNS = [
    "model",
    "benchmark",
    "layer_set",
    "direction_sample_size",
    "beta",
    "hallucinations_induced_baseline_tn_to_steered_fp",
    "hallucinations_removed_baseline_fp_to_steered_tn",
    "n_decision_flips",
    "n_flip_correct_to_wrong",
    "n_flip_wrong_to_correct",
    "n_items_joined",
]
CHAIR_COLUMNS = [
    "model",
    "layer_set",
    "direction_sample_size",
    "beta",
    "chair_s",
    "chair_i",
    "n_total",
    "n_nonempty",
    "n_empty",
    "empty_fraction",
    "avg_objects_mentioned",
    "avg_caption_len_chars",
]
WILSON_COLUMNS = [
    "model",
    "benchmark",
    "n_correct",
    "n_total",
    "accuracy",
    "wilson_lower_95",
    "wilson_upper_95",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_date", required=True)
    parser.add_argument("--output_dir", default="evaluation/results")
    return parser.parse_args()

def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)

def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")

def parse_iv_dir(iv_dir: str) -> dict[str, Any] | None:
    if iv_dir == "no_intervention":
        return {
            "intervention": "no_intervention",
            "layer_set": None,
            "direction_sample_size": None,
            "beta": None,
            "is_baseline": True,
        }

    match = STEERED_RE.match(iv_dir)
    if match is None:
        return None

    layer_label = match.group("layers")
    layer_set = "all" if layer_label == "all" else layer_label.replace("_", "-")
    return {
        "intervention": "vti_textual_additive_mlp",
        "layer_set": layer_set,
        "direction_sample_size": int(match.group("nd")),
        "beta": float(match.group("beta")),
        "is_baseline": False,
    }

def expected_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for model in MODELS:
        for benchmark in BENCHMARKS:
            cells.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "intervention": "no_intervention",
                    "layer_set": None,
                    "direction_sample_size": None,
                    "beta": None,
                }
            )
            for layer_set in LAYER_SETS_BY_MODEL[model]:
                for nd in DIRECTION_SAMPLE_SIZES:
                    for beta in BETAS:
                        cells.append(
                            {
                                "model": model,
                                "benchmark": benchmark,
                                "intervention": "vti_textual_additive_mlp",
                                "layer_set": layer_set,
                                "direction_sample_size": nd,
                                "beta": beta,
                            }
                        )
    return cells

def cell_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    return (
        cell["model"],
        cell["benchmark"],
        cell["intervention"],
        cell["layer_set"],
        cell["direction_sample_size"],
        cell["beta"],
    )

def discover_complete_cells(run_root: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    complete: dict[tuple[Any, ...], dict[str, Any]] = {}
    for model_dir in sorted(p for p in run_root.iterdir() if p.is_dir()) if run_root.exists() else []:
        if model_dir.name == ANALYSIS_DIR_NAME:
            continue
        for benchmark_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for iv_path in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
                summary_path = iv_path / "metric_summary.json"
                if not summary_path.exists():
                    continue
                parsed = parse_iv_dir(iv_path.name)
                if parsed is None:
                    continue
                cell = {
                    "model": model_dir.name,
                    "benchmark": benchmark_dir.name,
                    "intervention": parsed["intervention"],
                    "layer_set": parsed["layer_set"],
                    "direction_sample_size": parsed["direction_sample_size"],
                    "beta": parsed["beta"],
                    "summary_path": summary_path,
                    "responses_path": iv_path / "responses.json",
                    "iv_dir": iv_path.name,
                    "is_baseline": parsed["is_baseline"],
                }
                complete[cell_key(cell)] = cell
    return complete

def make_coverage(run_date: str, complete: dict[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    expected = expected_cells()
    complete_keys = set(complete)
    cells_complete = [
        {k: cell[k] for k in ("model", "benchmark", "intervention", "layer_set", "direction_sample_size", "beta")}
        for cell in expected
        if cell_key(cell) in complete_keys
    ]
    cells_missing = [cell for cell in expected if cell_key(cell) not in complete_keys]
    benchmarks_not_started = [
        benchmark
        for benchmark in BENCHMARKS
        if not any(cell["benchmark"] == benchmark for cell in cells_complete)
    ]
    return {
        "run_date": run_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cells_complete": cells_complete,
        "cells_missing": cells_missing,
        "benchmarks_not_started": benchmarks_not_started,
        "n_complete": len(cells_complete),
        "n_expected": EXPECTED_CELL_COUNT,
    }

def safe_div(num: float, den: float) -> float | None:
    return num / den if den else None

def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def recompute_discriminative_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    n_total = n_unparsed = 0
    tp = fp = tn = fn = 0
    n_gold_yes = n_gold_no = 0

    for record in records:
        gold = _normalize_yes_no(record.get("ground_truth") or "")
        if gold is None:
            continue

        pred = _normalize_yes_no(record.get("response") or "")
        n_total += 1
        if gold == "yes":
            n_gold_yes += 1
        else:
            n_gold_no += 1

        if pred is None:
            n_unparsed += 1
            if gold == "yes":
                fn += 1
            continue

        if gold == "yes" and pred == "yes":
            tp += 1
        elif gold == "yes" and pred == "no":
            fn += 1
        elif gold == "no" and pred == "yes":
            fp += 1
        elif gold == "no" and pred == "no":
            tn += 1

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    n_parsed = n_total - n_unparsed
    return {
        "n_total": n_total,
        "n_unparsed": n_unparsed,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_div(tp + tn, n_total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_rate_over_parsed_answers": safe_div(tp + fp, n_parsed),
        "yes_ratio_over_all_items": safe_div(tp + fp, n_total),
        "accuracy_gold_no": safe_div(tn, n_gold_no),
        "n_gold_no": n_gold_no,
        "accuracy_gold_yes": safe_div(tp, n_gold_yes),
        "n_gold_yes": n_gold_yes,
        "n_correct": tp + tn,
    }

def assert_close(name: str, got: float | None, expected: float | None, tol: float = 1e-9) -> None:
    if got is None and expected is None:
        return
    if got is None or expected is None or abs(got - expected) > tol:
        raise ValueError(f"AMBER metric mismatch for {name}: recomputed={got} summary={expected}")

def cross_check_amber(counts: dict[str, Any], summary: dict[str, Any], summary_path: Path) -> None:
    if counts["n_gold_yes"] != summary.get("n_pos_total"):
        raise ValueError(
            f"AMBER n_pos_total mismatch at {summary_path}: "
            f"recomputed={counts['n_gold_yes']} summary={summary.get('n_pos_total')}"
        )
    if counts["n_gold_no"] != summary.get("n_neg_total"):
        raise ValueError(
            f"AMBER n_neg_total mismatch at {summary_path}: "
            f"recomputed={counts['n_gold_no']} summary={summary.get('n_neg_total')}"
        )
    assert_close("pos_item_accuracy", counts["accuracy_gold_yes"], summary.get("pos_item_accuracy"))
    assert_close("neg_item_accuracy", counts["accuracy_gold_no"], summary.get("neg_item_accuracy"))

_ZERO_FLIPS = {
    "n_decision_flips": 0,
    "n_flip_correct_to_wrong": 0,
    "n_flip_wrong_to_correct": 0,
    "flip_tp_to_fn": 0,
    "flip_tn_to_fp": 0,
    "flip_fn_to_tp": 0,
    "flip_fp_to_tn": 0,
}

def records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["id"]): record for record in records}

def flip_counts_from_records(
    steered_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
) -> dict[str, Any]:
    steered_by_id = records_by_id(steered_records)
    baseline_by_id = records_by_id(baseline_records)
    assert set(steered_by_id) == set(baseline_by_id), "steered and baseline id sets differ"

    out = dict(_ZERO_FLIPS)
    for sample_id in sorted(steered_by_id):
        steered_record = steered_by_id[sample_id]
        baseline_record = baseline_by_id[sample_id]
        gold = _normalize_yes_no(steered_record.get("ground_truth") or baseline_record.get("ground_truth") or "")
        baseline = _normalize_yes_no(baseline_record.get("response") or "")
        steered = _normalize_yes_no(steered_record.get("response") or "")
        if baseline is None or steered is None or baseline == steered:
            continue
        out["n_decision_flips"] += 1
        if gold is None:
            continue
        if baseline == gold and steered != gold:
            out["n_flip_correct_to_wrong"] += 1
            if gold == "yes":
                out["flip_tp_to_fn"] += 1
            else:
                out["flip_tn_to_fp"] += 1
        elif baseline != gold and steered == gold:
            out["n_flip_wrong_to_correct"] += 1
            if gold == "yes":
                out["flip_fn_to_tp"] += 1
            else:
                out["flip_fp_to_tn"] += 1

    out["n_items_joined"] = len(steered_by_id)
    return out

def wilson_interval(successes: float, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half_width = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)

def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def build_tables(run_root: Path, complete: dict[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    discriminative_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    chair_rows: list[dict[str, Any]] = []
    wilson_rows: list[dict[str, Any]] = []

    response_cache: dict[Path, list[dict[str, Any]]] = {}

    def load_responses(path: Path) -> list[dict[str, Any]]:
        if path not in response_cache:
            response_cache[path] = load_json(path)
        return response_cache[path]

    for key in sorted(complete):
        cell = complete[key]
        benchmark = cell["benchmark"]
        summary = load_json(cell["summary_path"])

        if benchmark in DISCRIMINATIVE_BENCHMARKS:
            if not cell["responses_path"].exists():
                raise FileNotFoundError(f"Missing responses.json for {cell['summary_path']}")
            records = load_responses(cell["responses_path"])
            counts = recompute_discriminative_counts(records)
            if benchmark == "amber":
                cross_check_amber(counts, summary, cell["summary_path"])

            row = {
                "model": cell["model"],
                "benchmark": benchmark,
                "layer_set": "baseline" if cell["is_baseline"] else cell["layer_set"],
                "direction_sample_size": cell["direction_sample_size"],
                "beta": cell["beta"],
                **{col: counts[col] for col in DISCRIMINATIVE_COLUMNS if col in counts},
            }
            discriminative_rows.append(row)

            if cell["is_baseline"]:
                lower, upper = wilson_interval(counts["n_correct"], counts["n_total"])
                wilson_rows.append(
                    {
                        "model": cell["model"],
                        "benchmark": benchmark,
                        "n_correct": counts["n_correct"],
                        "n_total": counts["n_total"],
                        "accuracy": counts["accuracy"],
                        "wilson_lower_95": lower,
                        "wilson_upper_95": upper,
                    }
                )
            else:
                baseline_key = (
                    cell["model"],
                    benchmark,
                    "no_intervention",
                    None,
                    None,
                    None,
                )
                flip_row = {
                    "model": cell["model"],
                    "benchmark": benchmark,
                    "layer_set": cell["layer_set"],
                    "direction_sample_size": cell["direction_sample_size"],
                    "beta": cell["beta"],
                    "hallucinations_induced_baseline_tn_to_steered_fp": None,
                    "hallucinations_removed_baseline_fp_to_steered_tn": None,
                    "n_decision_flips": None,
                    "n_flip_correct_to_wrong": None,
                    "n_flip_wrong_to_correct": None,
                    "n_items_joined": None,
                }
                baseline_cell = complete.get(baseline_key)
                if baseline_cell is not None and baseline_cell["responses_path"].exists():
                    flips = flip_counts_from_records(
                        records,
                        load_responses(baseline_cell["responses_path"]),
                    )
                    flip_row.update(
                        {
                            "hallucinations_induced_baseline_tn_to_steered_fp": flips["flip_tn_to_fp"],
                            "hallucinations_removed_baseline_fp_to_steered_tn": flips["flip_fp_to_tn"],
                            "n_decision_flips": flips["n_decision_flips"],
                            "n_flip_correct_to_wrong": flips["n_flip_correct_to_wrong"],
                            "n_flip_wrong_to_correct": flips["n_flip_wrong_to_correct"],
                            "n_items_joined": flips["n_items_joined"],
                        }
                    )
                flip_rows.append(flip_row)

        elif benchmark == "chair":
            row = {
                "model": cell["model"],
                "layer_set": "baseline" if cell["is_baseline"] else cell["layer_set"],
                "direction_sample_size": cell["direction_sample_size"],
                "beta": cell["beta"],
            }
            for name in CHAIR_COLUMNS:
                if name in row:
                    continue
                row[name] = summary.get(name)
            chair_rows.append(row)

            if cell["is_baseline"]:
                n_nonempty = int(summary.get("n_nonempty") or 0)
                chair_s = summary.get("chair_s")
                successes = chair_s * n_nonempty if isinstance(chair_s, (int, float)) else 0
                lower, upper = wilson_interval(successes, n_nonempty)
                wilson_rows.append(
                    {
                        "model": cell["model"],
                        "benchmark": "chair",
                        "n_correct": successes,
                        "n_total": n_nonempty,
                        "accuracy": chair_s,
                        "wilson_lower_95": lower,
                        "wilson_upper_95": upper,
                    }
                )

    discriminative_rows.sort(key=lambda r: (r["model"], r["benchmark"], r["layer_set"], r["direction_sample_size"] or -1, r["beta"] or -1))
    flip_rows.sort(key=lambda r: (r["model"], r["benchmark"], r["layer_set"], r["direction_sample_size"], r["beta"]))
    chair_rows.sort(key=lambda r: (r["model"], r["layer_set"], r["direction_sample_size"] or -1, r["beta"] or -1))
    wilson_rows.sort(key=lambda r: (r["model"], r["benchmark"]))

    return {
        "discriminative_counts_and_accuracy_by_gold_label": discriminative_rows,
        "hallucinations_induced_and_removed_vs_baseline": flip_rows,
        "chair_scores_with_length_controls": chair_rows,
        "baseline_accuracy_wilson_intervals": wilson_rows,
    }

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    run_root = output_dir / args.run_date
    analysis_dir = run_root / ANALYSIS_DIR_NAME

    complete = discover_complete_cells(run_root)
    coverage = make_coverage(args.run_date, complete)
    tables = build_tables(run_root, complete)

    write_csv(
        analysis_dir / "discriminative_counts_and_accuracy_by_gold_label.csv",
        tables["discriminative_counts_and_accuracy_by_gold_label"],
        DISCRIMINATIVE_COLUMNS,
    )
    write_csv(
        analysis_dir / "hallucinations_induced_and_removed_vs_baseline.csv",
        tables["hallucinations_induced_and_removed_vs_baseline"],
        FLIP_COLUMNS,
    )
    write_csv(
        analysis_dir / "chair_scores_with_length_controls.csv",
        tables["chair_scores_with_length_controls"],
        CHAIR_COLUMNS,
    )
    write_csv(
        analysis_dir / "baseline_accuracy_wilson_intervals.csv",
        tables["baseline_accuracy_wilson_intervals"],
        WILSON_COLUMNS,
    )
    dump_json(analysis_dir / "coverage.json", coverage)
    dump_json(
        analysis_dir / "result_tables.json",
        {
            "schema_version": 1,
            "run_date": args.run_date,
            "generated_at": coverage["generated_at"],
            "coverage": coverage,
            "tables": tables,
        },
    )

if __name__ == "__main__":
    main()
