#!/usr/bin/env python3
"""Build per-configuration and per-item tables for the AMBER expanded grid.

Walks evaluation/results/{run_date}/{model_short}/amber/{cell_dir}/, skips cells
without metric_summary.json, recomputes counts from responses.json, cross-checks
against the cell's own metric_summary.json, and writes five artifact files under
_analysis_steering_vector_validation_continuation/. Computes no cell-to-cell
contrasts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.classifiers.metrics import _normalize_yes_no  # noqa: E402

MODELS = ("llava-1.5-7b-hf", "qwen2.5-vl-7b-instruct")
ANALYSIS_DIR_NAME = "_analysis_steering_vector_validation_continuation"
EXPECTED_CELL_COUNT = 38

STEERED_RE = re.compile(
    r"^vti_textual_additive_mlp__b(?P<beta>[^_]+)__d(?P<dim>[^_]+)__"
    r"nd(?P<nd>\d+)__(?P<recon>meandiff|pc1_plus_mean)__layers_(?P<layers>.+)$"
)

RECON_DIR_TO_META = {
    "meandiff": "raw_mean_difference",
    "pc1_plus_mean": "live_pc1_plus_mean",
}

PER_CONFIG_COLUMNS = [
    "model",
    "steer_reconstruction",
    "layer_window",
    "beta",
    "direction_sample_size",
    "direction_slug",
    "n_total",
    "n_unparsed",
    "n_empty_response",
    "n_nonempty_unparsed",
    "tp",
    "fp",
    "tn",
    "fn",
    "n_correct",
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

PER_QTYPE_COLUMNS = PER_CONFIG_COLUMNS + ["question_type"]

PER_ITEM_COLUMNS = [
    "model",
    "steer_reconstruction",
    "layer_window",
    "beta",
    "direction_sample_size",
    "item_id",
    "question_type",
    "ground_truth",
    "image_file",
    "response_is_empty",
    "parsed_answer",
    "is_correct",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_date", default="2026-08-05")
    p.add_argument("--output_dir", default="evaluation/results")
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c) for c in columns})


def layer_label_to_window(label: str) -> str:
    if label == "all":
        return "all"
    return label.replace("_", "-")


def parse_cell_dir(iv_dir: str) -> Optional[dict[str, Any]]:
    if iv_dir == "no_intervention":
        return {
            "is_baseline": True,
            "layer_window_from_dir": "none",
            "beta_from_dir": None,
            "nd_from_dir": None,
            "recon_from_dir": "none",
        }
    m = STEERED_RE.match(iv_dir)
    if m is None:
        return None
    return {
        "is_baseline": False,
        "layer_window_from_dir": layer_label_to_window(m.group("layers")),
        "beta_from_dir": float(m.group("beta")),
        "nd_from_dir": int(m.group("nd")),
        "recon_from_dir": RECON_DIR_TO_META[m.group("recon")],
    }


def score_records(records: list[dict]) -> dict[str, Any]:
    """Recompute overall and by-qtype counts from responses.json."""
    tp = fp = tn = fn = 0
    n_total = 0
    n_unparsed = 0
    n_empty = 0
    n_nonempty_unparsed = 0
    n_correct = 0
    n_gold_yes = n_gold_no = 0
    n_correct_yes = n_correct_no = 0

    by_qtype: dict[str, dict[str, Any]] = {}

    def _bucket(qtype: str) -> dict[str, Any]:
        if qtype not in by_qtype:
            by_qtype[qtype] = {
                "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "n_total": 0, "n_unparsed": 0,
                "n_empty_response": 0, "n_nonempty_unparsed": 0,
                "n_correct": 0,
                "n_gold_yes": 0, "n_gold_no": 0,
                "n_correct_yes": 0, "n_correct_no": 0,
            }
        return by_qtype[qtype]

    per_item: list[dict[str, Any]] = []

    for r in records:
        gt = _normalize_yes_no(r.get("ground_truth") or "")
        if gt is None:
            continue
        response = r.get("response") or ""
        pred = _normalize_yes_no(response)
        qtype = (r.get("metadata") or {}).get("category") or "unknown"
        image_path = (r.get("metadata") or {}).get("image_path") or ""
        image_file = Path(image_path).name if image_path else ""
        b = _bucket(qtype)

        n_total += 1
        b["n_total"] += 1
        if gt == "yes":
            n_gold_yes += 1
            b["n_gold_yes"] += 1
        else:
            n_gold_no += 1
            b["n_gold_no"] += 1

        response_is_empty = response.strip() == ""
        if pred is None:
            n_unparsed += 1
            b["n_unparsed"] += 1
            if response_is_empty:
                n_empty += 1
                b["n_empty_response"] += 1
            else:
                n_nonempty_unparsed += 1
                b["n_nonempty_unparsed"] += 1
            if gt == "yes":
                fn += 1
                b["fn"] += 1
            parsed_answer = "unparsed"
            is_correct = False
        else:
            parsed_answer = pred
            is_correct = pred == gt
            if is_correct:
                n_correct += 1
                b["n_correct"] += 1
                if gt == "yes":
                    n_correct_yes += 1
                    b["n_correct_yes"] += 1
                else:
                    n_correct_no += 1
                    b["n_correct_no"] += 1
            if gt == "yes" and pred == "yes":
                tp += 1; b["tp"] += 1
            elif gt == "no" and pred == "yes":
                fp += 1; b["fp"] += 1
            elif gt == "yes" and pred == "no":
                fn += 1; b["fn"] += 1
            elif gt == "no" and pred == "no":
                tn += 1; b["tn"] += 1

        per_item.append({
            "item_id": r.get("id"),
            "question_type": qtype,
            "ground_truth": gt,
            "image_file": image_file,
            "response_is_empty": response_is_empty,
            "parsed_answer": parsed_answer,
            "is_correct": is_correct,
        })

    def _rates(tp_, fp_, tn_, fn_, n_total_, n_unparsed_, n_correct_,
               n_gold_yes_, n_gold_no_, n_correct_yes_, n_correct_no_):
        precision = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        recall = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        n_parsed = n_total_ - n_unparsed_
        return {
            "tp": tp_, "fp": fp_, "tn": tn_, "fn": fn_,
            "n_total": n_total_,
            "n_unparsed": n_unparsed_,
            "n_correct": n_correct_,
            "accuracy": (tp_ + tn_) / n_total_ if n_total_ else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "yes_rate_over_parsed_answers": (
                (tp_ + fp_) / n_parsed if n_parsed else 0.0
            ),
            "yes_ratio_over_all_items": (
                (tp_ + fp_) / n_total_ if n_total_ else 0.0
            ),
            "accuracy_gold_no": (
                n_correct_no_ / n_gold_no_ if n_gold_no_ else 0.0
            ),
            "n_gold_no": n_gold_no_,
            "accuracy_gold_yes": (
                n_correct_yes_ / n_gold_yes_ if n_gold_yes_ else 0.0
            ),
            "n_gold_yes": n_gold_yes_,
        }

    overall = _rates(
        tp, fp, tn, fn, n_total, n_unparsed, n_correct,
        n_gold_yes, n_gold_no, n_correct_yes, n_correct_no,
    )
    overall["n_empty_response"] = n_empty
    overall["n_nonempty_unparsed"] = n_nonempty_unparsed

    by_qtype_rates: dict[str, dict[str, Any]] = {}
    for q, b in by_qtype.items():
        rates = _rates(
            b["tp"], b["fp"], b["tn"], b["fn"],
            b["n_total"], b["n_unparsed"], b["n_correct"],
            b["n_gold_yes"], b["n_gold_no"],
            b["n_correct_yes"], b["n_correct_no"],
        )
        rates["n_empty_response"] = b["n_empty_response"]
        rates["n_nonempty_unparsed"] = b["n_nonempty_unparsed"]
        by_qtype_rates[q] = rates

    return {"overall": overall, "by_qtype": by_qtype_rates, "per_item": per_item}


def _approx_eq(a: float, b: float, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a == b
    return abs(float(a) - float(b)) <= tol


def cross_check(summary: dict, recomputed: dict) -> None:
    """Hard error on mismatch with scorer fields."""
    o = recomputed["overall"]
    checks = [
        ("neg_item_accuracy", o["accuracy_gold_no"], summary.get("neg_item_accuracy")),
        ("pos_item_accuracy", o["accuracy_gold_yes"], summary.get("pos_item_accuracy")),
        ("n_neg_total", o["n_gold_no"], summary.get("n_neg_total")),
        ("n_pos_total", o["n_gold_yes"], summary.get("n_pos_total")),
    ]
    for name, got, exp in checks:
        if isinstance(got, float) or isinstance(exp, float):
            ok = _approx_eq(float(got), float(exp or 0.0))
        else:
            ok = got == exp
        if not ok:
            raise RuntimeError(
                f"Cross-check failed on {name}: recomputed={got} "
                f"metric_summary={exp}"
            )

    by_qtype_summary = summary.get("by_qtype") or {}
    for qtype, rates in recomputed["by_qtype"].items():
        if qtype not in by_qtype_summary:
            raise RuntimeError(
                f"Cross-check failed: question type {qtype!r} missing from "
                f"metric_summary by_qtype"
            )
        s = by_qtype_summary[qtype]
        for key_sum, key_re in [
            ("neg_item_accuracy", "accuracy_gold_no"),
            ("pos_item_accuracy", "accuracy_gold_yes"),
            ("n_neg_total", "n_gold_no"),
            ("n_pos_total", "n_gold_yes"),
            ("accuracy", "accuracy"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1", "f1"),
        ]:
            got = rates[key_re]
            exp = s.get(key_sum)
            if isinstance(got, float) or isinstance(exp, float):
                ok = _approx_eq(float(got), float(exp or 0.0))
            else:
                ok = got == exp
            if not ok:
                raise RuntimeError(
                    f"Cross-check failed on by_qtype[{qtype}].{key_sum}: "
                    f"recomputed={got} metric_summary={exp}"
                )


def expected_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    layer_by_model = {
        "llava-1.5-7b-hf": ("all", "5-14", "20-29"),
        "qwen2.5-vl-7b-instruct": ("all", "5-14", "15-24"),
    }
    for model in MODELS:
        cells.append({
            "model": model,
            "steer_reconstruction": "none",
            "layer_window": "none",
            "beta": None,
        })
        for recon in ("raw_mean_difference", "live_pc1_plus_mean"):
            for layer in layer_by_model[model]:
                for beta in (0.2, 0.5, 0.9):
                    cells.append({
                        "model": model,
                        "steer_reconstruction": recon,
                        "layer_window": layer,
                        "beta": beta,
                    })
    assert len(cells) == EXPECTED_CELL_COUNT
    return cells


def main() -> None:
    args = parse_args()
    results_root = Path(args.output_dir) / args.run_date
    analysis_dir = results_root / ANALYSIS_DIR_NAME
    analysis_dir.mkdir(parents=True, exist_ok=True)

    per_config_rows: list[dict] = []
    per_qtype_rows: list[dict] = []
    per_item_rows: list[dict] = []
    complete: list[dict] = []
    absent: list[dict] = []

    found_keys: set[tuple] = set()

    for model in MODELS:
        amber_dir = results_root / model / "amber"
        if not amber_dir.is_dir():
            continue
        for cell_dir in sorted(p.name for p in amber_dir.iterdir() if p.is_dir()):
            parsed = parse_cell_dir(cell_dir)
            if parsed is None:
                continue
            summary_path = amber_dir / cell_dir / "metric_summary.json"
            responses_path = amber_dir / cell_dir / "responses.json"
            if not summary_path.is_file():
                continue
            summary = load_json(summary_path)
            records = load_json(responses_path) if responses_path.is_file() else []
            recomputed = score_records(records)
            cross_check(summary, recomputed)

            cfg = summary.get("intervention_config") or {}
            if parsed["is_baseline"]:
                steer_reconstruction = "none"
                layer_window = "none"
                beta = None
                nd = None
                slug = None
            else:
                steer_reconstruction = cfg.get("steer_reconstruction") or parsed[
                    "recon_from_dir"
                ]
                layer_window = (
                    layer_label_to_window(cfg["layer_set_label"])
                    if cfg.get("layer_set_label")
                    else parsed["layer_window_from_dir"]
                )
                beta = cfg.get("beta", parsed["beta_from_dir"])
                nd = cfg.get("num_demos", parsed["nd_from_dir"])
                slug = cfg.get("direction_slug")

            o = recomputed["overall"]
            base_row = {
                "model": model,
                "steer_reconstruction": steer_reconstruction,
                "layer_window": layer_window,
                "beta": beta,
                "direction_sample_size": nd,
                "direction_slug": slug,
                **o,
            }
            per_config_rows.append(base_row)
            for qtype, rates in sorted(recomputed["by_qtype"].items()):
                per_qtype_rows.append({
                    **{k: base_row[k] for k in (
                        "model", "steer_reconstruction", "layer_window", "beta",
                        "direction_sample_size", "direction_slug",
                    )},
                    "question_type": qtype,
                    **rates,
                })
            for item in recomputed["per_item"]:
                per_item_rows.append({
                    "model": model,
                    "steer_reconstruction": steer_reconstruction,
                    "layer_window": layer_window,
                    "beta": beta,
                    "direction_sample_size": nd,
                    **item,
                })

            cell_id = {
                "model": model,
                "steer_reconstruction": steer_reconstruction,
                "layer_window": layer_window,
                "beta": beta,
            }
            complete.append(cell_id)
            found_keys.add(
                (model, steer_reconstruction, layer_window, beta)
            )

    for exp in expected_cells():
        key = (
            exp["model"],
            exp["steer_reconstruction"],
            exp["layer_window"],
            exp["beta"],
        )
        if key not in found_keys:
            absent.append(exp)

    coverage = {
        "n_complete": len(complete),
        "n_expected": EXPECTED_CELL_COUNT,
        "complete": complete,
        "absent": absent,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    write_csv(
        analysis_dir / "per_configuration_counts_and_accuracy.csv",
        PER_CONFIG_COLUMNS,
        per_config_rows,
    )
    write_csv(
        analysis_dir / "per_configuration_counts_and_accuracy_by_question_type.csv",
        PER_QTYPE_COLUMNS,
        per_qtype_rows,
    )
    write_csv(
        analysis_dir / "per_item_answers_and_correctness.csv",
        PER_ITEM_COLUMNS,
        per_item_rows,
    )
    dump_json(analysis_dir / "coverage.json", coverage)
    dump_json(
        analysis_dir / "result_tables.json",
        {
            "schema_version": 1,
            "run_date": args.run_date,
            "coverage": coverage,
            "per_configuration": per_config_rows,
            "per_configuration_by_question_type": per_qtype_rows,
            # per_item omitted from JSON to keep file size manageable; CSV is
            # the canonical per-item record. Keep a count only.
            "n_per_item_rows": len(per_item_rows),
        },
    )

    print(f"[wrote] tables under {analysis_dir}")
    print(f"  n_complete={coverage['n_complete']}/{EXPECTED_CELL_COUNT}")
    print(f"  per_config_rows={len(per_config_rows)}")
    print(f"  per_item_rows={len(per_item_rows)}")


if __name__ == "__main__":
    main()
