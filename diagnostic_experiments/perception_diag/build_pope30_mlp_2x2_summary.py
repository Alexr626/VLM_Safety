#!/usr/bin/env python3
"""Build further-condensed POPE-30-yes / POPE-30-no MLP 2×2 summaries.

Writes, under ``windowed_steering_summary/``, per dataset:

- ``pope30_{yes|no}_mlp_2x2_mean_p_yes_raw_summary.json``
- ``pope30_{yes|no}_mlp_2x2_accuracy_and_flips_summary.json``

Axes match the existing 2×2 (model × additive/rotation @ mlp). Gold=no
accuracy = fraction of parseable responses parsed ``no``; flip of concern =
baseline-``no`` → steered-``yes`` (``flips_no_to_yes``). Gold=yes keeps
accuracy = fraction parsed ``yes`` and ``flips_yes_to_no``.

Usage::

    python diagnostic_experiments/perception_diag/build_pope30_mlp_2x2_summary.py
    python diagnostic_experiments/perception_diag/build_pope30_mlp_2x2_summary.py \\
      --datasets pope30_yes pope30_no
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from diagnostic_experiments.perception_diag.build_windowed_steering_summary import (  # noqa: E402
    STRENGTHS,
    expected_cell_ids,
    load_manifest,
)
from diagnostic_experiments.perception_diag.plot_pope30_windowed_steering import (  # noqa: E402
    BETAS_BY_PREFIX,
    MODEL_SPECS,
    _cell_id_for,
    bootstrap_mean_ci,
    is_parseable,
    window_labels,
)
from src.paths import perception_dump_dir, project_root  # noqa: E402

MLP_METHODS = (
    ("additive_mlp", "additive", "mlp", "additive @ mlp"),
    ("rotation_mlp", "uniform_rotation", "mlp", "rotation @ mlp"),
)

DATASETS: Dict[str, dict] = {
    "pope30_yes": {
        "run_tag": "pope30_yes_windowed_steering",
        "baseline_in_run_tag": True,
        "conditions": ("neutral", "assertive_toward_no"),
        "condition_labels": {
            "neutral": "neutral question",
            "assertive_toward_no": "leading clause toward no",
        },
        "correct_label": "yes",
        "flip_from": "yes",
        "flip_to": "no",
        "flip_key": "flips_yes_to_no",
        "gold": "all yes",
    },
    "pope30_no": {
        "run_tag": "pope30_no_windowed_steering",
        "baseline_in_run_tag": True,
        "conditions": ("neutral", "assertive_toward_yes"),
        "condition_labels": {
            "neutral": "neutral question",
            "assertive_toward_yes": "leading clause toward yes",
        },
        "correct_label": "no",
        "flip_from": "no",
        "flip_to": "yes",
        "flip_key": "flips_no_to_yes",
        "gold": "all no",
    },
}


def _mean_p_yes_entry(values: np.ndarray) -> Dict[str, Any]:
    mean, lo, hi = bootstrap_mean_ci(values)
    return {
        "n_items": int(len(values)),
        "mean_p_yes_raw": mean,
        "bootstrap_ci95_low": lo,
        "bootstrap_ci95_high": hi,
    }


def _accuracy_entry(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n_items": m["n_items"],
        "n_parseable": m["n_parseable"],
        "n_unparseable": m["n_unparseable"],
        "n_correct_among_parseable": m["n_correct_parseable"],
        "accuracy": m["accuracy"],
    }


def _flips_entry(m: Dict[str, Any], baseline_correct_side: int, flip_key: str) -> Dict[str, Any]:
    return {
        "n_items": m["n_items"],
        "baseline_parsed_flip_from": baseline_correct_side,
        flip_key: m[flip_key],
        "flip_pairs_excluded": m["flip_pairs_excluded"],
    }


def collect_cell_metrics_gold_aware(
    steered_root: Path,
    baseline_man: Dict[Tuple[str, str], dict],
    cell_ids: Sequence[str],
    *,
    conditions: Sequence[str],
    correct_label: str,
    flip_from: str,
    flip_to: str,
    flip_key: str,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for cell_id in cell_ids:
        man = load_manifest(steered_root / cell_id)
        if not man:
            raise FileNotFoundError(
                f"missing/empty steered manifest: {steered_root / cell_id}"
            )
        out[cell_id] = {}
        for cond in conditions:
            item_ids = sorted({iid for iid, c in man if c == cond})
            rows = [man[(iid, cond)] for iid in item_ids]
            parseable = [r for r in rows if is_parseable(r.get("parsed_outcome"))]
            n_unparseable = sum(
                1 for r in rows if not is_parseable(r.get("parsed_outcome"))
            )
            n_parseable = len(parseable)
            n_correct = sum(
                1 for r in parseable if r.get("parsed_outcome") == correct_label
            )
            accuracy = (n_correct / n_parseable) if n_parseable else float("nan")
            p_yes = np.asarray(
                [float(r["score_p_yes_raw"]) for r in rows], dtype=float
            )
            flips = 0
            excluded = 0
            for iid in item_ids:
                srow = man[(iid, cond)]
                brow = baseline_man.get((iid, cond))
                if brow is None:
                    excluded += 1
                    continue
                b_out = brow.get("parsed_outcome")
                s_out = srow.get("parsed_outcome")
                if not is_parseable(b_out) or not is_parseable(s_out):
                    excluded += 1
                    continue
                if b_out == flip_from and s_out == flip_to:
                    flips += 1
            out[cell_id][cond] = {
                "n_items": len(rows),
                "n_parseable": n_parseable,
                "n_unparseable": n_unparseable,
                "n_correct_parseable": n_correct,
                "accuracy": accuracy,
                "p_yes_raw": p_yes,
                "mean_p_yes_raw": float(np.mean(p_yes)) if len(p_yes) else float("nan"),
                flip_key: flips,
                "flip_pairs_excluded": excluded,
            }
    return out


def baseline_condition_stats_gold_aware(
    baseline_man: Dict[Tuple[str, str], dict],
    *,
    conditions: Sequence[str],
    correct_label: str,
    flip_from: str,
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for cond in conditions:
        rows = [r for (iid, c), r in baseline_man.items() if c == cond]
        n = len(rows)
        parseable = [r for r in rows if is_parseable(r.get("parsed_outcome"))]
        n_parseable = len(parseable)
        n_correct = sum(1 for r in parseable if r.get("parsed_outcome") == correct_label)
        n_flip_from = sum(1 for r in rows if r.get("parsed_outcome") == flip_from)
        n_unparseable = n - n_parseable
        p_yes = np.asarray(
            [float(r["score_p_yes_raw"]) for r in rows], dtype=float
        )
        accuracy = (n_correct / n_parseable) if n_parseable else float("nan")
        stats[cond] = {
            "n_items": n,
            "n_parseable": n_parseable,
            "n_unparseable": n_unparseable,
            "n_correct": n_correct,
            "n_flip_from": n_flip_from,
            "accuracy": accuracy,
            "mean_p_yes_raw": float(np.mean(p_yes)) if len(p_yes) else float("nan"),
            "p_yes_raw": p_yes,
        }
    return stats


def _load_dataset_bundle(model_short: str, dataset: str) -> Dict[str, Any]:
    cfg = DATASETS[dataset]
    spec = MODEL_SPECS[model_short]
    num_layers = int(spec["num_layers"])
    steered_root = perception_dump_dir("pope", model_short, cfg["run_tag"])
    baseline_dir = (
        steered_root / "baseline"
        if cfg["baseline_in_run_tag"]
        else perception_dump_dir("pope", model_short, cfg["baseline_run_tag"]) / "baseline"
    )
    cell_ids = expected_cell_ids(num_layers, include_baseline=False)
    missing = [
        c for c in cell_ids if not (steered_root / c / "manifest.jsonl").exists()
    ]
    if missing:
        raise SystemExit(
            f"{model_short}/{dataset}: missing {len(missing)} steered cells, "
            f"e.g. {missing[:3]}"
        )
    baseline_man = load_manifest(baseline_dir)
    if not baseline_man:
        raise SystemExit(f"Empty baseline manifest at {baseline_dir}")
    conditions = cfg["conditions"]
    baseline_man = {k: v for k, v in baseline_man.items() if k[1] in conditions}
    baseline_stats = baseline_condition_stats_gold_aware(
        baseline_man,
        conditions=conditions,
        correct_label=cfg["correct_label"],
        flip_from=cfg["flip_from"],
    )
    metrics = collect_cell_metrics_gold_aware(
        steered_root,
        baseline_man,
        cell_ids,
        conditions=conditions,
        correct_label=cfg["correct_label"],
        flip_from=cfg["flip_from"],
        flip_to=cfg["flip_to"],
        flip_key=cfg["flip_key"],
    )
    return {
        "model_short": model_short,
        "display_name": spec["display_name"],
        "num_layers": num_layers,
        "dataset": dataset,
        "metrics": metrics,
        "baseline_stats": baseline_stats,
        "cfg": cfg,
    }


def build_mean_p_yes_summary(bundles: Sequence[Dict[str, Any]], dataset: str) -> dict:
    cfg = DATASETS[dataset]
    conditions = cfg["conditions"]
    labels = cfg["condition_labels"]
    models_block: Dict[str, Any] = {}
    for bundle in bundles:
        model_short = bundle["model_short"]
        metrics = bundle["metrics"]
        baseline_stats = bundle["baseline_stats"]
        wlabs = window_labels(bundle["num_layers"])

        baseline = {}
        for cond in conditions:
            baseline[cond] = {
                "condition_label": labels[cond],
                **_mean_p_yes_entry(baseline_stats[cond]["p_yes_raw"]),
                "n_correct": baseline_stats[cond]["n_correct"],
                "accuracy": baseline_stats[cond]["accuracy"],
            }

        methods: Dict[str, Any] = {}
        for prefix, method, site, method_label in MLP_METHODS:
            betas = BETAS_BY_PREFIX[prefix]
            by_cond: Dict[str, Any] = {}
            for cond in conditions:
                by_window: Dict[str, Any] = {}
                for wlab in wlabs:
                    by_beta: Dict[str, Any] = {}
                    for beta in betas:
                        cell = _cell_id_for(prefix, beta, wlab)
                        by_beta[str(beta)] = {
                            "cell_id": cell,
                            **_mean_p_yes_entry(metrics[cell][cond]["p_yes_raw"]),
                            "delta_mean_p_yes_raw_vs_baseline": (
                                float(np.mean(metrics[cell][cond]["p_yes_raw"]))
                                - float(baseline_stats[cond]["mean_p_yes_raw"])
                            ),
                        }
                    by_window[wlab] = by_beta
                by_cond[cond] = {
                    "condition_label": labels[cond],
                    "by_layer_window": by_window,
                }
            methods[prefix] = {
                "method": method,
                "site": site,
                "label": method_label,
                "betas": list(betas),
                "layer_windows": wlabs,
                "by_condition": by_cond,
            }

        models_block[model_short] = {
            "display_name": bundle["display_name"],
            "num_layers": bundle["num_layers"],
            "layer_windows": wlabs,
            "baseline_by_condition": baseline,
            "by_method": methods,
        }

    return {
        "_description": (
            f"Further-condensed {dataset} summary for model × additive/rotation @ mlp "
            f"2×2: mean unconditional score_p_yes_raw over n=30 items per "
            f"cell-condition (bootstrap 95% CIs)."
        ),
        "dataset": dataset,
        "benchmark": dataset,
        "gold": cfg["gold"],
        "scope": {
            "models": [b["model_short"] for b in bundles],
            "methods": [p for p, *_ in MLP_METHODS],
            "excluded_methods": ["rotation_layer"],
            "conditions": list(conditions),
            "betas": list(STRENGTHS),
            "n_items_per_cell_condition": 30,
        },
        "models": models_block,
    }


def build_accuracy_flips_summary(bundles: Sequence[Dict[str, Any]], dataset: str) -> dict:
    cfg = DATASETS[dataset]
    conditions = cfg["conditions"]
    labels = cfg["condition_labels"]
    flip_key = cfg["flip_key"]
    models_block: Dict[str, Any] = {}
    for bundle in bundles:
        metrics = bundle["metrics"]
        baseline_stats = bundle["baseline_stats"]
        wlabs = window_labels(bundle["num_layers"])

        baseline = {}
        for cond in conditions:
            bs = baseline_stats[cond]
            baseline[cond] = {
                "condition_label": labels[cond],
                "n_items": bs["n_items"],
                "n_parseable": bs["n_parseable"],
                "n_unparseable": bs["n_unparseable"],
                "n_correct": bs["n_correct"],
                "accuracy": bs["accuracy"],
            }

        methods: Dict[str, Any] = {}
        for prefix, method, site, method_label in MLP_METHODS:
            betas = BETAS_BY_PREFIX[prefix]
            by_cond: Dict[str, Any] = {}
            for cond in conditions:
                base_flip_from = baseline_stats[cond]["n_flip_from"]
                by_window: Dict[str, Any] = {}
                for wlab in wlabs:
                    by_beta: Dict[str, Any] = {}
                    for beta in betas:
                        cell = _cell_id_for(prefix, beta, wlab)
                        m = metrics[cell][cond]
                        by_beta[str(beta)] = {
                            "cell_id": cell,
                            "accuracy": _accuracy_entry(m),
                            "flips_from_baseline": _flips_entry(
                                m, base_flip_from, flip_key
                            ),
                        }
                    by_window[wlab] = by_beta
                by_cond[cond] = {
                    "condition_label": labels[cond],
                    "baseline_parsed_flip_from_denominator": base_flip_from,
                    "by_layer_window": by_window,
                }
            methods[prefix] = {
                "method": method,
                "site": site,
                "label": method_label,
                "betas": list(betas),
                "layer_windows": wlabs,
                "by_condition": by_cond,
            }

        models_block[bundle["model_short"]] = {
            "display_name": bundle["display_name"],
            "num_layers": bundle["num_layers"],
            "layer_windows": wlabs,
            "baseline_by_condition": baseline,
            "by_method": methods,
        }

    correct = cfg["correct_label"]
    return {
        "_description": (
            f"Companion further-condensed {dataset} summary on model × "
            f"additive/rotation @ mlp axes: accuracy = fraction of parseable "
            f"responses parsed '{correct}'; flip of concern = baseline-"
            f"'{cfg['flip_from']}' → steered-'{cfg['flip_to']}' ({flip_key}). "
            f"n=30 items per cell-condition."
        ),
        "dataset": dataset,
        "benchmark": dataset,
        "gold": cfg["gold"],
        "accuracy_definition": f"fraction parseable parsed '{correct}'",
        "flip_definition": (
            f"baseline '{cfg['flip_from']}' → steered '{cfg['flip_to']}' ({flip_key})"
        ),
        "scope": {
            "models": [b["model_short"] for b in bundles],
            "methods": [p for p, *_ in MLP_METHODS],
            "excluded_methods": ["rotation_layer"],
            "conditions": list(conditions),
            "betas": list(STRENGTHS),
            "n_items_per_cell_condition": 30,
        },
        "models": models_block,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llava-1.5-7b-hf", "qwen2.5-vl-7b-instruct"],
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args(argv)

    out_dir = args.out_dir or (
        project_root()
        / "diagnostic_experiments"
        / "perception_diag"
        / "windowed_steering_summary"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        bundles = [_load_dataset_bundle(m, dataset) for m in args.models]
        for b in bundles:
            for prefix, *_ in MLP_METHODS:
                for beta in STRENGTHS:
                    for wlab in window_labels(b["num_layers"]):
                        cell = _cell_id_for(prefix, beta, wlab)
                        if cell not in b["metrics"]:
                            raise SystemExit(
                                f"missing metrics for {b['model_short']}/{dataset} {cell}"
                            )

        mean_path = out_dir / f"{dataset}_mlp_2x2_mean_p_yes_raw_summary.json"
        acc_path = out_dir / f"{dataset}_mlp_2x2_accuracy_and_flips_summary.json"
        mean_path.write_text(
            json.dumps(build_mean_p_yes_summary(bundles, dataset), indent=2) + "\n",
            encoding="utf-8",
        )
        acc_path.write_text(
            json.dumps(build_accuracy_flips_summary(bundles, dataset), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {mean_path} ({mean_path.stat().st_size / 1e3:.1f} KB)")
        print(f"wrote {acc_path} ({acc_path.stat().st_size / 1e3:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
