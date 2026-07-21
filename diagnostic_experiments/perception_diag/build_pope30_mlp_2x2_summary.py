#!/usr/bin/env python3
"""Build further-condensed POPE-30 summaries for the LLaVA/Qwen × MLP 2×2 plot.

Writes two JSON files under ``windowed_steering_summary/``:

1. ``pope30_mlp_2x2_mean_p_yes_raw_summary.json``
   — mean unconditional P(yes) (+ bootstrap 95% CI), matching the joint PNG.
2. ``pope30_mlp_2x2_accuracy_and_flips_summary.json``
   — accuracy among parseable + yes→no flip counts on the same axes.

Scope (matches the 2×2 figure)::

    models:     LLaVA-1.5, Qwen2.5-VL
    methods:    additive @ mlp, rotation @ mlp   (not rotation @ layer)
    conditions: neutral, assertive_toward_no
    betas:      0.2, 0.5, 0.9
    windows:    model-specific layer_windows + all layers
    baseline:   included once per (model, condition)

Usage::

    python diagnostic_experiments/perception_diag/build_pope30_mlp_2x2_summary.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from diagnostic_experiments.perception_diag.build_windowed_steering_summary import (  # noqa: E402
    STRENGTHS,
    expected_cell_ids,
)
from diagnostic_experiments.perception_diag.plot_pope30_windowed_steering import (  # noqa: E402
    BETAS_BY_PREFIX,
    CONDITIONS,
    MODEL_SPECS,
    _cell_id_for,
    _load_model_bundle,
    bootstrap_mean_ci,
    window_labels,
)
from src.paths import project_root  # noqa: E402

MLP_METHODS = (
    ("additive_mlp", "additive", "mlp", "additive @ mlp"),
    ("rotation_mlp", "uniform_rotation", "mlp", "rotation @ mlp"),
)
CONDITION_LABELS = {
    "neutral": "neutral question",
    "assertive_toward_no": "leading clause toward no",
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
        "n_yes_among_parseable": m["n_yes_parseable"],
        "accuracy": m["accuracy"],
    }


def _flips_entry(m: Dict[str, Any], baseline_yes: int) -> Dict[str, Any]:
    return {
        "n_items": m["n_items"],
        "baseline_parsed_yes": baseline_yes,
        "flips_yes_to_no": m["flips_yes_to_no"],
        "flip_pairs_excluded": m["flip_pairs_excluded"],
    }


def build_mean_p_yes_summary(bundles: Sequence[Dict[str, Any]]) -> dict:
    models_block: Dict[str, Any] = {}
    for bundle in bundles:
        model_short = bundle["model_short"]
        display = bundle["display_name"]
        num_layers = bundle["num_layers"]
        metrics = bundle["metrics"]
        baseline_stats = bundle["baseline_stats"]
        wlabs = window_labels(num_layers)

        baseline = {}
        for cond in CONDITIONS:
            baseline[cond] = {
                "condition_label": CONDITION_LABELS[cond],
                **_mean_p_yes_entry(baseline_stats[cond]["p_yes_raw"]),
                "n_parsed_yes": baseline_stats[cond]["n_yes"],
                "accuracy": baseline_stats[cond]["accuracy"],
            }

        methods: Dict[str, Any] = {}
        for prefix, method, site, method_label in MLP_METHODS:
            betas = BETAS_BY_PREFIX[prefix]
            by_cond: Dict[str, Any] = {}
            for cond in CONDITIONS:
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
                    "condition_label": CONDITION_LABELS[cond],
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
            "display_name": display,
            "num_layers": num_layers,
            "layer_windows": wlabs,
            "baseline_by_condition": baseline,
            "by_method": methods,
        }

    return {
        "_description": (
            "Further-condensed POPE-30 summary for the joint 2×2 mean-P(yes) plot: "
            "rows = model (LLaVA / Qwen), columns = additive @ mlp / rotation @ mlp. "
            "Values are mean unconditional score_p_yes_raw over n=30 items per "
            "cell-condition, with bootstrap 95% CIs (1000 resamples)."
        ),
        "_matches_plot": (
            "diagnostic_experiments/perception_diag/windowed_steering_summary/plots/"
            "pope30_mean_p_yes_raw_by_layer_window_llava_vs_qwen_additive_vs_rotation_mlp.png"
        ),
        "benchmark": "pope30",
        "gold": "all yes",
        "scope": {
            "models": [b["model_short"] for b in bundles],
            "methods": [p for p, *_ in MLP_METHODS],
            "excluded_methods": ["rotation_layer"],
            "conditions": list(CONDITIONS),
            "betas": list(STRENGTHS),
            "n_items_per_cell_condition": 30,
        },
        "models": models_block,
    }


def build_accuracy_flips_summary(bundles: Sequence[Dict[str, Any]]) -> dict:
    models_block: Dict[str, Any] = {}
    for bundle in bundles:
        model_short = bundle["model_short"]
        display = bundle["display_name"]
        num_layers = bundle["num_layers"]
        metrics = bundle["metrics"]
        baseline_stats = bundle["baseline_stats"]
        wlabs = window_labels(num_layers)

        baseline = {}
        for cond in CONDITIONS:
            bs = baseline_stats[cond]
            baseline[cond] = {
                "condition_label": CONDITION_LABELS[cond],
                "n_items": bs["n_items"],
                "n_parseable": bs["n_parseable"],
                "n_unparseable": bs["n_unparseable"],
                "n_parsed_yes": bs["n_yes"],
                "accuracy": bs["accuracy"],
            }

        methods: Dict[str, Any] = {}
        for prefix, method, site, method_label in MLP_METHODS:
            betas = BETAS_BY_PREFIX[prefix]
            by_cond: Dict[str, Any] = {}
            for cond in CONDITIONS:
                base_yes = baseline_stats[cond]["n_yes"]
                by_window: Dict[str, Any] = {}
                for wlab in wlabs:
                    by_beta: Dict[str, Any] = {}
                    for beta in betas:
                        cell = _cell_id_for(prefix, beta, wlab)
                        m = metrics[cell][cond]
                        by_beta[str(beta)] = {
                            "cell_id": cell,
                            "accuracy": _accuracy_entry(m),
                            "flips_from_baseline_yes": _flips_entry(m, base_yes),
                        }
                    by_window[wlab] = by_beta
                by_cond[cond] = {
                    "condition_label": CONDITION_LABELS[cond],
                    "baseline_parsed_yes_denominator": base_yes,
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
            "display_name": display,
            "num_layers": num_layers,
            "layer_windows": wlabs,
            "baseline_by_condition": baseline,
            "by_method": methods,
        }

    return {
        "_description": (
            "Companion further-condensed POPE-30 summary on the same 2×2 axes as the "
            "joint mean-P(yes) plot (model × additive/rotation @ mlp): accuracy among "
            "parseable responses, and counts of items flipped from baseline yes to "
            "steered no. Flip denominators differ by (model, condition) — see "
            "baseline_parsed_yes_denominator."
        ),
        "_matches_plot_axes": (
            "diagnostic_experiments/perception_diag/windowed_steering_summary/plots/"
            "pope30_mean_p_yes_raw_by_layer_window_llava_vs_qwen_additive_vs_rotation_mlp.png"
        ),
        "benchmark": "pope30",
        "gold": "all yes",
        "scope": {
            "models": [b["model_short"] for b in bundles],
            "methods": [p for p, *_ in MLP_METHODS],
            "excluded_methods": ["rotation_layer"],
            "conditions": list(CONDITIONS),
            "betas": list(STRENGTHS),
            "n_items_per_cell_condition": 30,
        },
        "models": models_block,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: windowed_steering_summary/)",
    )
    args = parser.parse_args(argv)

    root = project_root()
    out_dir = args.out_dir or (
        root
        / "diagnostic_experiments"
        / "perception_diag"
        / "windowed_steering_summary"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = [
        _load_model_bundle("llava-1.5-7b-hf"),
        _load_model_bundle("qwen2.5-vl-7b-instruct"),
    ]
    # Sanity: expected cells present
    for b in bundles:
        expected = set(expected_cell_ids(b["num_layers"]))
        # only need mlp cells for this summary
        for prefix, *_ in MLP_METHODS:
            for beta in STRENGTHS:
                for wlab in window_labels(b["num_layers"]):
                    cell = _cell_id_for(prefix, beta, wlab)
                    if cell not in b["metrics"]:
                        raise SystemExit(f"missing metrics for {b['model_short']} {cell}")
                    if cell not in expected:
                        raise SystemExit(f"{cell} not in expected set for {b['model_short']}")

    mean_path = out_dir / "pope30_mlp_2x2_mean_p_yes_raw_summary.json"
    acc_path = out_dir / "pope30_mlp_2x2_accuracy_and_flips_summary.json"

    mean_doc = build_mean_p_yes_summary(bundles)
    acc_doc = build_accuracy_flips_summary(bundles)

    mean_path.write_text(json.dumps(mean_doc, indent=2) + "\n", encoding="utf-8")
    acc_path.write_text(json.dumps(acc_doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {mean_path} ({mean_path.stat().st_size / 1e3:.1f} KB)")
    print(f"wrote {acc_path} ({acc_path.stat().st_size / 1e3:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
