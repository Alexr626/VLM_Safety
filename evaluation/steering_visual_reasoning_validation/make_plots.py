#!/usr/bin/env python3
"""Make plots from steering visual reasoning validation result tables."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

ANALYSIS_DIR_NAME = "_analysis_steering_visual_reasoning_validation"
MODELS = ("llava-1.5-7b-hf", "qwen2.5-vl-7b-instruct")
DIRECTION_SAMPLE_SIZES = (50, 100, 200, 500)
BETAS = (0.2, 0.5, 0.9)
POPE_BENCHMARKS = ("pope_random", "pope_popular", "pope_adversarial")
LAYER_SETS_BY_MODEL = {
    "llava-1.5-7b-hf": ("all", "5-14", "20-29"),
    "qwen2.5-vl-7b-instruct": ("all", "5-14", "15-24"),
}
LAYER_COLORS = {
    "all": "tab:blue",
    "5-14": "tab:orange",
    "20-29": "tab:green",
    "15-24": "tab:green",
}
LAYER_MARKERS = {
    "all": "o",
    "5-14": "s",
    "20-29": "^",
    "15-24": "^",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_date", required=True)
    parser.add_argument("--output_dir", default="evaluation/results")
    return parser.parse_args()

def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)

def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out

def rows_for(rows: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            out.append(row)
    return out

def baseline_row(rows: list[dict[str, Any]], model: str, benchmark: str) -> dict[str, Any] | None:
    matches = rows_for(rows, model=model, benchmark=benchmark, layer_set="baseline")
    return matches[0] if matches else None

def wilson_row(rows: list[dict[str, Any]], model: str, benchmark: str) -> dict[str, Any] | None:
    matches = rows_for(rows, model=model, benchmark=benchmark)
    return matches[0] if matches else None

def metric_value(row: dict[str, Any], metric: str) -> float | None:
    return as_float(row.get(metric))

def set_percent_axis(ax: plt.Axes, label: str) -> None:
    ax.set_ylabel(label)
    ax.grid(True, axis="y", alpha=0.25)

def draw_baseline(
    ax: plt.Axes,
    baseline: dict[str, Any] | None,
    metric: str,
    wilson: dict[str, Any] | None = None,
) -> None:
    if baseline is None:
        return
    value = metric_value(baseline, metric)
    if value is None:
        return
    ax.axhline(value * 100, color="black", linestyle="--", linewidth=1.2, label="baseline")
    if wilson is not None:
        lower = as_float(wilson.get("wilson_lower_95"))
        upper = as_float(wilson.get("wilson_upper_95"))
        if lower is not None and upper is not None:
            ax.axhspan(lower * 100, upper * 100, color="gray", alpha=0.15, label="baseline Wilson 95%")

def mark_empty(ax: plt.Axes) -> None:
    ax.text(0.5, 0.5, "not yet run", ha="center", va="center", transform=ax.transAxes, color="gray")
    ax.set_xticks(BETAS)

def plot_lines_by_layer(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    model: str,
    benchmark: str,
    nd: int,
    metric: str,
    layer_sets: tuple[str, ...],
) -> bool:
    plotted = False
    for layer_set in layer_sets:
        xs: list[float] = []
        ys: list[float] = []
        for beta in BETAS:
            matches = rows_for(
                rows,
                model=model,
                benchmark=benchmark,
                layer_set=layer_set,
                direction_sample_size=nd,
                beta=beta,
            )
            if not matches:
                continue
            value = metric_value(matches[0], metric)
            if value is None:
                continue
            xs.append(beta)
            ys.append(value * 100)
        if xs:
            plotted = True
            ax.plot(
                xs,
                ys,
                marker=LAYER_MARKERS.get(layer_set, "o"),
                color=LAYER_COLORS.get(layer_set),
                label=layer_set,
            )
    return plotted

def finish_figure(fig: plt.Figure, axes: list[plt.Axes], output_path: Path) -> None:
    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)))
        fig.subplots_adjust(bottom=0.14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_amber_accuracy(tables: dict[str, list[dict[str, Any]]], model: str, out_dir: Path) -> None:
    rows = tables["discriminative_counts_and_accuracy_by_gold_label"]
    wilson_rows = tables["baseline_accuracy_wilson_intervals"]
    layer_sets = LAYER_SETS_BY_MODEL[model]
    fig, axs = plt.subplots(1, len(DIRECTION_SAMPLE_SIZES), figsize=(16, 4), sharey=True)
    axes = list(axs)

    for ax, nd in zip(axes, DIRECTION_SAMPLE_SIZES):
        baseline = baseline_row(rows, model, "amber")
        wilson = wilson_row(wilson_rows, model, "amber")
        draw_baseline(ax, baseline, "accuracy", wilson)
        plotted = plot_lines_by_layer(ax, rows, model, "amber", nd, "accuracy", layer_sets)
        if not plotted:
            mark_empty(ax)
        ax.set_title(f"AMBER, direction sample size {nd}")
        ax.set_xlabel("beta")
        ax.set_xticks(BETAS)
        set_percent_axis(ax, "accuracy (%)")

    fig.suptitle(f"AMBER accuracy by beta, layer set, and direction sample size: {model}")
    finish_figure(fig, axes, out_dir / f"amber_accuracy_by_beta_layer_set_and_direction_sample_size_{model}.png")

def plot_pope_accuracy(tables: dict[str, list[dict[str, Any]]], model: str, out_dir: Path) -> None:
    rows = tables["discriminative_counts_and_accuracy_by_gold_label"]
    wilson_rows = tables["baseline_accuracy_wilson_intervals"]
    layer_sets = LAYER_SETS_BY_MODEL[model]
    fig, axs = plt.subplots(len(POPE_BENCHMARKS), len(DIRECTION_SAMPLE_SIZES), figsize=(16, 10), sharey=True)
    axes = [ax for row in axs for ax in row]

    for row_axes, benchmark in zip(axs, POPE_BENCHMARKS):
        split = benchmark.replace("pope_", "")
        for ax, nd in zip(row_axes, DIRECTION_SAMPLE_SIZES):
            baseline = baseline_row(rows, model, benchmark)
            wilson = wilson_row(wilson_rows, model, benchmark)
            draw_baseline(ax, baseline, "accuracy", wilson)
            plotted = plot_lines_by_layer(ax, rows, model, benchmark, nd, "accuracy", layer_sets)
            if not plotted:
                mark_empty(ax)
            ax.set_title(f"POPE {split}, direction sample size {nd}")
            ax.set_xlabel("beta")
            ax.set_xticks(BETAS)
            set_percent_axis(ax, "accuracy (%)")

    fig.suptitle(f"POPE accuracy by beta, layer set, direction sample size, and split: {model}")
    finish_figure(fig, axes, out_dir / f"pope_accuracy_by_beta_layer_set_and_direction_sample_size_{model}.png")

def plot_chair_hallucination_rates(tables: dict[str, list[dict[str, Any]]], model: str, out_dir: Path) -> None:
    rows = tables["chair_scores_with_length_controls"]
    wilson_rows = tables["baseline_accuracy_wilson_intervals"]
    layer_sets = LAYER_SETS_BY_MODEL[model]
    metrics = (("chair_s", "CHAIR_s hallucination rate (%), lower is better"), ("chair_i", "CHAIR_i hallucination rate (%), lower is better"))
    fig, axs = plt.subplots(len(metrics), len(DIRECTION_SAMPLE_SIZES), figsize=(16, 7), sharey=False)
    axes = [ax for row in axs for ax in row]

    for metric_index, (metric, ylabel) in enumerate(metrics):
        for ax, nd in zip(axs[metric_index], DIRECTION_SAMPLE_SIZES):
            baseline = next((r for r in rows if r.get("model") == model and r.get("layer_set") == "baseline"), None)
            wilson = wilson_row(wilson_rows, model, "chair") if metric == "chair_s" else None
            draw_baseline(ax, baseline, metric, wilson)
            plotted = False
            for layer_set in layer_sets:
                xs: list[float] = []
                ys: list[float] = []
                for beta in BETAS:
                    matches = rows_for(
                        rows,
                        model=model,
                        layer_set=layer_set,
                        direction_sample_size=nd,
                        beta=beta,
                    )
                    if not matches:
                        continue
                    value = metric_value(matches[0], metric)
                    if value is None:
                        continue
                    xs.append(beta)
                    ys.append(value * 100)
                if xs:
                    plotted = True
                    ax.plot(
                        xs,
                        ys,
                        marker=LAYER_MARKERS.get(layer_set, "o"),
                        color=LAYER_COLORS.get(layer_set),
                        label=layer_set,
                    )
            if not plotted:
                mark_empty(ax)
            ax.set_title(f"{metric}, direction sample size {nd}")
            ax.set_xlabel("beta")
            ax.set_xticks(BETAS)
            set_percent_axis(ax, ylabel)

    fig.suptitle(f"CHAIR hallucination rate by beta, layer set, and direction sample size: {model}")
    finish_figure(fig, axes, out_dir / f"chair_hallucination_rate_by_beta_layer_set_and_direction_sample_size_{model}.png")

def plot_amber_gold_label_accuracy(tables: dict[str, list[dict[str, Any]]], model: str, out_dir: Path) -> None:
    rows = tables["discriminative_counts_and_accuracy_by_gold_label"]
    layer_sets = LAYER_SETS_BY_MODEL[model]
    metrics = (
        ("accuracy_gold_no", "accuracy on gold-no items (%)"),
        ("accuracy_gold_yes", "accuracy on gold-yes items (%)"),
    )
    fig, axs = plt.subplots(len(metrics), len(DIRECTION_SAMPLE_SIZES), figsize=(16, 7), sharey=True)
    axes = [ax for row in axs for ax in row]

    for metric_index, (metric, ylabel) in enumerate(metrics):
        for ax, nd in zip(axs[metric_index], DIRECTION_SAMPLE_SIZES):
            baseline = baseline_row(rows, model, "amber")
            draw_baseline(ax, baseline, metric, None)
            plotted = plot_lines_by_layer(ax, rows, model, "amber", nd, metric, layer_sets)
            if not plotted:
                mark_empty(ax)
            ax.set_title(f"{ylabel}, direction sample size {nd}")
            ax.set_xlabel("beta")
            ax.set_xticks(BETAS)
            set_percent_axis(ax, ylabel)

    fig.suptitle(f"AMBER accuracy on gold-no and gold-yes items by beta: {model}")
    finish_figure(fig, axes, out_dir / f"amber_accuracy_on_gold_no_and_gold_yes_items_by_beta_{model}.png")

def plot_yes_rate(tables: dict[str, list[dict[str, Any]]], model: str, group: str, out_dir: Path) -> None:
    rows = tables["discriminative_counts_and_accuracy_by_gold_label"]
    layer_sets = LAYER_SETS_BY_MODEL[model]
    benchmarks = ("amber",) if group == "amber" else POPE_BENCHMARKS
    n_rows = len(benchmarks)
    fig, axs = plt.subplots(n_rows, len(DIRECTION_SAMPLE_SIZES), figsize=(16, 4 * n_rows), sharey=True)
    if n_rows == 1:
        axs = [axs]
    axes = [ax for row in axs for ax in row]

    for row_axes, benchmark in zip(axs, benchmarks):
        label = "AMBER" if benchmark == "amber" else f"POPE {benchmark.replace('pope_', '')}"
        for ax, nd in zip(row_axes, DIRECTION_SAMPLE_SIZES):
            baseline = baseline_row(rows, model, benchmark)
            draw_baseline(ax, baseline, "yes_rate_over_parsed_answers", None)
            plotted = plot_lines_by_layer(ax, rows, model, benchmark, nd, "yes_rate_over_parsed_answers", layer_sets)
            if not plotted:
                mark_empty(ax)
            ax.set_title(f"{label}, direction sample size {nd}")
            ax.set_xlabel("beta")
            ax.set_xticks(BETAS)
            set_percent_axis(ax, "yes rate over parsed answers (%)")

    fig.suptitle(f"{group.upper()} yes rate over parsed answers by beta: {model}")
    finish_figure(fig, axes, out_dir / f"yes_rate_over_parsed_answers_by_beta_{model}_{group}.png")

def plot_flip_bars(tables: dict[str, list[dict[str, Any]]], model: str, group: str, out_dir: Path) -> None:
    rows = tables["hallucinations_induced_and_removed_vs_baseline"]
    layer_sets = LAYER_SETS_BY_MODEL[model]
    benchmarks = ("amber",) if group == "amber" else POPE_BENCHMARKS
    n_rows = len(benchmarks)
    fig, axs = plt.subplots(n_rows, len(DIRECTION_SAMPLE_SIZES), figsize=(16, 4 * n_rows), sharey=True)
    if n_rows == 1:
        axs = [axs]
    axes = [ax for row in axs for ax in row]

    for row_axes, benchmark in zip(axs, benchmarks):
        label = "AMBER" if benchmark == "amber" else f"POPE {benchmark.replace('pope_', '')}"
        for ax, nd in zip(row_axes, DIRECTION_SAMPLE_SIZES):
            subset = [
                r
                for r in rows
                if r.get("model") == model
                and r.get("benchmark") == benchmark
                and r.get("direction_sample_size") == nd
            ]
            if not subset:
                mark_empty(ax)
            else:
                plotted = False
                offsets = {
                    layer_set: (i - (len(layer_sets) - 1) / 2) * 0.035
                    for i, layer_set in enumerate(layer_sets)
                }
                for layer_set in layer_sets:
                    layer_rows = [r for r in subset if r.get("layer_set") == layer_set]
                    for row in layer_rows:
                        beta = as_float(row.get("beta"))
                        induced = as_float(row.get("hallucinations_induced_baseline_tn_to_steered_fp"))
                        removed = as_float(row.get("hallucinations_removed_baseline_fp_to_steered_tn"))
                        if beta is None or induced is None or removed is None:
                            continue
                        x = beta + offsets[layer_set]
                        ax.bar(
                            x - 0.01,
                            induced,
                            width=0.018,
                            color=LAYER_COLORS.get(layer_set),
                            alpha=0.8,
                            label=f"{layer_set} induced",
                        )
                        ax.bar(
                            x + 0.01,
                            -removed,
                            width=0.018,
                            color=LAYER_COLORS.get(layer_set),
                            alpha=0.35,
                            label=f"{layer_set} removed",
                        )
                        plotted = True
                if not plotted:
                    mark_empty(ax)
                ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{label}, direction sample size {nd}")
            ax.set_xlabel("beta")
            ax.set_xticks(BETAS)
            ax.set_ylabel("count; removed shown below zero")
            ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"{group.upper()} hallucinations induced and removed by beta: {model}")
    finish_figure(fig, axes, out_dir / f"hallucinations_induced_and_removed_by_beta_{model}_{group}.png")

def plot_amber_chair_change(tables: dict[str, list[dict[str, Any]]], model: str, out_dir: Path) -> None:
    disc_rows = tables["discriminative_counts_and_accuracy_by_gold_label"]
    chair_rows = tables["chair_scores_with_length_controls"]
    layer_sets = LAYER_SETS_BY_MODEL[model]
    amber_baseline = baseline_row(disc_rows, model, "amber")
    chair_baseline = next((r for r in chair_rows if r.get("model") == model and r.get("layer_set") == "baseline"), None)

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False
    amber_base_acc = metric_value(amber_baseline, "accuracy") if amber_baseline else None
    chair_base_i = metric_value(chair_baseline, "chair_i") if chair_baseline else None

    if amber_base_acc is not None and chair_base_i is not None:
        for layer_set in layer_sets:
            for nd in DIRECTION_SAMPLE_SIZES:
                for beta in BETAS:
                    amber_matches = rows_for(
                        disc_rows,
                        model=model,
                        benchmark="amber",
                        layer_set=layer_set,
                        direction_sample_size=nd,
                        beta=beta,
                    )
                    chair_matches = rows_for(
                        chair_rows,
                        model=model,
                        layer_set=layer_set,
                        direction_sample_size=nd,
                        beta=beta,
                    )
                    if not amber_matches or not chair_matches:
                        continue
                    amber_acc = metric_value(amber_matches[0], "accuracy")
                    chair_i = metric_value(chair_matches[0], "chair_i")
                    if amber_acc is None or chair_i is None:
                        continue
                    ax.scatter(
                        (amber_acc - amber_base_acc) * 100,
                        (chair_i - chair_base_i) * 100,
                        color=LAYER_COLORS.get(layer_set),
                        marker=LAYER_MARKERS.get(layer_set, "o"),
                        label=layer_set,
                    )
                    ax.annotate(f"nd {nd}, beta {beta}", ((amber_acc - amber_base_acc) * 100, (chair_i - chair_base_i) * 100), fontsize=7)
                    plotted = True

    if not plotted:
        mark_empty(ax)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("AMBER accuracy minus baseline accuracy (percentage points)")
    ax.set_ylabel("CHAIR_i minus baseline CHAIR_i (percentage points)")
    ax.set_title(f"AMBER accuracy change and CHAIR hallucination change from baseline by arm: {model}")
    ax.grid(True, alpha=0.25)
    finish_figure(fig, [ax], out_dir / f"amber_accuracy_change_and_chair_hallucination_change_from_baseline_by_arm_{model}.png")

def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.output_dir) / args.run_date / ANALYSIS_DIR_NAME
    payload = load_json(analysis_dir / "result_tables.json")
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported result_tables.json schema_version: {payload.get('schema_version')}")
    tables = payload["tables"]

    for model in MODELS:
        plot_amber_accuracy(tables, model, analysis_dir)
        plot_pope_accuracy(tables, model, analysis_dir)
        plot_chair_hallucination_rates(tables, model, analysis_dir)
        plot_amber_gold_label_accuracy(tables, model, analysis_dir)
        plot_yes_rate(tables, model, "amber", analysis_dir)
        plot_yes_rate(tables, model, "pope", analysis_dir)
        plot_flip_bars(tables, model, "amber", analysis_dir)
        plot_flip_bars(tables, model, "pope", analysis_dir)
        plot_amber_chair_change(tables, model, analysis_dir)

if __name__ == "__main__":
    main()
