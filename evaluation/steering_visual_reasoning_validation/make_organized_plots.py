#!/usr/bin/env python3
"""Organized per-view plots for CHAIR and POPE (AMBER-style subdirectory trees).

Mirrors the layout under ``{llava,qwen}_amber_results/plots/``:

- ``steering_vector_sample_size/{50,100,200,500}/…_by_layer_window_and_beta.png``
- ``layer_windows/{early,late,all}/…_vs_beta_by_steering_vector_sample_size.png``
- ``layer_windows/…_early_late_all.png`` (stitched)
- POPE only: ``accuracy_yes_vs_no_comparisons/`` gold-label views

Reads ``result_tables.json`` produced by ``build_result_tables.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_DIR_NAME = "_analysis_steering_visual_reasoning_validation"
BETAS = (0.2, 0.5, 0.9)
NDS = (50, 100, 200, 500)
POPE_SPLITS = ("random", "popular", "adversarial")

ND_COLORS = {50: "#4C72B0", 100: "#DD8452", 200: "#55A868", 500: "#C44E52"}
BAR_COLORS = {"baseline": "#7A7A7A", 0.2: "#4C72B0", 0.5: "#DD8452", 0.9: "#C44E52"}

# (layer_set key in tables, filename slug, bar xtick label, layer_windows dir, title line)
LAYER_SETS_BY_MODEL = {
    "llava-1.5-7b-hf": (
        ("5-14", "5_14", "early–middle\n(layers 5–14)", "early", "early–middle window (layers 5–14)"),
        ("20-29", "20_29", "late\n(layers 20–29)", "late", "late window (layers 20–29)"),
        ("all", "all", "all layers\n(0–31)", "all", "all layers (0–31)"),
    ),
    "qwen2.5-vl-7b-instruct": (
        ("5-14", "5_14", "early–middle\n(layers 5–14)", "early", "early–middle window (layers 5–14)"),
        ("15-24", "15_24", "late\n(layers 15–24)", "late", "late window (layers 15–24)"),
        ("all", "all", "all layers\n(0–27)", "all", "all layers (0–27)"),
    ),
}

MODEL_DISPLAY = {
    "llava-1.5-7b-hf": "LLaVA",
    "qwen2.5-vl-7b-instruct": "Qwen",
}

MODEL_OUT_STEM = {
    "llava-1.5-7b-hf": "llava",
    "qwen2.5-vl-7b-instruct": "qwen",
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


def require_one(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any]:
    matches = rows_for(rows, **filters)
    if len(matches) != 1:
        raise ValueError(f"expected 1 row for {filters}, got {len(matches)}")
    return matches[0]


def percent(value: float | None) -> float:
    if value is None:
        raise ValueError("missing metric value")
    return value * 100.0


def stitch_early_late_all(lw_root: Path, stem: str) -> Path:
    early = Image.open(lw_root / "early" / f"{stem}.png").convert("RGB")
    late = Image.open(lw_root / "late" / f"{stem}.png").convert("RGB")
    all_img = Image.open(lw_root / "all" / f"{stem}.png").convert("RGB")
    gap = 12
    top_h = max(early.height, late.height)
    top_w = early.width + gap + late.width
    top = Image.new("RGB", (top_w, top_h), (255, 255, 255))
    top.paste(early, (0, 0))
    top.paste(late, (early.width + gap, 0))
    canvas_w = max(top_w, all_img.width)
    canvas_h = top_h + gap + all_img.height
    out = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    out.paste(top, ((canvas_w - top_w) // 2, 0))
    out.paste(all_img, ((canvas_w - all_img.width) // 2, top_h + gap))
    path = lw_root / f"{stem}_early_late_all.png"
    out.save(path, dpi=(180, 180))
    return path


def plot_grouped_bars_by_layer_window(
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    layer_sets: tuple[tuple[str, str, str, str, str], ...],
    baseline_value: float,
    value_at: Any,
    ylim_pad: float = 3.0,
    ylim_max: float = 100.0,
) -> None:
    bar_keys: tuple[Any, ...] = ("baseline", 0.2, 0.5, 0.9)
    bar_labels = ["baseline", "beta 0.2", "beta 0.5", "beta 0.9"]
    n_layers = len(layer_sets)
    n_bars = len(bar_keys)
    x = np.arange(n_layers)
    width = 0.18

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    all_vals: list[float] = []
    for i, key in enumerate(bar_keys):
        if key == "baseline":
            vals = [baseline_value] * n_layers
        else:
            vals = [value_at(lk, key) for lk, *_ in layer_sets]
        all_vals.extend(vals)
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=bar_labels[i],
            color=BAR_COLORS[key],
            edgecolor="white",
            linewidth=0.4,
        )
        for bar, v in zip(bars, vals):
            ax.annotate(
                f"{v:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, _, lab, _, _ in layer_sets])
    ax.set_ylabel(ylabel)
    ax.set_ylim(max(0.0, min(all_vals) - ylim_pad), min(ylim_max, max(all_vals) + ylim_pad))
    ax.set_xlabel("layer window")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lines_vs_beta_by_nd(
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    baseline_value: float,
    value_at: Any,
    ylim_pad: float = 3.0,
    ylim_max: float = 100.0,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    all_vals = [baseline_value]
    for nd in NDS:
        ys = [value_at(nd, b) for b in BETAS]
        all_vals.extend(ys)
        ax.plot(BETAS, ys, marker="o", linewidth=2, color=ND_COLORS[nd], label=f"nd = {nd}")
        for b, y in zip(BETAS, ys):
            ax.annotate(
                f"{y:.1f}",
                (b, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color=ND_COLORS[nd],
            )
    ax.axhline(
        baseline_value,
        color="#7A7A7A",
        linestyle="--",
        linewidth=1.3,
        label=f"baseline ({baseline_value:.1f}%)",
    )
    ax.set_xticks(list(BETAS))
    ax.set_xlabel("beta")
    ax.set_ylabel(ylabel)
    ax.set_ylim(max(0.0, min(all_vals) - ylim_pad), min(ylim_max, max(all_vals) + ylim_pad))
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_metric_views(
    *,
    plots_root: Path,
    display: str,
    layer_sets: tuple[tuple[str, str, str, str, str], ...],
    metric_stem: str,
    metric_title: str,
    ylabel: str,
    baseline_value: float,
    steered_value: Any,
    ylim_pad: float = 3.0,
) -> list[Path]:
    """Write AMBER-style nd and layer-window views for one scalar metric."""
    written: list[Path] = []
    nd_root = plots_root / "steering_vector_sample_size"
    lw_root = plots_root / "layer_windows"

    for nd in NDS:
        path = nd_root / str(nd) / f"{metric_stem}_by_layer_window_and_beta.png"
        plot_grouped_bars_by_layer_window(
            out_path=path,
            title=(
                f"{display} {metric_title} by layer window and beta\n"
                f"steering-vector sample size nd = {nd}"
            ),
            ylabel=ylabel,
            layer_sets=layer_sets,
            baseline_value=baseline_value,
            value_at=lambda lk, beta, nd=nd: steered_value(lk, nd, beta),
            ylim_pad=ylim_pad,
        )
        written.append(path)

    line_stem = f"{metric_stem}_vs_beta_by_steering_vector_sample_size"
    for layer_key, _slug, _short, dir_name, title in layer_sets:
        path = lw_root / dir_name / f"{line_stem}.png"
        plot_lines_vs_beta_by_nd(
            out_path=path,
            title=f"{display} {metric_title} vs beta\n{title}",
            ylabel=ylabel,
            baseline_value=baseline_value,
            value_at=lambda nd, beta, lk=layer_key: steered_value(lk, nd, beta),
            ylim_pad=ylim_pad,
        )
        written.append(path)

    stitch = stitch_early_late_all(lw_root, line_stem)
    written.append(stitch)
    return written


def make_gold_label_views(
    *,
    plots_root: Path,
    model: str,
    display: str,
    layer_sets: tuple[tuple[str, str, str, str, str], ...],
    benchmark_label: str,
    baseline_no: float,
    baseline_yes: float,
    n_gold_no: int,
    n_gold_yes: int,
    gold_at: Any,
) -> list[Path]:
    written: list[Path] = []
    out_dir = plots_root / "accuracy_yes_vs_no_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = {
        "5-14": "early–middle window (layers 5–14)",
        "15-24": "late window (layers 15–24)",
        "20-29": "late window (layers 20–29)",
        "all": "all layers (0–31)" if model.startswith("llava") else "all layers (0–27)",
    }
    labels = ["baseline"] + [label_map[lk] for lk, *_ in layer_sets]
    nos = [baseline_no]
    yeses = [baseline_yes]
    for lk, *_ in layer_sets:
        no_acc, yes_acc = gold_at(lk, 500, 0.9)
        nos.append(no_acc)
        yeses.append(yes_acc)

    vals = nos + yeses
    ymin = max(0.0, min(vals) - 8.0)
    ymax = min(100.0, max(vals) + 8.0)
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_no = ax.bar(x - width / 2, nos, width, label=f"gold-no accuracy (n={n_gold_no})", color="#4C72B0")
    bars_yes = ax.bar(x + width / 2, yeses, width, label=f"gold-yes accuracy (n={n_gold_yes})", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(ymin, ymax)
    ax.set_title(
        f"{display} {benchmark_label} gold-label accuracy by layer window\n"
        "nd = 500, beta = 0.9 (baseline shown for reference)"
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    for bars in (bars_no, bars_yes):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    path = out_dir / "gold_label_accuracy_by_layer_window_nd500_beta0.9.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    for layer_key, slug, _short, _dir, title in layer_sets:
        nos_b = []
        yeses_b = []
        for beta in BETAS:
            no_acc, yes_acc = gold_at(layer_key, 500, beta)
            nos_b.append(no_acc)
            yeses_b.append(yes_acc)
        vals = nos_b + yeses_b + [baseline_no, baseline_yes]
        ymin = max(0.0, min(vals) - 8.0)
        ymax = min(100.0, max(vals) + 8.0)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.plot(
            BETAS,
            nos_b,
            marker="o",
            color="#4C72B0",
            linewidth=2,
            label=f"gold-no accuracy (n={n_gold_no})",
        )
        ax.plot(
            BETAS,
            yeses_b,
            marker="s",
            color="#DD8452",
            linewidth=2,
            label=f"gold-yes accuracy (n={n_gold_yes})",
        )
        ax.axhline(baseline_no, color="#4C72B0", linestyle="--", linewidth=1.2, alpha=0.7, label="baseline gold-no")
        ax.axhline(baseline_yes, color="#DD8452", linestyle="--", linewidth=1.2, alpha=0.7, label="baseline gold-yes")
        for beta, no_v, yes_v in zip(BETAS, nos_b, yeses_b):
            ax.annotate(
                f"{no_v:.1f}",
                (beta, no_v),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="#4C72B0",
            )
            ax.annotate(
                f"{yes_v:.1f}",
                (beta, yes_v),
                textcoords="offset points",
                xytext=(0, -12),
                ha="center",
                fontsize=8,
                color="#DD8452",
            )
        ax.set_xticks(list(BETAS))
        ax.set_xlabel("beta")
        ax.set_ylabel("accuracy (%)")
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{display} {benchmark_label} gold-label accuracy vs beta\n{title}, nd = 500")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = out_dir / f"gold_label_accuracy_vs_beta_nd500_layers_{slug}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    # Stitch gold vs-beta windows: early|late on top, all below
    early = Image.open(out_dir / f"gold_label_accuracy_vs_beta_nd500_layers_{layer_sets[0][1]}.png").convert("RGB")
    late = Image.open(out_dir / f"gold_label_accuracy_vs_beta_nd500_layers_{layer_sets[1][1]}.png").convert("RGB")
    all_img = Image.open(out_dir / f"gold_label_accuracy_vs_beta_nd500_layers_{layer_sets[2][1]}.png").convert("RGB")
    gap = 12
    top_h = max(early.height, late.height)
    top_w = early.width + gap + late.width
    top = Image.new("RGB", (top_w, top_h), (255, 255, 255))
    top.paste(early, (0, 0))
    top.paste(late, (early.width + gap, 0))
    canvas_w = max(top_w, all_img.width)
    canvas_h = top_h + gap + all_img.height
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(top, ((canvas_w - top_w) // 2, 0))
    canvas.paste(all_img, ((canvas_w - all_img.width) // 2, top_h + gap))
    stitch = out_dir / (
        f"gold_label_accuracy_vs_beta_nd500_layers_{layer_sets[0][1]}_"
        f"{layer_sets[1][1]}_{layer_sets[2][1]}.png"
    )
    canvas.save(stitch, dpi=(180, 180))
    written.append(stitch)
    return written


def make_chair_plots(analysis_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    rows = tables["chair_scores_with_length_controls"]
    for model, layer_sets in LAYER_SETS_BY_MODEL.items():
        display = MODEL_DISPLAY[model]
        out_root = analysis_dir / f"{MODEL_OUT_STEM[model]}_chair_results" / "plots"
        baseline = require_one(rows, model=model, layer_set="baseline")
        for metric, stem, metric_title, ylabel in (
            (
                "chair_s",
                "chair_s_hallucination_rate",
                "CHAIR_s hallucination rate",
                "CHAIR_s hallucination rate (%)",
            ),
            (
                "chair_i",
                "chair_i_hallucination_rate",
                "CHAIR_i hallucination rate",
                "CHAIR_i hallucination rate (%)",
            ),
        ):
            baseline_value = percent(as_float(baseline[metric]))

            def steered_value(lk: str, nd: int, beta: float, metric=metric) -> float:
                row = require_one(
                    rows,
                    model=model,
                    layer_set=lk,
                    direction_sample_size=nd,
                    beta=beta,
                )
                return percent(as_float(row[metric]))

            written = make_metric_views(
                plots_root=out_root,
                display=display,
                layer_sets=layer_sets,
                metric_stem=stem,
                metric_title=metric_title,
                ylabel=ylabel,
                baseline_value=baseline_value,
                steered_value=steered_value,
                ylim_pad=3.0,
            )
            for path in written:
                print("wrote", path.relative_to(analysis_dir))


def make_pope_plots(analysis_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    rows = tables["discriminative_counts_and_accuracy_by_gold_label"]
    for model, layer_sets in LAYER_SETS_BY_MODEL.items():
        display = MODEL_DISPLAY[model]
        for split in POPE_SPLITS:
            benchmark = f"pope_{split}"
            plots_root = analysis_dir / f"{MODEL_OUT_STEM[model]}_pope_results" / "plots" / split
            baseline = require_one(rows, model=model, benchmark=benchmark, layer_set="baseline")
            baseline_acc = percent(as_float(baseline["accuracy"]))

            def steered_acc(lk: str, nd: int, beta: float, benchmark=benchmark) -> float:
                row = require_one(
                    rows,
                    model=model,
                    benchmark=benchmark,
                    layer_set=lk,
                    direction_sample_size=nd,
                    beta=beta,
                )
                return percent(as_float(row["accuracy"]))

            written = make_metric_views(
                plots_root=plots_root,
                display=display,
                layer_sets=layer_sets,
                metric_stem="accuracy",
                metric_title=f"POPE {split} accuracy",
                ylabel="accuracy (%)",
                baseline_value=baseline_acc,
                steered_value=steered_acc,
                ylim_pad=3.0,
            )
            for path in written:
                print("wrote", path.relative_to(analysis_dir))

            baseline_no = percent(as_float(baseline["accuracy_gold_no"]))
            baseline_yes = percent(as_float(baseline["accuracy_gold_yes"]))
            n_gold_no = int(baseline["n_gold_no"])
            n_gold_yes = int(baseline["n_gold_yes"])

            def gold_at(lk: str, nd: int, beta: float, benchmark=benchmark) -> tuple[float, float]:
                row = require_one(
                    rows,
                    model=model,
                    benchmark=benchmark,
                    layer_set=lk,
                    direction_sample_size=nd,
                    beta=beta,
                )
                return (
                    percent(as_float(row["accuracy_gold_no"])),
                    percent(as_float(row["accuracy_gold_yes"])),
                )

            gold_written = make_gold_label_views(
                plots_root=plots_root,
                model=model,
                display=display,
                layer_sets=layer_sets,
                benchmark_label=f"POPE {split}",
                baseline_no=baseline_no,
                baseline_yes=baseline_yes,
                n_gold_no=n_gold_no,
                n_gold_yes=n_gold_yes,
                gold_at=gold_at,
            )
            for path in gold_written:
                print("wrote", path.relative_to(analysis_dir))


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.output_dir) / args.run_date / ANALYSIS_DIR_NAME
    payload = load_json(analysis_dir / "result_tables.json")
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported result_tables.json schema_version: {payload.get('schema_version')}")
    tables = payload["tables"]

    make_chair_plots(analysis_dir, tables)
    make_pope_plots(analysis_dir, tables)

    print("\n=== organized plot trees ===")
    for name in (
        "llava_chair_results",
        "qwen_chair_results",
        "llava_pope_results",
        "qwen_pope_results",
    ):
        base = analysis_dir / name / "plots"
        print(name)
        for path in sorted(base.rglob("*.png")):
            print(" ", path.relative_to(base))


if __name__ == "__main__":
    main()
