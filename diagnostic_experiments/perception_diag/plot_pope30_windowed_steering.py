#!/usr/bin/env python3
"""Plot POPE-30 windowed-steering results (accuracy, P(yes), flips).

Supports LLaVA-1.5 and Qwen2.5-VL. One PNG per (metric × intervention method),
always with no-intervention baseline drawn as grey hatched bars on the same
x-axis. For rotation @ layer, β=0.9 is omitted (degenerate generations).

Outputs land under model-named subdirectories::

    windowed_steering_summary/plots/llava-1.5-7b-hf/
    windowed_steering_summary/plots/qwen2.5-vl-7b-instruct/

Also can write a joint 2×2 mean-P(yes) figure (model × mlp method).

Plan: implementation_plans/pope30_windowed_steering_plots_plan_2026-07-21.md
      (+ Alex feedback 2026-07-21)

Usage::

    python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \\
      --model llava-1.5-7b-hf
    python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \\
      --model qwen2.5-vl-7b-instruct
    python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \\
      --joint_mean_p_yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from diagnostic_experiments.perception_diag.build_windowed_steering_summary import (  # noqa: E402
    STRENGTHS,
    expected_cell_ids,
    load_manifest,
)
from src.paths import perception_dump_dir, project_root  # noqa: E402
from src.prompt_spans import layer_windows  # noqa: E402

MODEL_SPECS = {
    "llava-1.5-7b-hf": {
        "display_name": "LLaVA-1.5",
        "num_layers": 32,
        "plots_subdir": "llava-1.5-7b-hf",
    },
    "qwen2.5-vl-7b-instruct": {
        "display_name": "Qwen2.5-VL",
        "num_layers": 28,
        "plots_subdir": "qwen2.5-vl-7b-instruct",
    },
}
CONDITIONS = ("neutral", "assertive_toward_no")
CONDITION_LABELS = {
    "neutral": "neutral question",
    "assertive_toward_no": "leading clause toward no",
}
CONFIG_ORDER = (
    ("rotation_mlp", "rotation @ mlp", "rotation_mlp"),
    ("rotation_layer", "rotation @ layer", "rotation_layer"),
    ("additive_mlp", "additive @ mlp", "additive_mlp"),
)
# rotation @ layer β=0.9 omitted: many cells are degenerate but still parse as yes/no
BETAS_BY_PREFIX = {
    "rotation_mlp": list(STRENGTHS),
    "rotation_layer": [0.2, 0.5],
    "additive_mlp": list(STRENGTHS),
}
BETA_COLORS = {0.2: "#4C78A8", 0.5: "#F58518", 0.9: "#54A24B"}
BASELINE_COLOR = "#9E9E9E"
BASELINE_EDGE = "#424242"
N_ITEMS = 30
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def window_labels(num_layers: int) -> List[str]:
    wins = layer_windows(num_layers, width=10, stride=5)
    return [f"{s}-{e}" for s, e in wins] + ["all layers"]


def window_display_positions(labels: Sequence[str]) -> np.ndarray:
    """x positions with a visual gap before 'all layers'."""
    pos = []
    x = 0.0
    for lab in labels:
        if lab == "all layers" and pos:
            x += 0.55
        pos.append(x)
        x += 1.0
    return np.asarray(pos, dtype=float)


def is_parseable(outcome: Optional[str]) -> bool:
    return outcome in ("yes", "no")


def collect_cell_metrics(
    steered_root: Path,
    baseline_man: Dict[Tuple[str, str], dict],
    cell_ids: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """cell_id -> condition_id -> metrics dict."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for cell_id in cell_ids:
        man = load_manifest(steered_root / cell_id)
        if not man:
            raise FileNotFoundError(
                f"missing/empty steered manifest: {steered_root / cell_id}"
            )
        out[cell_id] = {}
        for cond in CONDITIONS:
            item_ids = sorted({iid for iid, c in man if c == cond})
            rows = [man[(iid, cond)] for iid in item_ids]

            parseable = [r for r in rows if is_parseable(r.get("parsed_outcome"))]
            n_unparseable = sum(
                1 for r in rows if not is_parseable(r.get("parsed_outcome"))
            )
            n_parseable = len(parseable)
            n_yes = sum(1 for r in parseable if r.get("parsed_outcome") == "yes")
            accuracy = (n_yes / n_parseable) if n_parseable else float("nan")

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
                if b_out == "yes" and s_out == "no":
                    flips += 1

            out[cell_id][cond] = {
                "n_items": len(rows),
                "n_parseable": n_parseable,
                "n_unparseable": n_unparseable,
                "n_yes_parseable": n_yes,
                "accuracy": accuracy,
                "p_yes_raw": p_yes,
                "mean_p_yes_raw": float(np.mean(p_yes)) if len(p_yes) else float("nan"),
                "flips_yes_to_no": flips,
                "flip_pairs_excluded": excluded,
            }
    return out


def baseline_condition_stats(
    baseline_man: Dict[Tuple[str, str], dict],
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for cond in CONDITIONS:
        rows = [r for (iid, c), r in baseline_man.items() if c == cond]
        n = len(rows)
        n_parseable = sum(1 for r in rows if is_parseable(r.get("parsed_outcome")))
        n_yes = sum(1 for r in rows if r.get("parsed_outcome") == "yes")
        n_unparseable = n - n_parseable
        p_yes = np.asarray(
            [float(r["score_p_yes_raw"]) for r in rows], dtype=float
        )
        parseable = [r for r in rows if is_parseable(r.get("parsed_outcome"))]
        accuracy = (
            sum(1 for r in parseable if r.get("parsed_outcome") == "yes") / n_parseable
            if n_parseable
            else float("nan")
        )
        stats[cond] = {
            "n_items": n,
            "n_parseable": n_parseable,
            "n_unparseable": n_unparseable,
            "n_yes": n_yes,
            "accuracy": accuracy,
            "mean_p_yes_raw": float(np.mean(p_yes)) if len(p_yes) else float("nan"),
            "p_yes_raw": p_yes,
        }
    return stats


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return mean, lo, hi


def _cell_id_for(prefix: str, beta: float, window_label: str) -> str:
    if window_label == "all layers":
        return f"{prefix}_{beta}_layers_all"
    start, end = window_label.split("-")
    return f"{prefix}_{beta}_layers_{start}_{end}"


def _series_width(n_series: int) -> float:
    if n_series <= 2:
        return 0.28
    if n_series == 3:
        return 0.22
    return 0.18


def _draw_grouped_bars(
    ax: plt.Axes,
    *,
    betas: Sequence[float],
    window_labs: Sequence[str],
    x_pos: np.ndarray,
    bar_values: Dict[float, Sequence[float]],
    bar_errs: Optional[Dict[float, Tuple[Sequence[float], Sequence[float]]]] = None,
    annotations: Optional[Dict[float, Sequence[Optional[str]]]] = None,
    full_height_notes: Optional[Dict[float, Sequence[Optional[str]]]] = None,
    baseline_y: Optional[float] = None,
    baseline_bars: Optional[Sequence[float]] = None,
    baseline_bar_errs: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    ylabel: str,
    title: str,
    ylim: Optional[Tuple[float, float]] = None,
    integer_y: bool = False,
    annotate_fontsize: float = 7.0,
) -> None:
    """Draw beta bars; optional dashed baseline line and/or per-window baseline bars."""
    n_series = len(betas) + (1 if baseline_bars is not None else 0)
    width = _series_width(n_series)
    # Order within each window: [baseline,] then betas
    series_keys: List[Tuple[str, Optional[float]]] = []
    if baseline_bars is not None:
        series_keys.append(("baseline", None))
    for b in betas:
        series_keys.append(("beta", float(b)))
    offsets = {
        key: (i - (n_series - 1) / 2) * width for i, key in enumerate(series_keys)
    }

    if baseline_bars is not None:
        xs = x_pos + offsets[("baseline", None)]
        plot_vals = [
            np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else v
            for v in baseline_bars
        ]
        yerr = None
        if baseline_bar_errs is not None:
            lo, hi = baseline_bar_errs
            means = np.asarray(plot_vals, dtype=float)
            lo_a = np.asarray(lo, dtype=float)
            hi_a = np.asarray(hi, dtype=float)
            yerr = np.vstack([means - lo_a, hi_a - means])
            yerr = np.where(np.isfinite(yerr), yerr, 0.0)
        ax.bar(
            xs,
            plot_vals,
            width=width,
            color=BASELINE_COLOR,
            edgecolor=BASELINE_EDGE,
            linewidth=0.5,
            hatch="//",
            label="no-intervention baseline",
            yerr=yerr,
            error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#333333"},
            zorder=3,
        )

    for beta in betas:
        vals = list(bar_values[beta])
        xs = x_pos + offsets[("beta", float(beta))]
        plot_vals = [
            np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else v
            for v in vals
        ]
        yerr = None
        if bar_errs is not None:
            lo, hi = bar_errs[beta]
            means = np.asarray(plot_vals, dtype=float)
            lo_a = np.asarray(lo, dtype=float)
            hi_a = np.asarray(hi, dtype=float)
            yerr = np.vstack([means - lo_a, hi_a - means])
            yerr = np.where(np.isfinite(yerr), yerr, 0.0)

        bars = ax.bar(
            xs,
            plot_vals,
            width=width,
            color=BETA_COLORS[beta],
            edgecolor="black",
            linewidth=0.4,
            label=f"β={beta}",
            yerr=yerr,
            error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#333333"},
            zorder=3,
        )

        if annotations is not None:
            for rect, note in zip(bars, annotations[beta]):
                if not note:
                    continue
                h = rect.get_height()
                if not np.isfinite(h):
                    continue
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    h,
                    note,
                    ha="center",
                    va="bottom",
                    fontsize=annotate_fontsize,
                    rotation=90,
                    clip_on=False,
                )

        if full_height_notes is not None:
            y_top = ylim[1] if ylim is not None else ax.get_ylim()[1]
            for x, note, v in zip(xs, full_height_notes[beta], plot_vals):
                if not note:
                    continue
                if v is not None and np.isfinite(v):
                    continue
                ax.text(
                    x,
                    0.55 * y_top,
                    note,
                    ha="center",
                    va="center",
                    fontsize=annotate_fontsize,
                    rotation=90,
                    color="#444444",
                    clip_on=False,
                )

    # Dashed reference line only when baseline is NOT already drawn as bars
    if (
        baseline_bars is None
        and baseline_y is not None
        and np.isfinite(baseline_y)
    ):
        ax.axhline(
            baseline_y,
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            zorder=2,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(window_labs), rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7, zorder=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if integer_y:
        ymax = (
            ylim[1]
            if ylim is not None
            else max(
                (
                    v
                    for vals in bar_values.values()
                    for v in vals
                    if v is not None and np.isfinite(v)
                ),
                default=1,
            )
        )
        ax.set_yticks(list(range(0, int(np.ceil(ymax)) + 1)))


def _legend_handles(
    betas: Sequence[float],
    *,
    include_baseline_line: bool,
    include_baseline_bar: bool,
) -> Tuple[List[Any], List[str]]:
    handles: List[Any] = []
    labels: List[str] = []
    if include_baseline_bar:
        handles.append(
            Patch(
                facecolor=BASELINE_COLOR,
                edgecolor=BASELINE_EDGE,
                hatch="//",
                label="no-intervention baseline",
            )
        )
        labels.append("no-intervention baseline")
    for beta in betas:
        handles.append(Patch(facecolor=BETA_COLORS[beta], edgecolor="black", linewidth=0.4))
        labels.append(f"β={beta}")
    if include_baseline_line:
        handles.append(
            Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.2)
        )
        labels.append("no-intervention baseline")
    return handles, labels


def _shared_legend(
    fig: plt.Figure,
    betas: Sequence[float],
    *,
    include_baseline_line: bool,
    include_baseline_bar: bool,
) -> None:
    """Legend below the title with enough spacing so line/bar keys do not overlap."""
    handles, labels = _legend_handles(
        betas,
        include_baseline_line=include_baseline_line,
        include_baseline_bar=include_baseline_bar,
    )
    # Two rows when both baseline and betas are present: betas on row 1, baseline alone on row 2
    if include_baseline_line or include_baseline_bar:
        beta_h = handles[1:] if include_baseline_bar else handles[:-1]
        beta_l = labels[1:] if include_baseline_bar else labels[:-1]
        base_h = [handles[0]] if include_baseline_bar else [handles[-1]]
        base_l = [labels[0]] if include_baseline_bar else [labels[-1]]
        # Place betas then baseline beneath, centered
        leg1 = fig.legend(
            beta_h,
            beta_l,
            loc="upper center",
            ncol=len(beta_h),
            fontsize=9,
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            columnspacing=1.4,
            handletextpad=0.6,
            bbox_to_anchor=(0.5, 0.955),
        )
        fig.add_artist(leg1)
        fig.legend(
            base_h,
            base_l,
            loc="upper center",
            ncol=1,
            fontsize=9,
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            bbox_to_anchor=(0.5, 0.905),
        )
    else:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            fontsize=9,
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            columnspacing=1.4,
            bbox_to_anchor=(0.5, 0.94),
        )


def _method_fig(
    *,
    method_title: str,
    suptitle: str,
    caption: str,
    figsize: Tuple[float, float] = (8.5, 8.2),
) -> Tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(f"{suptitle}\n{method_title}", fontsize=12, y=1.02)
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8)
    return fig, axes


def _omit_note(prefix: str) -> str:
    if prefix == "rotation_layer":
        return " β=0.9 omitted (degenerate generations)."
    return ""


def plot_accuracy_method(
    metrics: Dict[str, Dict[str, Dict[str, Any]]],
    baseline_stats: Dict[str, Dict[str, Any]],
    *,
    prefix: str,
    method_title: str,
    betas: Sequence[float],
    out_path: Path,
    baseline_as_bars: bool,
    model_title: str,
    num_layers: int,
) -> None:
    wlabs = window_labels(num_layers)
    xpos = window_display_positions(wlabs)
    mode_note = (
        " Grey hatched bars = no-intervention baseline (same value at every window)."
        if baseline_as_bars
        else " Dashed line = no-intervention baseline."
    )
    fig, axes = _method_fig(
        method_title=method_title,
        suptitle=(
            f"{model_title}, POPE-30 (all gold=yes): accuracy by steering layer window"
        ),
        caption=(
            "n=30 items per cell-condition. Accuracy = fraction of parseable responses "
            "answered yes (all items are gold=yes). Annotations: u=k = k unparseable; "
            "“all unparseable” when no bar."
            + mode_note
            + _omit_note(prefix)
        ),
    )

    for row, cond in enumerate(CONDITIONS):
        ax = axes[row]
        bar_values: Dict[float, List[float]] = {b: [] for b in betas}
        annotations: Dict[float, List[Optional[str]]] = {b: [] for b in betas}
        full_notes: Dict[float, List[Optional[str]]] = {b: [] for b in betas}
        for wlab in wlabs:
            for beta in betas:
                cell = _cell_id_for(prefix, beta, wlab)
                m = metrics[cell][cond]
                if m["n_parseable"] == 0:
                    bar_values[beta].append(float("nan"))
                    annotations[beta].append(None)
                    full_notes[beta].append("all unparseable")
                else:
                    bar_values[beta].append(m["accuracy"])
                    note = f"u={m['n_unparseable']}" if m["n_unparseable"] else None
                    annotations[beta].append(note)
                    full_notes[beta].append(None)

        base_acc = baseline_stats[cond]["accuracy"]
        baseline_bars = [base_acc] * len(wlabs) if baseline_as_bars else None
        _draw_grouped_bars(
            ax,
            betas=betas,
            window_labs=wlabs,
            x_pos=xpos,
            bar_values=bar_values,
            annotations=annotations,
            full_height_notes=full_notes,
            baseline_y=None if baseline_as_bars else base_acc,
            baseline_bars=baseline_bars,
            ylabel="Accuracy",
            title=CONDITION_LABELS[cond],
            ylim=(0.0, 1.15),
        )

    _shared_legend(
        fig,
        betas,
        include_baseline_line=not baseline_as_bars,
        include_baseline_bar=baseline_as_bars,
    )
    fig.tight_layout(rect=[0.02, 0.05, 1.0, 0.86])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_mean_p_yes_raw_method(
    metrics: Dict[str, Dict[str, Dict[str, Any]]],
    baseline_stats: Dict[str, Dict[str, Any]],
    *,
    prefix: str,
    method_title: str,
    betas: Sequence[float],
    out_path: Path,
    baseline_as_bars: bool,
    model_title: str,
    num_layers: int,
) -> None:
    wlabs = window_labels(num_layers)
    xpos = window_display_positions(wlabs)
    mode_note = (
        " Grey hatched bars = no-intervention baseline (same value at every window)."
        if baseline_as_bars
        else " Dashed line = no-intervention baseline."
    )
    fig, axes = _method_fig(
        method_title=method_title,
        figsize=(8.5, 8.4),
        suptitle=(
            f"{model_title}, POPE-30: mean unconditional P(yes) at first answer token, "
            "by steering layer window"
        ),
        caption=(
            "n=30 items per cell-condition. Mean of score_p_yes_raw over all items "
            "(scores exist regardless of parseability). Error bars: bootstrap 95% CI "
            f"(n={BOOTSTRAP_N} resamples over items)."
            + mode_note
            + _omit_note(prefix)
        ),
    )

    for row, cond in enumerate(CONDITIONS):
        ax = axes[row]
        base_mean, base_lo, base_hi = bootstrap_mean_ci(
            baseline_stats[cond]["p_yes_raw"]
        )
        bar_values: Dict[float, List[float]] = {b: [] for b in betas}
        bar_errs: Dict[float, Tuple[List[float], List[float]]] = {
            b: ([], []) for b in betas
        }
        for wlab in wlabs:
            for beta in betas:
                cell = _cell_id_for(prefix, beta, wlab)
                mean, lo, hi = bootstrap_mean_ci(metrics[cell][cond]["p_yes_raw"])
                bar_values[beta].append(mean)
                bar_errs[beta][0].append(lo)
                bar_errs[beta][1].append(hi)

        baseline_bars = [base_mean] * len(wlabs) if baseline_as_bars else None
        baseline_bar_errs = (
            ([base_lo] * len(wlabs), [base_hi] * len(wlabs))
            if baseline_as_bars
            else None
        )
        _draw_grouped_bars(
            ax,
            betas=betas,
            window_labs=wlabs,
            x_pos=xpos,
            bar_values=bar_values,
            bar_errs=bar_errs,
            baseline_y=None if baseline_as_bars else base_mean,
            baseline_bars=baseline_bars,
            baseline_bar_errs=baseline_bar_errs,
            ylabel="Mean unconditional P(yes)",
            title=CONDITION_LABELS[cond],
            ylim=(0.0, 1.05),
        )

    _shared_legend(
        fig,
        betas,
        include_baseline_line=not baseline_as_bars,
        include_baseline_bar=baseline_as_bars,
    )
    fig.tight_layout(rect=[0.02, 0.05, 1.0, 0.86])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_flips_method(
    metrics: Dict[str, Dict[str, Dict[str, Any]]],
    baseline_stats: Dict[str, Dict[str, Any]],
    *,
    prefix: str,
    method_title: str,
    betas: Sequence[float],
    out_path: Path,
    baseline_as_bars: bool,
    model_title: str,
    num_layers: int,
) -> None:
    wlabs = window_labels(num_layers)
    xpos = window_display_positions(wlabs)
    mode_note = (
        " Grey hatched bars = no-intervention baseline (0 flips by definition)."
        if baseline_as_bars
        else ""
    )
    fig, axes = _method_fig(
        method_title=method_title,
        figsize=(8.5, 8.6),
        suptitle=(
            f"{model_title}, POPE-30: items flipped from baseline yes to no, "
            "by steering layer window"
        ),
        caption=(
            "n=30 items per cell-condition. Flip = baseline parsed_outcome=yes and "
            "steered=no for the same item_id and condition_id. Pairs with either side "
            "unparseable are excluded (x=k excl when some pairs remain). Fully "
            "unparseable cells: no bar, labeled “all unparseable”."
            + mode_note
            + _omit_note(prefix)
        ),
    )

    for row, cond in enumerate(CONDITIONS):
        ax = axes[row]
        bs = baseline_stats[cond]
        denom_subtitle = (
            f"{CONDITION_LABELS[cond]}  ·  baseline: {bs['n_yes']}/{bs['n_items']} "
            f"parsed yes"
        )
        bar_values: Dict[float, List[float]] = {b: [] for b in betas}
        annotations: Dict[float, List[Optional[str]]] = {b: [] for b in betas}
        full_notes: Dict[float, List[Optional[str]]] = {b: [] for b in betas}
        for wlab in wlabs:
            for beta in betas:
                cell = _cell_id_for(prefix, beta, wlab)
                m = metrics[cell][cond]
                if m["n_parseable"] == 0:
                    bar_values[beta].append(float("nan"))
                    annotations[beta].append(None)
                    full_notes[beta].append("all unparseable")
                else:
                    bar_values[beta].append(float(m["flips_yes_to_no"]))
                    excl = m["flip_pairs_excluded"]
                    annotations[beta].append(f"x={excl} excl" if excl else None)
                    full_notes[beta].append(None)

        baseline_bars = [0.0] * len(wlabs) if baseline_as_bars else None
        _draw_grouped_bars(
            ax,
            betas=betas,
            window_labs=wlabs,
            x_pos=xpos,
            bar_values=bar_values,
            annotations=annotations,
            full_height_notes=full_notes,
            baseline_y=None,
            baseline_bars=baseline_bars,
            ylabel="Items flipped yes→no",
            title=denom_subtitle,
            ylim=(0, N_ITEMS + 2),
            integer_y=True,
        )

    _shared_legend(
        fig,
        betas,
        include_baseline_line=False,
        include_baseline_bar=baseline_as_bars,
    )
    fig.tight_layout(rect=[0.02, 0.06, 1.0, 0.86 if baseline_as_bars else 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_plot_set(
    *,
    out_dir: Path,
    metrics: Dict[str, Dict[str, Dict[str, Any]]],
    baseline_stats: Dict[str, Dict[str, Any]],
    model_title: str,
    num_layers: int,
) -> List[Path]:
    """Write per-method accuracy / P(yes) / flips plots (baseline bars always on)."""
    written: List[Path] = []
    for prefix, method_title, slug in CONFIG_ORDER:
        betas = BETAS_BY_PREFIX[prefix]
        paths = {
            "accuracy": out_dir / f"pope30_accuracy_by_layer_window_{slug}.png",
            "p_yes": out_dir / f"pope30_mean_p_yes_raw_by_layer_window_{slug}.png",
            "flips": out_dir
            / f"pope30_flips_from_baseline_yes_by_layer_window_{slug}.png",
        }
        plot_accuracy_method(
            metrics,
            baseline_stats,
            prefix=prefix,
            method_title=method_title,
            betas=betas,
            out_path=paths["accuracy"],
            baseline_as_bars=True,
            model_title=model_title,
            num_layers=num_layers,
        )
        plot_mean_p_yes_raw_method(
            metrics,
            baseline_stats,
            prefix=prefix,
            method_title=method_title,
            betas=betas,
            out_path=paths["p_yes"],
            baseline_as_bars=True,
            model_title=model_title,
            num_layers=num_layers,
        )
        plot_flips_method(
            metrics,
            baseline_stats,
            prefix=prefix,
            method_title=method_title,
            betas=betas,
            out_path=paths["flips"],
            baseline_as_bars=True,
            model_title=model_title,
            num_layers=num_layers,
        )
        written.extend(paths.values())
    return written


def _load_model_bundle(model_short: str) -> Dict[str, Any]:
    spec = MODEL_SPECS[model_short]
    num_layers = int(spec["num_layers"])
    steered_root = perception_dump_dir("pope", model_short, "pope30_windowed_steering")
    baseline_dir = (
        perception_dump_dir("pope", model_short, "pope30_existence_yes_baseline")
        / "baseline"
    )
    cell_ids = expected_cell_ids(num_layers)
    missing = [
        c for c in cell_ids if not (steered_root / c / "manifest.jsonl").exists()
    ]
    if missing:
        raise SystemExit(
            f"{model_short}: missing {len(missing)} steered cells, e.g. {missing[:3]}"
        )
    baseline_man = load_manifest(baseline_dir)
    if not baseline_man:
        raise SystemExit(f"Empty baseline manifest at {baseline_dir}")
    baseline_man = {k: v for k, v in baseline_man.items() if k[1] in CONDITIONS}
    baseline_stats = baseline_condition_stats(baseline_man)
    metrics = collect_cell_metrics(steered_root, baseline_man, cell_ids)
    return {
        "model_short": model_short,
        "display_name": spec["display_name"],
        "num_layers": num_layers,
        "metrics": metrics,
        "baseline_stats": baseline_stats,
    }


def plot_joint_mean_p_yes_raw_model_by_mlp(
    bundles: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    """2x2: rows = model, cols = additive @ mlp | rotation @ mlp.

    Each pane stacks the two prompt conditions (neutral / leading toward no).
    """
    mlp_cols = (
        ("additive_mlp", "additive @ mlp"),
        ("rotation_mlp", "rotation @ mlp"),
    )
    n_models = len(bundles)
    n_conds = len(CONDITIONS)
    fig = plt.figure(figsize=(16.5, 3.6 * n_models * n_conds + 1.4))
    outer = fig.add_gridspec(
        n_models,
        2,
        hspace=0.32,
        wspace=0.16,
        left=0.06,
        right=0.98,
        top=0.90,
        bottom=0.05,
    )
    fig.suptitle(
        "POPE-30: mean unconditional P(yes) at first answer token, "
        "by steering layer window\n"
        "Rows = model · columns = MLP intervention "
        "(baseline = grey hatched bars)",
        fontsize=13,
        y=0.985,
    )

    for row, bundle in enumerate(bundles):
        metrics = bundle["metrics"]
        baseline_stats = bundle["baseline_stats"]
        num_layers = bundle["num_layers"]
        model_title = bundle["display_name"]
        wlabs = window_labels(num_layers)
        xpos = window_display_positions(wlabs)
        betas = BETAS_BY_PREFIX["additive_mlp"]

        for col, (prefix, method_title) in enumerate(mlp_cols):
            inner = outer[row, col].subgridspec(n_conds, 1, hspace=0.38)
            for crow, cond in enumerate(CONDITIONS):
                ax = fig.add_subplot(inner[crow, 0])
                base_mean, base_lo, base_hi = bootstrap_mean_ci(
                    baseline_stats[cond]["p_yes_raw"]
                )
                bar_values: Dict[float, List[float]] = {b: [] for b in betas}
                bar_errs: Dict[float, Tuple[List[float], List[float]]] = {
                    b: ([], []) for b in betas
                }
                for wlab in wlabs:
                    for beta in betas:
                        cell = _cell_id_for(prefix, beta, wlab)
                        mean, lo, hi = bootstrap_mean_ci(
                            metrics[cell][cond]["p_yes_raw"]
                        )
                        bar_values[beta].append(mean)
                        bar_errs[beta][0].append(lo)
                        bar_errs[beta][1].append(hi)

                pane_title = (
                    f"{model_title}  ·  {method_title}  ·  "
                    f"{CONDITION_LABELS[cond]}"
                )
                _draw_grouped_bars(
                    ax,
                    betas=betas,
                    window_labs=wlabs,
                    x_pos=xpos,
                    bar_values=bar_values,
                    bar_errs=bar_errs,
                    baseline_y=None,
                    baseline_bars=[base_mean] * len(wlabs),
                    baseline_bar_errs=(
                        [base_lo] * len(wlabs),
                        [base_hi] * len(wlabs),
                    ),
                    ylabel="Mean P(yes)" if col == 0 else "",
                    title=pane_title,
                    ylim=(0.0, 1.05),
                )
                if crow < n_conds - 1:
                    ax.set_xticklabels([])

    handles, labels = _legend_handles(
        BETAS_BY_PREFIX["additive_mlp"],
        include_baseline_line=False,
        include_baseline_bar=True,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.text(
        0.5,
        0.008,
        "n=30 items per cell-condition. Mean of score_p_yes_raw over all items; "
        f"error bars = bootstrap 95% CI (n={BOOTSTRAP_N}). "
        "Grey hatched = no-intervention baseline (repeated at each window). "
        "Layer windows differ by model depth (LLaVA 32 vs Qwen 28).",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS.keys()),
        default=None,
        help="Model short name; omit when only writing the joint figure",
    )
    parser.add_argument(
        "--joint_mean_p_yes",
        action="store_true",
        help="Write 2x2 joint mean-P(yes) PNG (LLaVA/Qwen x additive/rotation mlp)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output plot directory for --model (default: plots/<model_short>/)",
    )
    parser.add_argument(
        "--joint_out",
        type=Path,
        default=None,
        help="Path for joint PNG (default under plots/)",
    )
    args = parser.parse_args(argv)

    if args.model is None and not args.joint_mean_p_yes:
        parser.error("pass --model and/or --joint_mean_p_yes")

    root = project_root()
    plots_root = (
        root
        / "diagnostic_experiments"
        / "perception_diag"
        / "windowed_steering_summary"
        / "plots"
    )
    written: List[Path] = []

    if args.model is not None:
        bundle = _load_model_bundle(args.model)
        for cond in CONDITIONS:
            bs = bundle["baseline_stats"][cond]
            print(
                f"{args.model} baseline[{cond}]: n={bs['n_items']} "
                f"parseable={bs['n_parseable']} yes={bs['n_yes']} "
                f"accuracy={bs['accuracy']:.3f} "
                f"mean_p_yes_raw={bs['mean_p_yes_raw']:.4f}"
            )
        out_dir = args.out_dir or (
            plots_root / str(MODEL_SPECS[args.model]["plots_subdir"])
        )
        written.extend(
            _write_plot_set(
                out_dir=out_dir,
                metrics=bundle["metrics"],
                baseline_stats=bundle["baseline_stats"],
                model_title=bundle["display_name"],
                num_layers=bundle["num_layers"],
            )
        )

    if args.joint_mean_p_yes:
        bundles = [
            _load_model_bundle("llava-1.5-7b-hf"),
            _load_model_bundle("qwen2.5-vl-7b-instruct"),
        ]
        joint_out = args.joint_out or (
            plots_root
            / "pope30_mean_p_yes_raw_by_layer_window_"
            "llava_vs_qwen_additive_vs_rotation_mlp.png"
        )
        plot_joint_mean_p_yes_raw_model_by_mlp(bundles, joint_out)
        written.append(joint_out)

    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
