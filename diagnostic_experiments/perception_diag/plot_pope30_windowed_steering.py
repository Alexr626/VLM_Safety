#!/usr/bin/env python3
"""Plot POPE-30 and AMBER-100 windowed-steering results (accuracy, P(token), flips).

Supports LLaVA-1.5 and Qwen2.5-VL. One PNG per (metric × intervention method),
always with no-intervention baseline drawn as grey hatched bars on the same
x-axis. For rotation @ layer, β=0.9 is omitted (degenerate generations).

Mean first-token probability: P(yes) for gold=yes subsets, P(no) for gold=no.

Outputs land under model × dataset subdirectories::

    windowed_steering_summary/plots/llava-1.5-7b-hf/pope30_yes/
    windowed_steering_summary/plots/llava-1.5-7b-hf/amber100_attribute_yes/
    …

Gold=yes: accuracy = fraction parsed yes; flips = baseline yes → steered no.
Gold=no: accuracy = fraction parsed no; flips = baseline no → steered yes.

Usage::

    python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \\
      --model llava-1.5-7b-hf --datasets pope30_yes pope30_no
    python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \\
      --model llava-1.5-7b-hf --datasets \\
      amber100_attribute_yes amber100_attribute_no \\
      amber100_relation_yes amber100_relation_no
    python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \\
      --joint_amber_llava
    python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \\
      --joint_pope_llava
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


def _amber100_subset_spec(
    *,
    key: str,
    qtype: str,
    gold: str,
) -> dict:
    """Build DATASET_SPECS entry for an AMBER-100 (qtype, gold) filter."""
    if gold == "yes":
        conditions = ("neutral", "assertive_toward_no")
        condition_labels = {
            "neutral": "neutral question",
            "assertive_toward_no": "leading clause toward no",
        }
        correct_label = "yes"
        flip_from, flip_to = "yes", "no"
        flip_key = "flips_yes_to_no"
        prob_token = "yes"
        prob_field = "score_p_yes_raw"
        accuracy_phrase = (
            f"Accuracy = fraction of parseable responses answered yes "
            f"(AMBER-100 {qtype}, gold=yes; n=20)."
        )
        flip_title = "items flipped from baseline yes to no"
        flip_ylabel = "Items flipped yes→no"
        flip_caption = (
            "Flip = baseline parsed_outcome=yes and steered=no for the same "
            "item_id and condition_id."
        )
    else:
        conditions = ("neutral", "assertive_toward_yes")
        condition_labels = {
            "neutral": "neutral question",
            "assertive_toward_yes": "leading clause toward yes",
        }
        correct_label = "no"
        flip_from, flip_to = "no", "yes"
        flip_key = "flips_no_to_yes"
        prob_token = "no"
        prob_field = "score_p_no_raw"
        accuracy_phrase = (
            f"Accuracy = fraction of parseable responses answered no "
            f"(AMBER-100 {qtype}, gold=no; n=20)."
        )
        flip_title = "items flipped from baseline no to yes"
        flip_ylabel = "Items flipped no→yes"
        flip_caption = (
            "Flip = baseline parsed_outcome=no and steered=yes for the same "
            "item_id and condition_id."
        )
    return {
        "benchmark": "amber",
        "run_tag": "amber100_windowed_steering",
        "baseline_in_run_tag": False,
        "baseline_run_tag": "amber100_baseline",
        "qtype": qtype,
        "gold": gold,
        "n_items": 20,
        "conditions": conditions,
        "condition_labels": condition_labels,
        "correct_label": correct_label,
        "flip_from": flip_from,
        "flip_to": flip_to,
        "flip_key": flip_key,
        "dataset_title": f"AMBER-100 {qtype} (gold={gold})",
        "accuracy_phrase": accuracy_phrase,
        "flip_title": flip_title,
        "flip_ylabel": flip_ylabel,
        "flip_caption": flip_caption,
        "flip_filename_stem": f"{key}_flips_from_baseline_{flip_from}",
        "prob_score_field": prob_field,
        "prob_token_label": prob_token,
        "prob_filename_stem": f"{key}_mean_p_{prob_token}_raw",
        "accuracy_filename_stem": f"{key}_accuracy",
    }


DATASET_SPECS: Dict[str, dict] = {
    "pope30_yes": {
        "benchmark": "pope",
        "run_tag": "pope30_yes_windowed_steering",
        "baseline_in_run_tag": True,
        "qtype": None,
        "gold": "yes",
        "n_items": 30,
        "conditions": ("neutral", "assertive_toward_no"),
        "condition_labels": {
            "neutral": "neutral question",
            "assertive_toward_no": "leading clause toward no",
        },
        "correct_label": "yes",
        "flip_from": "yes",
        "flip_to": "no",
        "flip_key": "flips_yes_to_no",
        "dataset_title": "POPE-30-yes (all gold=yes)",
        "accuracy_phrase": (
            "Accuracy = fraction of parseable responses answered yes "
            "(all items are gold=yes)."
        ),
        "flip_title": "items flipped from baseline yes to no",
        "flip_ylabel": "Items flipped yes→no",
        "flip_caption": (
            "Flip = baseline parsed_outcome=yes and steered=no for the same "
            "item_id and condition_id."
        ),
        "flip_filename_stem": "pope30_flips_from_baseline_yes",
        "prob_score_field": "score_p_yes_raw",
        "prob_token_label": "yes",
        "prob_filename_stem": "pope30_mean_p_yes_raw",
        "accuracy_filename_stem": "pope30_accuracy",
    },
    "pope30_no": {
        "benchmark": "pope",
        "run_tag": "pope30_no_windowed_steering",
        "baseline_in_run_tag": True,
        "qtype": None,
        "gold": "no",
        "n_items": 30,
        "conditions": ("neutral", "assertive_toward_yes"),
        "condition_labels": {
            "neutral": "neutral question",
            "assertive_toward_yes": "leading clause toward yes",
        },
        "correct_label": "no",
        "flip_from": "no",
        "flip_to": "yes",
        "flip_key": "flips_no_to_yes",
        "dataset_title": "POPE-30-no (all gold=no)",
        "accuracy_phrase": (
            "Accuracy = fraction of parseable responses answered no "
            "(all items are gold=no)."
        ),
        "flip_title": "items flipped from baseline no to yes",
        "flip_ylabel": "Items flipped no→yes",
        "flip_caption": (
            "Flip = baseline parsed_outcome=no and steered=yes for the same "
            "item_id and condition_id."
        ),
        "flip_filename_stem": "pope30_flips_from_baseline_no",
        "prob_score_field": "score_p_no_raw",
        "prob_token_label": "no",
        "prob_filename_stem": "pope30_mean_p_no_raw",
        "accuracy_filename_stem": "pope30_accuracy",
    },
}
for _amber_key, _qtype, _gold in (
    ("amber100_attribute_yes", "attribute", "yes"),
    ("amber100_attribute_no", "attribute", "no"),
    ("amber100_relation_yes", "relation", "yes"),
    ("amber100_relation_no", "relation", "no"),
):
    DATASET_SPECS[_amber_key] = _amber100_subset_spec(
        key=_amber_key, qtype=_qtype, gold=_gold
    )

# LLaVA-only merged figures: gold=yes on top, gold=no on bottom; MLP methods as cols.
AMBER_LLAVA_CAPABILITIES: Dict[str, Tuple[str, str]] = {
    "attribute": ("amber100_attribute_yes", "amber100_attribute_no"),
    "relation": ("amber100_relation_yes", "amber100_relation_no"),
}

MLP_JOINT_COLS: Tuple[Tuple[str, str], ...] = (
    ("additive_mlp", "additive @ mlp"),
    ("rotation_mlp", "rotation @ mlp"),
)

# Legacy aliases used by joint figure / older call sites (yes-set only).
CONDITIONS = DATASET_SPECS["pope30_yes"]["conditions"]
CONDITION_LABELS = DATASET_SPECS["pope30_yes"]["condition_labels"]
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
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def flips_ylim_max(n_items: int, observed_max: float = 0.0) -> int:
    """Tighter flips y-axis: prefer round(n/4), but never clip bars."""
    preferred = max(1, int(round(n_items / 4.0)))
    if not np.isfinite(observed_max) or observed_max <= 0:
        return preferred
    needed = int(np.ceil(float(observed_max)))
    return max(preferred, needed)


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


def _row_matches_subset(row: dict, *, qtype: Optional[str], gold: Optional[str]) -> bool:
    if qtype is not None and row.get("qtype") != qtype:
        return False
    if gold is not None and row.get("gold") != gold:
        return False
    return True


def _filter_manifest(
    man: Dict[Tuple[str, str], dict],
    *,
    conditions: Sequence[str],
    qtype: Optional[str],
    gold: Optional[str],
) -> Dict[Tuple[str, str], dict]:
    return {
        k: v
        for k, v in man.items()
        if k[1] in conditions and _row_matches_subset(v, qtype=qtype, gold=gold)
    }


def _gate_item_counts(
    man: Dict[Tuple[str, str], dict],
    *,
    conditions: Sequence[str],
    n_items: int,
    label: str,
) -> None:
    for cond in conditions:
        ids = {iid for iid, c in man if c == cond}
        if len(ids) != n_items:
            raise SystemExit(
                f"{label}: condition {cond!r} has {len(ids)} items, expected {n_items}"
            )
    all_ids = {iid for iid, _c in man}
    if len(all_ids) != n_items:
        raise SystemExit(
            f"{label}: {len(all_ids)} distinct item_ids, expected {n_items}"
        )


def collect_cell_metrics(
    steered_root: Path,
    baseline_man: Dict[Tuple[str, str], dict],
    cell_ids: Sequence[str],
    *,
    conditions: Sequence[str],
    correct_label: str,
    flip_from: str,
    flip_to: str,
    flip_key: str,
    prob_score_field: str,
    qtype: Optional[str] = None,
    gold: Optional[str] = None,
    n_items: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """cell_id -> condition_id -> metrics dict (gold-aware accuracy / flips / P)."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for cell_id in cell_ids:
        man = load_manifest(steered_root / cell_id)
        if not man:
            raise FileNotFoundError(
                f"missing/empty steered manifest: {steered_root / cell_id}"
            )
        man = _filter_manifest(
            man, conditions=conditions, qtype=qtype, gold=gold
        )
        if n_items is not None:
            _gate_item_counts(
                man,
                conditions=conditions,
                n_items=n_items,
                label=f"steered cell {cell_id}",
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

            p_token = np.asarray(
                [float(r[prob_score_field]) for r in rows], dtype=float
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
                "p_token_raw": p_token,
                "mean_p_token_raw": (
                    float(np.mean(p_token)) if len(p_token) else float("nan")
                ),
                flip_key: flips,
                "flip_pairs_excluded": excluded,
            }
    return out


def baseline_condition_stats(
    baseline_man: Dict[Tuple[str, str], dict],
    *,
    conditions: Sequence[str],
    correct_label: str,
    flip_from: str,
    prob_score_field: str,
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
        p_token = np.asarray(
            [float(r[prob_score_field]) for r in rows], dtype=float
        )
        accuracy = (n_correct / n_parseable) if n_parseable else float("nan")
        stats[cond] = {
            "n_items": n,
            "n_parseable": n_parseable,
            "n_unparseable": n_unparseable,
            "n_correct": n_correct,
            "n_flip_from": n_flip_from,
            "accuracy": accuracy,
            "mean_p_token_raw": (
                float(np.mean(p_token)) if len(p_token) else float("nan")
            ),
            "p_token_raw": p_token,
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
        from matplotlib.ticker import MaxNLocator

        ax.yaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=8, min_n_ticks=3)
        )


def _legend_handles(
    betas: Sequence[float],
    *,
    include_baseline_line: bool,
    include_baseline_bar: bool,
    baseline_bar_label: str = "no-intervention baseline",
) -> Tuple[List[Any], List[str]]:
    handles: List[Any] = []
    labels: List[str] = []
    if include_baseline_bar:
        handles.append(
            Patch(
                facecolor=BASELINE_COLOR,
                edgecolor=BASELINE_EDGE,
                hatch="//",
                label=baseline_bar_label,
            )
        )
        labels.append(baseline_bar_label)
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
    baseline_bar_label: str = "no-intervention baseline",
) -> None:
    """Legend below the title with enough spacing so line/bar keys do not overlap."""
    handles, labels = _legend_handles(
        betas,
        include_baseline_line=include_baseline_line,
        include_baseline_bar=include_baseline_bar,
        baseline_bar_label=baseline_bar_label,
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
    dataset_cfg: dict,
) -> None:
    conditions = dataset_cfg["conditions"]
    condition_labels = dataset_cfg["condition_labels"]
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
            f"{model_title}, {dataset_cfg['dataset_title']}: "
            "accuracy by steering layer window"
        ),
        caption=(
            f"n={dataset_cfg['n_items']} items per cell-condition. "
            + dataset_cfg["accuracy_phrase"]
            + " Annotations: u=k = k unparseable; "
            "“all unparseable” when no bar."
            + mode_note
            + _omit_note(prefix)
        ),
    )

    for row, cond in enumerate(conditions):
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
            title=condition_labels[cond],
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


def plot_mean_token_prob_method(
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
    dataset_cfg: dict,
) -> None:
    conditions = dataset_cfg["conditions"]
    condition_labels = dataset_cfg["condition_labels"]
    token = dataset_cfg["prob_token_label"]
    score_field = dataset_cfg["prob_score_field"]
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
            f"{model_title}, {dataset_cfg['dataset_title']}: "
            f"mean unconditional P({token}) at first answer token, "
            "by steering layer window"
        ),
        caption=(
            f"n={dataset_cfg['n_items']} items per cell-condition. Mean of {score_field} over all items "
            "(scores exist regardless of parseability). Error bars: bootstrap 95% CI "
            f"(n={BOOTSTRAP_N} resamples over items)."
            + mode_note
            + _omit_note(prefix)
        ),
    )

    for row, cond in enumerate(conditions):
        ax = axes[row]
        base_mean, base_lo, base_hi = bootstrap_mean_ci(
            baseline_stats[cond]["p_token_raw"]
        )
        bar_values: Dict[float, List[float]] = {b: [] for b in betas}
        bar_errs: Dict[float, Tuple[List[float], List[float]]] = {
            b: ([], []) for b in betas
        }
        for wlab in wlabs:
            for beta in betas:
                cell = _cell_id_for(prefix, beta, wlab)
                mean, lo, hi = bootstrap_mean_ci(metrics[cell][cond]["p_token_raw"])
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
            ylabel=f"Mean unconditional P({token})",
            title=condition_labels[cond],
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


# Back-compat alias for older import sites.
plot_mean_p_yes_raw_method = plot_mean_token_prob_method


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
    dataset_cfg: dict,
) -> None:
    conditions = dataset_cfg["conditions"]
    condition_labels = dataset_cfg["condition_labels"]
    flip_key = dataset_cfg["flip_key"]
    flip_from = dataset_cfg["flip_from"]
    wlabs = window_labels(num_layers)
    xpos = window_display_positions(wlabs)
    fig, axes = _method_fig(
        method_title=method_title,
        figsize=(8.5, 8.6),
        suptitle=(
            f"{model_title}, {dataset_cfg['dataset_title']}: "
            f"{dataset_cfg['flip_title']}, by steering layer window"
        ),
        caption=(
            f"n={dataset_cfg['n_items']} items per cell-condition. "
            + dataset_cfg["flip_caption"]
            + " Bars = flips from the same-condition no-intervention baseline "
            "to the steered response. Pairs with either side unparseable are "
            "excluded (x=k excl when some pairs remain). Fully unparseable "
            "cells: no bar, labeled “all unparseable”."
            + _omit_note(prefix)
        ),
    )

    observed_max = 0.0
    for row, cond in enumerate(conditions):
        ax = axes[row]
        bs = baseline_stats[cond]
        denom_subtitle = (
            f"{condition_labels[cond]}  ·  baseline: "
            f"{bs['n_flip_from']}/{bs['n_items']} parsed {flip_from}"
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
                    val = float(m[flip_key])
                    bar_values[beta].append(val)
                    if np.isfinite(val):
                        observed_max = max(observed_max, val)
                    excl = m["flip_pairs_excluded"]
                    annotations[beta].append(f"x={excl} excl" if excl else None)
                    full_notes[beta].append(None)

        y_top = flips_ylim_max(int(dataset_cfg["n_items"]), observed_max)
        _draw_grouped_bars(
            ax,
            betas=betas,
            window_labs=wlabs,
            x_pos=xpos,
            bar_values=bar_values,
            annotations=annotations,
            full_height_notes=full_notes,
            baseline_y=None,
            baseline_bars=None,
            ylabel=dataset_cfg["flip_ylabel"],
            title=denom_subtitle,
            ylim=(0, y_top),
            integer_y=True,
        )

    # Re-apply shared ylim after both rows collected observed_max.
    y_top = flips_ylim_max(int(dataset_cfg["n_items"]), observed_max)
    for ax in axes:
        ax.set_ylim(0, y_top)

    _shared_legend(
        fig,
        betas,
        include_baseline_line=False,
        include_baseline_bar=False,
    )
    fig.tight_layout(rect=[0.02, 0.06, 1.0, 0.90])
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
    dataset_cfg: dict,
) -> List[Path]:
    """Write per-method accuracy / mean P(token) / flips plots (baseline bars always on)."""
    written: List[Path] = []
    flip_stem = dataset_cfg["flip_filename_stem"]
    prob_stem = dataset_cfg["prob_filename_stem"]
    acc_stem = dataset_cfg.get("accuracy_filename_stem", "pope30_accuracy")
    for prefix, method_title, slug in CONFIG_ORDER:
        betas = BETAS_BY_PREFIX[prefix]
        paths = {
            "accuracy": out_dir / f"{acc_stem}_by_layer_window_{slug}.png",
            "p_token": out_dir / f"{prob_stem}_by_layer_window_{slug}.png",
            "flips": out_dir / f"{flip_stem}_by_layer_window_{slug}.png",
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
            dataset_cfg=dataset_cfg,
        )
        plot_mean_token_prob_method(
            metrics,
            baseline_stats,
            prefix=prefix,
            method_title=method_title,
            betas=betas,
            out_path=paths["p_token"],
            baseline_as_bars=True,
            model_title=model_title,
            num_layers=num_layers,
            dataset_cfg=dataset_cfg,
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
            dataset_cfg=dataset_cfg,
        )
        written.extend(paths.values())
    return written


def _load_model_bundle(model_short: str, dataset: str) -> Dict[str, Any]:
    if dataset not in DATASET_SPECS:
        raise SystemExit(f"unknown dataset {dataset!r}; choose from {sorted(DATASET_SPECS)}")
    cfg = DATASET_SPECS[dataset]
    spec = MODEL_SPECS[model_short]
    num_layers = int(spec["num_layers"])
    benchmark = cfg.get("benchmark", "pope")
    steered_root = perception_dump_dir(benchmark, model_short, cfg["run_tag"])
    if cfg.get("baseline_in_run_tag", True):
        baseline_dir = steered_root / "baseline"
    else:
        baseline_dir = (
            perception_dump_dir(benchmark, model_short, cfg["baseline_run_tag"])
            / "baseline"
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
    qtype = cfg.get("qtype")
    gold = cfg.get("gold")
    n_items = int(cfg["n_items"])
    baseline_man = _filter_manifest(
        baseline_man, conditions=conditions, qtype=qtype, gold=gold
    )
    _gate_item_counts(
        baseline_man,
        conditions=conditions,
        n_items=n_items,
        label=f"{model_short}/{dataset} baseline",
    )
    baseline_stats = baseline_condition_stats(
        baseline_man,
        conditions=conditions,
        correct_label=cfg["correct_label"],
        flip_from=cfg["flip_from"],
        prob_score_field=cfg["prob_score_field"],
    )
    metrics = collect_cell_metrics(
        steered_root,
        baseline_man,
        cell_ids,
        conditions=conditions,
        correct_label=cfg["correct_label"],
        flip_from=cfg["flip_from"],
        flip_to=cfg["flip_to"],
        flip_key=cfg["flip_key"],
        prob_score_field=cfg["prob_score_field"],
        qtype=qtype,
        gold=gold,
        n_items=n_items,
    )
    return {
        "model_short": model_short,
        "dataset": dataset,
        "dataset_cfg": cfg,
        "display_name": spec["display_name"],
        "num_layers": num_layers,
        "metrics": metrics,
        "baseline_stats": baseline_stats,
    }


def plot_joint_mean_token_prob_model_by_mlp(
    bundles: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    """2x2: rows = model, cols = additive @ mlp | rotation @ mlp.

    Each pane stacks the two prompt conditions for the shared dataset.
    Plots mean P(yes) for pope30_yes and mean P(no) for pope30_no.
    """
    if not bundles:
        raise ValueError("bundles must be non-empty")
    dataset_cfg = bundles[0]["dataset_cfg"]
    for b in bundles[1:]:
        if b["dataset"] != bundles[0]["dataset"]:
            raise ValueError("joint figure requires a single dataset across bundles")
    conditions = dataset_cfg["conditions"]
    condition_labels = dataset_cfg["condition_labels"]
    token = dataset_cfg["prob_token_label"]
    score_field = dataset_cfg["prob_score_field"]
    mlp_cols = (
        ("additive_mlp", "additive @ mlp"),
        ("rotation_mlp", "rotation @ mlp"),
    )
    n_models = len(bundles)
    n_conds = len(conditions)
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
        f"{dataset_cfg['dataset_title']}: mean unconditional P({token}) at first "
        "answer token, by steering layer window\n"
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
            for crow, cond in enumerate(conditions):
                ax = fig.add_subplot(inner[crow, 0])
                base_mean, base_lo, base_hi = bootstrap_mean_ci(
                    baseline_stats[cond]["p_token_raw"]
                )
                bar_values: Dict[float, List[float]] = {b: [] for b in betas}
                bar_errs: Dict[float, Tuple[List[float], List[float]]] = {
                    b: ([], []) for b in betas
                }
                for wlab in wlabs:
                    for beta in betas:
                        cell = _cell_id_for(prefix, beta, wlab)
                        mean, lo, hi = bootstrap_mean_ci(
                            metrics[cell][cond]["p_token_raw"]
                        )
                        bar_values[beta].append(mean)
                        bar_errs[beta][0].append(lo)
                        bar_errs[beta][1].append(hi)

                pane_title = (
                    f"{model_title}  ·  {method_title}  ·  "
                    f"{condition_labels[cond]}"
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
                    ylabel=f"Mean P({token})" if col == 0 else "",
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
        f"n={dataset_cfg['n_items']} items per cell-condition. Mean of {score_field} "
        f"over all items; error bars = bootstrap 95% CI (n={BOOTSTRAP_N}). "
        "Grey hatched = no-intervention baseline (repeated at each window). "
        "Layer windows differ by model depth (LLaVA 32 vs Qwen 28).",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


plot_joint_mean_p_yes_raw_model_by_mlp = plot_joint_mean_token_prob_model_by_mlp


def plot_joint_flips_model_by_mlp(
    bundles: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    """2x2: rows = model, cols = additive @ mlp | rotation @ mlp (flips metric)."""
    if not bundles:
        raise ValueError("bundles must be non-empty")
    dataset_cfg = bundles[0]["dataset_cfg"]
    for b in bundles[1:]:
        if b["dataset"] != bundles[0]["dataset"]:
            raise ValueError("joint figure requires a single dataset across bundles")
    conditions = dataset_cfg["conditions"]
    condition_labels = dataset_cfg["condition_labels"]
    flip_key = dataset_cfg["flip_key"]
    flip_from = dataset_cfg["flip_from"]
    n_items = int(dataset_cfg["n_items"])
    mlp_cols = (
        ("additive_mlp", "additive @ mlp"),
        ("rotation_mlp", "rotation @ mlp"),
    )
    n_models = len(bundles)
    n_conds = len(conditions)
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
        f"{dataset_cfg['dataset_title']}: {dataset_cfg['flip_title']}, "
        "by steering layer window\n"
        "Rows = model · columns = MLP intervention",
        fontsize=13,
        y=0.985,
    )

    observed_max = 0.0
    axes_drawn: List[Any] = []
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
            for crow, cond in enumerate(conditions):
                ax = fig.add_subplot(inner[crow, 0])
                axes_drawn.append(ax)
                bs = baseline_stats[cond]
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
                            val = float(m[flip_key])
                            bar_values[beta].append(val)
                            if np.isfinite(val):
                                observed_max = max(observed_max, val)
                            excl = m["flip_pairs_excluded"]
                            annotations[beta].append(
                                f"x={excl} excl" if excl else None
                            )
                            full_notes[beta].append(None)

                pane_title = (
                    f"{model_title}  ·  {method_title}  ·  "
                    f"{condition_labels[cond]}  ·  baseline: "
                    f"{bs['n_flip_from']}/{bs['n_items']} parsed {flip_from}"
                )
                _draw_grouped_bars(
                    ax,
                    betas=betas,
                    window_labs=wlabs,
                    x_pos=xpos,
                    bar_values=bar_values,
                    annotations=annotations,
                    full_height_notes=full_notes,
                    baseline_y=None,
                    baseline_bars=None,
                    ylabel=dataset_cfg["flip_ylabel"] if col == 0 else "",
                    title=pane_title,
                    ylim=(0, flips_ylim_max(n_items, observed_max)),
                    integer_y=True,
                )
                if crow < n_conds - 1:
                    ax.set_xticklabels([])

    y_top = flips_ylim_max(n_items, observed_max)
    for ax in axes_drawn:
        ax.set_ylim(0, y_top)

    handles, labels = _legend_handles(
        BETAS_BY_PREFIX["additive_mlp"],
        include_baseline_line=False,
        include_baseline_bar=False,
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
        f"n={n_items} items per cell-condition. "
        + dataset_cfg["flip_caption"]
        + " Bars = flips from the same-condition no-intervention baseline "
        "to the steered response. "
        "Layer windows differ by model depth (LLaVA 32 vs Qwen 28).",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _legend_and_footer(
    fig: Any,
    *,
    footer: str,
    include_baseline_bar: bool = True,
    baseline_bar_label: str = "no-intervention baseline",
) -> None:
    handles, labels = _legend_handles(
        BETAS_BY_PREFIX["additive_mlp"],
        include_baseline_line=False,
        include_baseline_bar=include_baseline_bar,
        baseline_bar_label=baseline_bar_label,
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
    fig.text(0.5, 0.008, footer, ha="center", va="bottom", fontsize=8)


def plot_joint_llava_by_gold_mlp(
    yes_bundle: Dict[str, Any],
    no_bundle: Dict[str, Any],
    *,
    metric: str,
    out_path: Path,
    family_label: str,
) -> None:
    """LLaVA merged figure: gold=yes (top) vs gold=no (bottom) × MLP methods.

    Outer rows = gold=yes then gold=no.
    Outer cols = additive @ mlp | rotation @ mlp.
    Each cell stacks the two gold-conditional prompt conditions.
    metric: "accuracy" | "mean_p" | "flips".
    family_label: e.g. "AMBER-100 attribute" or "POPE-30".
    """
    if metric not in {"accuracy", "mean_p", "flips"}:
        raise ValueError(f"unknown metric {metric!r}")
    yes_cfg = yes_bundle["dataset_cfg"]
    no_cfg = no_bundle["dataset_cfg"]
    if yes_cfg.get("qtype") != no_cfg.get("qtype"):
        raise ValueError("yes/no bundles must share the same qtype")
    if yes_cfg.get("gold") != "yes" or no_cfg.get("gold") != "no":
        raise ValueError("bundles must be gold=yes then gold=no")
    n_items = int(yes_cfg["n_items"])
    model_title = yes_bundle["display_name"]
    num_layers = yes_bundle["num_layers"]
    wlabs = window_labels(num_layers)
    xpos = window_display_positions(wlabs)
    betas = BETAS_BY_PREFIX["additive_mlp"]

    gold_rows = (
        ("gold=yes", yes_bundle, yes_cfg),
        ("gold=no", no_bundle, no_cfg),
    )
    # Each gold half has two condition panes → 4 panes top, 4 bottom.
    panes_per_gold = 2
    fig = plt.figure(figsize=(16.5, 3.4 * len(gold_rows) * panes_per_gold + 1.6))
    outer = fig.add_gridspec(
        len(gold_rows),
        2,
        hspace=0.36,
        wspace=0.16,
        left=0.06,
        right=0.98,
        top=0.90,
        bottom=0.05,
    )

    if metric == "accuracy":
        fig.suptitle(
            f"{model_title}, {family_label}: accuracy by steering layer window\n"
            "Rows = gold answer (yes then no) · columns = MLP intervention "
            "(baseline = grey hatched bars)",
            fontsize=13,
            y=0.985,
        )
        footer = (
            f"n={n_items} items per cell-condition. "
            "Top half: accuracy = fraction parsed yes (gold=yes). "
            "Bottom half: accuracy = fraction parsed no (gold=no). "
            "Grey hatched = no-intervention baseline. rotation @ layer omitted."
        )
    elif metric == "mean_p":
        fig.suptitle(
            f"{model_title}, {family_label}: mean unconditional first-token "
            "probability by steering layer window\n"
            "Top = mean P(yes) (gold=yes) · bottom = mean P(no) (gold=no) · "
            "columns = MLP intervention (baseline = grey hatched bars)",
            fontsize=13,
            y=0.985,
        )
        footer = (
            f"n={n_items} items per cell-condition. "
            "Top: mean score_p_yes_raw; bottom: mean score_p_no_raw. "
            f"Error bars: bootstrap 95% CI (n={BOOTSTRAP_N}). "
            "Grey hatched = no-intervention baseline. rotation @ layer omitted."
        )
    else:
        fig.suptitle(
            f"{model_title}, {family_label}: items flipped from baseline "
            "by steering layer window\n"
            "Top = yes→no (gold=yes) · bottom = no→yes (gold=no) · "
            "columns = MLP intervention",
            fontsize=13,
            y=0.985,
        )
        footer = (
            f"n={n_items} items per cell-condition. "
            "Top: flip = baseline yes and steered no. "
            "Bottom: flip = baseline no and steered yes. "
            "Bars = flips from the same-condition no-intervention baseline "
            "to the steered response. rotation @ layer omitted."
        )

    observed_flip_max = 0.0
    flip_axes: List[Any] = []

    for row, (gold_label, bundle, cfg) in enumerate(gold_rows):
        metrics = bundle["metrics"]
        baseline_stats = bundle["baseline_stats"]
        conditions = cfg["conditions"]
        condition_labels = cfg["condition_labels"]
        flip_key = cfg["flip_key"]
        flip_from = cfg["flip_from"]
        token = cfg["prob_token_label"]

        for col, (prefix, method_title) in enumerate(MLP_JOINT_COLS):
            inner = outer[row, col].subgridspec(len(conditions), 1, hspace=0.38)
            for crow, cond in enumerate(conditions):
                ax = fig.add_subplot(inner[crow, 0])
                pane_title = (
                    f"{gold_label}  ·  {method_title}  ·  "
                    f"{condition_labels[cond]}"
                )

                if metric == "accuracy":
                    bar_values: Dict[float, List[float]] = {b: [] for b in betas}
                    annotations: Dict[float, List[Optional[str]]] = {
                        b: [] for b in betas
                    }
                    full_notes: Dict[float, List[Optional[str]]] = {
                        b: [] for b in betas
                    }
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
                                note = (
                                    f"u={m['n_unparseable']}"
                                    if m["n_unparseable"]
                                    else None
                                )
                                annotations[beta].append(note)
                                full_notes[beta].append(None)
                    base_acc = baseline_stats[cond]["accuracy"]
                    _draw_grouped_bars(
                        ax,
                        betas=betas,
                        window_labs=wlabs,
                        x_pos=xpos,
                        bar_values=bar_values,
                        annotations=annotations,
                        full_height_notes=full_notes,
                        baseline_y=None,
                        baseline_bars=[base_acc] * len(wlabs),
                        ylabel="Accuracy" if col == 0 else "",
                        title=pane_title,
                        ylim=(0.0, 1.15),
                    )
                elif metric == "mean_p":
                    bar_values = {b: [] for b in betas}
                    bar_errs: Dict[float, Tuple[List[float], List[float]]] = {
                        b: ([], []) for b in betas
                    }
                    base_mean, base_lo, base_hi = bootstrap_mean_ci(
                        baseline_stats[cond]["p_token_raw"]
                    )
                    for wlab in wlabs:
                        for beta in betas:
                            cell = _cell_id_for(prefix, beta, wlab)
                            mean, lo, hi = bootstrap_mean_ci(
                                metrics[cell][cond]["p_token_raw"]
                            )
                            bar_values[beta].append(mean)
                            bar_errs[beta][0].append(lo)
                            bar_errs[beta][1].append(hi)
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
                        ylabel=f"Mean P({token})" if col == 0 else "",
                        title=pane_title,
                        ylim=(0.0, 1.05),
                    )
                else:
                    flip_axes.append(ax)
                    bar_values = {b: [] for b in betas}
                    annotations = {b: [] for b in betas}
                    full_notes = {b: [] for b in betas}
                    bs = baseline_stats[cond]
                    pane_title = (
                        f"{gold_label}  ·  {method_title}  ·  "
                        f"{condition_labels[cond]}  ·  baseline: "
                        f"{bs['n_flip_from']}/{bs['n_items']} parsed {flip_from}"
                    )
                    for wlab in wlabs:
                        for beta in betas:
                            cell = _cell_id_for(prefix, beta, wlab)
                            m = metrics[cell][cond]
                            if m["n_parseable"] == 0:
                                bar_values[beta].append(float("nan"))
                                annotations[beta].append(None)
                                full_notes[beta].append("all unparseable")
                            else:
                                val = float(m[flip_key])
                                bar_values[beta].append(val)
                                if np.isfinite(val):
                                    observed_flip_max = max(observed_flip_max, val)
                                excl = m["flip_pairs_excluded"]
                                annotations[beta].append(
                                    f"x={excl} excl" if excl else None
                                )
                                full_notes[beta].append(None)
                    _draw_grouped_bars(
                        ax,
                        betas=betas,
                        window_labs=wlabs,
                        x_pos=xpos,
                        bar_values=bar_values,
                        annotations=annotations,
                        full_height_notes=full_notes,
                        baseline_y=None,
                        baseline_bars=None,
                        ylabel=cfg["flip_ylabel"] if col == 0 else "",
                        title=pane_title,
                        ylim=(0, flips_ylim_max(n_items, observed_flip_max)),
                        integer_y=True,
                    )

                if crow < len(conditions) - 1:
                    ax.set_xticklabels([])

    if metric == "flips":
        y_top = flips_ylim_max(n_items, observed_flip_max)
        for ax in flip_axes:
            ax.set_ylim(0, y_top)
        _legend_and_footer(
            fig,
            footer=footer,
            include_baseline_bar=False,
        )
    else:
        _legend_and_footer(fig, footer=footer, include_baseline_bar=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# Backward-compatible alias.
def plot_joint_amber_llava_by_gold_mlp(
    yes_bundle: Dict[str, Any],
    no_bundle: Dict[str, Any],
    *,
    metric: str,
    out_path: Path,
) -> None:
    yes_cfg = yes_bundle["dataset_cfg"]
    qtype = yes_cfg.get("qtype") or "subset"
    plot_joint_llava_by_gold_mlp(
        yes_bundle,
        no_bundle,
        metric=metric,
        out_path=out_path,
        family_label=f"AMBER-100 {qtype}",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS.keys()),
        default=None,
        help="Model short name; omit when only writing the joint figure",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS.keys()),
        default=None,
        help="Dataset keys (default: both pope30_yes and pope30_no)",
    )
    parser.add_argument(
        "--joint_mean_p_yes",
        action="store_true",
        help="Write 2x2 joint mean-P(token) PNG (LLaVA/Qwen x additive/rotation mlp)",
    )
    parser.add_argument(
        "--joint_flips",
        action="store_true",
        help="Write 2x2 joint flips PNG (LLaVA/Qwen x additive/rotation mlp)",
    )
    parser.add_argument(
        "--joint_amber_llava",
        action="store_true",
        help=(
            "Write LLaVA AMBER-100 merged PNGs (gold=yes top / gold=no bottom × "
            "additive|rotation mlp) for accuracy, mean P, and flips"
        ),
    )
    parser.add_argument(
        "--joint_pope_llava",
        action="store_true",
        help=(
            "Write LLaVA POPE-30 merged flips PNG (gold=yes top / gold=no bottom × "
            "additive|rotation mlp)"
        ),
    )
    parser.add_argument(
        "--amber_capabilities",
        nargs="+",
        choices=sorted(AMBER_LLAVA_CAPABILITIES.keys()),
        default=None,
        help="With --joint_amber_llava: which capabilities (default: attribute relation)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=(
            "Output plot directory for a single --datasets entry "
            "(default: plots/<model>/<dataset>/)"
        ),
    )
    parser.add_argument(
        "--joint_out",
        type=Path,
        default=None,
        help="Path for a single joint PNG when writing one joint figure",
    )
    args = parser.parse_args(argv)

    joint_flags = (
        args.joint_mean_p_yes,
        args.joint_flips,
        args.joint_amber_llava,
        args.joint_pope_llava,
    )
    if args.model is None and not any(joint_flags):
        parser.error(
            "pass --model and/or --joint_mean_p_yes / --joint_flips / "
            "--joint_amber_llava / --joint_pope_llava"
        )

    datasets = (
        list(args.datasets)
        if args.datasets is not None
        else (["pope30_yes", "pope30_no"] if args.model is not None else [])
    )
    if args.out_dir is not None and len(datasets) != 1:
        parser.error("--out_dir requires exactly one --datasets entry")

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
        for dataset in datasets:
            bundle = _load_model_bundle(args.model, dataset)
            cfg = bundle["dataset_cfg"]
            for cond in cfg["conditions"]:
                bs = bundle["baseline_stats"][cond]
                print(
                    f"{args.model}/{dataset} baseline[{cond}]: n={bs['n_items']} "
                    f"parseable={bs['n_parseable']} correct={bs['n_correct']} "
                    f"flip_from={bs['n_flip_from']} "
                    f"accuracy={bs['accuracy']:.3f} "
                    f"mean_p_{cfg['prob_token_label']}_raw={bs['mean_p_token_raw']:.4f}"
                )
            out_dir = args.out_dir or (
                plots_root
                / str(MODEL_SPECS[args.model]["plots_subdir"])
                / dataset
            )
            written.extend(
                _write_plot_set(
                    out_dir=out_dir,
                    metrics=bundle["metrics"],
                    baseline_stats=bundle["baseline_stats"],
                    model_title=bundle["display_name"],
                    num_layers=bundle["num_layers"],
                    dataset_cfg=cfg,
                )
            )

    joint_requested = args.joint_mean_p_yes or args.joint_flips
    if joint_requested:
        if len(datasets) != 1:
            parser.error(
                "--joint_mean_p_yes / --joint_flips require exactly one --datasets entry"
            )
        if args.joint_mean_p_yes and args.joint_flips and args.joint_out is not None:
            parser.error("--joint_out cannot be used when writing both joint figures")
        dataset = datasets[0]
        bundles = [
            _load_model_bundle("llava-1.5-7b-hf", dataset),
            _load_model_bundle("qwen2.5-vl-7b-instruct", dataset),
        ]
        cfg = DATASET_SPECS[dataset]
        pope_joint_dir = plots_root / "merged_plots" / "pope-30-runs"
        if args.joint_mean_p_yes:
            token = cfg["prob_token_label"]
            joint_out = args.joint_out or (
                pope_joint_dir
                / (
                    f"{dataset}_mean_p_{token}_raw_by_layer_window_"
                    "llava_vs_qwen_additive_vs_rotation_mlp.png"
                )
            )
            plot_joint_mean_token_prob_model_by_mlp(bundles, joint_out)
            written.append(joint_out)
        if args.joint_flips:
            flip_from = cfg["flip_from"]
            joint_out = args.joint_out or (
                pope_joint_dir
                / (
                    f"{dataset}_flips_from_baseline_{flip_from}_by_layer_window_"
                    "llava_vs_qwen_additive_vs_rotation_mlp.png"
                )
            )
            plot_joint_flips_model_by_mlp(bundles, joint_out)
            written.append(joint_out)

    if args.joint_amber_llava:
        if args.joint_out is not None:
            parser.error("--joint_out cannot be used with --joint_amber_llava")
        caps = list(args.amber_capabilities) if args.amber_capabilities else list(
            AMBER_LLAVA_CAPABILITIES.keys()
        )
        amber_dir = plots_root / "merged_plots" / "amber-100-runs"
        for capability in caps:
            yes_key, no_key = AMBER_LLAVA_CAPABILITIES[capability]
            yes_bundle = _load_model_bundle("llava-1.5-7b-hf", yes_key)
            no_bundle = _load_model_bundle("llava-1.5-7b-hf", no_key)
            metric_files = (
                (
                    "accuracy",
                    (
                        f"amber100_{capability}_accuracy_by_layer_window_"
                        "llava_gold_yes_vs_no_additive_vs_rotation_mlp.png"
                    ),
                ),
                (
                    "mean_p",
                    (
                        f"amber100_{capability}_mean_p_yes_vs_p_no_raw_by_layer_window_"
                        "llava_gold_yes_vs_no_additive_vs_rotation_mlp.png"
                    ),
                ),
                (
                    "flips",
                    (
                        f"amber100_{capability}_flips_from_baseline_by_layer_window_"
                        "llava_gold_yes_vs_no_additive_vs_rotation_mlp.png"
                    ),
                ),
            )
            for metric, filename in metric_files:
                out_path = amber_dir / filename
                plot_joint_llava_by_gold_mlp(
                    yes_bundle,
                    no_bundle,
                    metric=metric,
                    out_path=out_path,
                    family_label=f"AMBER-100 {capability}",
                )
                written.append(out_path)

    if args.joint_pope_llava:
        if args.joint_out is not None:
            parser.error("--joint_out cannot be used with --joint_pope_llava")
        yes_bundle = _load_model_bundle("llava-1.5-7b-hf", "pope30_yes")
        no_bundle = _load_model_bundle("llava-1.5-7b-hf", "pope30_no")
        out_path = (
            plots_root
            / "merged_plots"
            / "pope-30-runs"
            / (
                "pope30_flips_from_baseline_by_layer_window_"
                "llava_gold_yes_vs_no_additive_vs_rotation_mlp.png"
            )
        )
        plot_joint_llava_by_gold_mlp(
            yes_bundle,
            no_bundle,
            metric="flips",
            out_path=out_path,
            family_label="POPE-30",
        )
        written.append(out_path)

    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
