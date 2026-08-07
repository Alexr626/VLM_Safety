#!/usr/bin/env python3
"""Compare matched LLaVA POPE cells: 2026-06-19 vs 2026-07-30.

Matched frame
-------------
- model: llava-1.5-7b-hf
- intervention: vti_textual_additive_mlp
- layers: all decoder layers
- beta: 0.2, 0.5, 0.9
- POPE: random / popular / adversarial (n=200 each; same pin)

Direction arms
--------------
- 06-19: author demos, nd=70, PC1+mean (legacy textual_directions cache)
- 07-30: demos850 partition, nd=500, raw mean-difference, layers_all
- magnitude control: demos850 nd=500 PC1+mean (*_r2_partition), not eval'd here
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.classifiers.metrics import _normalize_yes_no  # noqa: E402

MODEL = "llava-1.5-7b-hf"
SPLITS = ("pope_random", "pope_popular", "pope_adversarial")
SPLIT_SHORT = {
    "pope_random": "random",
    "pope_popular": "popular",
    "pope_adversarial": "adversarial",
}
BETAS = (0.2, 0.5, 0.9)
METRICS = (
    "accuracy",
    "precision",
    "recall",
    "accuracy_gold_no",
    "accuracy_gold_yes",
)
METRIC_LABELS = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "accuracy_gold_no": "accuracy on gold-no",
    "accuracy_gold_yes": "accuracy on gold-yes",
}

RUN_0619 = "2026-06-19"
RUN_0730 = "2026-07-30"
LABEL_0619 = "06-19 author nd70 PC1+mean"
LABEL_0730 = "07-30 demos850 nd500 meandiff"
LABEL_BASELINE = "baseline (no intervention)"

COLOR_0619 = "#4C72B0"
COLOR_0730 = "#C44E52"
COLOR_BASELINE = "#7A7A7A"
COLOR_CONTROL = "#55A868"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_root",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Default: results_root/2026-08-06/_analysis_pope_0619_vs_0730_matched",
    )
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def safe_div(num: float, den: float) -> float | None:
    return num / den if den else None


def recompute_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
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
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "n_total": n_total,
        "n_unparsed": n_unparsed,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_gold_yes": n_gold_yes,
        "n_gold_no": n_gold_no,
        "accuracy": safe_div(tp + tn, n_total),
        "precision": precision,
        "recall": recall,
        "accuracy_gold_no": safe_div(tn, n_gold_no),
        "accuracy_gold_yes": safe_div(tp, n_gold_yes),
    }


def cell_dir_0619(results_root: Path, split: str, beta: float | None) -> Path:
    base = results_root / RUN_0619 / MODEL / split
    if beta is None:
        return base / "no_intervention"
    return base / f"vti_textual_additive_mlp__b{beta}"


def cell_dir_0730(results_root: Path, split: str, beta: float | None) -> Path:
    base = results_root / RUN_0730 / MODEL / split
    if beta is None:
        return base / "no_intervention"
    return (
        base
        / f"vti_textual_additive_mlp__b{beta}__dall__nd500__meandiff__layers_all"
    )


def load_cell_metrics(cell_dir: Path) -> dict[str, Any]:
    responses_path = cell_dir / "responses.json"
    if not responses_path.exists():
        raise FileNotFoundError(responses_path)
    records = load_json(responses_path)
    counts = recompute_counts(records)
    summary_path = cell_dir / "metric_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    # Cross-check accuracy / precision / recall against summary when present.
    for key_sum, key_c in (
        ("accuracy_overall", "accuracy"),
        ("precision_overall", "precision"),
        ("recall_overall", "recall"),
    ):
        if key_sum in summary and counts[key_c] is not None:
            if abs(float(summary[key_sum]) - float(counts[key_c])) > 1e-9:
                raise ValueError(
                    f"Mismatch {key_sum} at {summary_path}: "
                    f"summary={summary[key_sum]} recomputed={counts[key_c]}"
                )
    counts["cell_dir"] = str(cell_dir)
    counts["intervention_config"] = summary.get("intervention_config", {})
    return counts


def load_directions(npz_path: Path) -> np.ndarray:
    """Load ``(num_layers, hidden_dim)`` float64 directions from a cache npz."""
    z = np.load(npz_path)
    keys = sorted(
        [k for k in z.files if k.startswith("layer_")],
        key=lambda s: int(s.split("_")[1]),
    )
    return np.stack([z[k] for k in keys], axis=0).astype(np.float64)


def per_layer_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity per layer row. NaN if either row has zero norm."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    out = np.empty(a.shape[0], dtype=np.float64)
    for i in range(a.shape[0]):
        na = np.linalg.norm(a[i])
        nb = np.linalg.norm(b[i])
        if na == 0.0 or nb == 0.0:
            out[i] = np.nan
        else:
            out[i] = float(np.dot(a[i], b[i]) / (na * nb))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def pct(x: float | None) -> str:
    if x is None:
        return ""
    return f"{100.0 * x:.2f}"


def delta_pp(a: float | None, b: float | None) -> str:
    """b - a in percentage points."""
    if a is None or b is None:
        return ""
    return f"{100.0 * (b - a):+.2f}"


def build_tables(
    per_split: dict[str, dict[str, dict[str, Any]]],
    baselines: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Return per-split rows, split-averaged rows, and a markdown report."""
    per_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        base = baselines[split]
        for beta in BETAS:
            a = per_split[split]["0619"][beta]
            b = per_split[split]["0730"][beta]
            row: dict[str, Any] = {
                "split": SPLIT_SHORT[split],
                "beta": beta,
                "n_total": a["n_total"],
                "n_gold_yes": a["n_gold_yes"],
                "n_gold_no": a["n_gold_no"],
                "baseline_accuracy": base["accuracy"],
                "baseline_precision": base["precision"],
                "baseline_recall": base["recall"],
                "baseline_accuracy_gold_no": base["accuracy_gold_no"],
                "baseline_accuracy_gold_yes": base["accuracy_gold_yes"],
            }
            for m in METRICS:
                row[f"0619_{m}"] = a[m]
                row[f"0730_{m}"] = b[m]
                row[f"delta_0730_minus_0619_{m}"] = (
                    None if a[m] is None or b[m] is None else b[m] - a[m]
                )
                row[f"delta_0619_minus_baseline_{m}"] = (
                    None if a[m] is None or base[m] is None else a[m] - base[m]
                )
                row[f"delta_0730_minus_baseline_{m}"] = (
                    None if b[m] is None or base[m] is None else b[m] - base[m]
                )
            per_rows.append(row)

    avg_rows: list[dict[str, Any]] = []
    for beta in BETAS:
        row = {"split": "split_average", "beta": beta, "n_total": 600}
        for prefix, getter in (
            ("baseline", lambda sp, _b: baselines[sp]),
            ("0619", lambda sp, b: per_split[sp]["0619"][b]),
            ("0730", lambda sp, b: per_split[sp]["0730"][b]),
        ):
            for m in METRICS:
                vals = [getter(sp, beta)[m] for sp in SPLITS]
                if any(v is None for v in vals):
                    row[f"{prefix}_{m}"] = None
                else:
                    row[f"{prefix}_{m}"] = sum(vals) / len(vals)
        for m in METRICS:
            a, b, base = row[f"0619_{m}"], row[f"0730_{m}"], row[f"baseline_{m}"]
            row[f"delta_0730_minus_0619_{m}"] = (
                None if a is None or b is None else b - a
            )
            row[f"delta_0619_minus_baseline_{m}"] = (
                None if a is None or base is None else a - base
            )
            row[f"delta_0730_minus_baseline_{m}"] = (
                None if b is None or base is None else b - base
            )
        avg_rows.append(row)

    # Markdown
    lines = [
        "# LLaVA POPE matched comparison: 2026-06-19 vs 2026-07-30",
        "",
        "Frame: `vti_textual_additive_mlp`, all layers, β ∈ {0.2, 0.5, 0.9},",
        "POPE random/popular/adversarial (200 each).",
        "",
        f"- **{LABEL_0619}** — `evaluation/results/{RUN_0619}/…`",
        f"- **{LABEL_0730}** — `evaluation/results/{RUN_0730}/…`",
        f"- **{LABEL_BASELINE}** — identical on both dates",
        "",
        "Deltas are in **percentage points** (pp). `delta_0730_minus_0619` = 07-30 − 06-19.",
        "",
        "## Split-averaged",
        "",
    ]
    for m in METRICS:
        lines.append(f"### {METRIC_LABELS[m]} (%)")
        lines.append("")
        lines.append(
            "| β | baseline | 06-19 | 07-30 | 07-30 − 06-19 | 06-19 − base | 07-30 − base |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in avg_rows:
            lines.append(
                "| {beta} | {base} | {a} | {b} | {d} | {da} | {db} |".format(
                    beta=row["beta"],
                    base=pct(row[f"baseline_{m}"]),
                    a=pct(row[f"0619_{m}"]),
                    b=pct(row[f"0730_{m}"]),
                    d=delta_pp(row[f"0619_{m}"], row[f"0730_{m}"]),
                    da=delta_pp(row[f"baseline_{m}"], row[f"0619_{m}"]),
                    db=delta_pp(row[f"baseline_{m}"], row[f"0730_{m}"]),
                )
            )
        lines.append("")

    lines.append("## Per split")
    lines.append("")
    for split in SPLITS:
        short = SPLIT_SHORT[split]
        lines.append(f"### {short}")
        lines.append("")
        for m in METRICS:
            lines.append(f"#### {METRIC_LABELS[m]} (%)")
            lines.append("")
            lines.append(
                "| β | baseline | 06-19 | 07-30 | 07-30 − 06-19 | 06-19 − base | 07-30 − base |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for beta in BETAS:
                row = next(
                    r for r in per_rows if r["split"] == short and r["beta"] == beta
                )
                lines.append(
                    "| {beta} | {base} | {a} | {b} | {d} | {da} | {db} |".format(
                        beta=beta,
                        base=pct(row[f"baseline_{m}"]),
                        a=pct(row[f"0619_{m}"]),
                        b=pct(row[f"0730_{m}"]),
                        d=delta_pp(row[f"0619_{m}"], row[f"0730_{m}"]),
                        da=delta_pp(row[f"baseline_{m}"], row[f"0619_{m}"]),
                        db=delta_pp(row[f"baseline_{m}"], row[f"0730_{m}"]),
                    )
                )
            lines.append("")

    return per_rows, avg_rows, "\n".join(lines)


def plot_metric_bars(
    out_path: Path,
    *,
    title: str,
    ylabel: str,
    baseline: float,
    values_0619: list[float],
    values_0730: list[float],
    as_percent: bool = True,
) -> None:
    scale = 100.0 if as_percent else 1.0
    base_v = baseline * scale
    v19 = [v * scale for v in values_0619]
    v30 = [v * scale for v in values_0730]

    x = np.arange(len(BETAS))
    width = 0.28
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    # Baseline as its own bar group at each beta (same height) so it sits
    # beside the steered arms, plus a dashed reference line.
    bars_b = ax.bar(
        x - width,
        [base_v] * len(BETAS),
        width,
        label=f"baseline ({base_v:.1f}%)" if as_percent else "baseline",
        color=COLOR_BASELINE,
        alpha=0.85,
    )
    bars_19 = ax.bar(
        x,
        v19,
        width,
        label=LABEL_0619,
        color=COLOR_0619,
    )
    bars_30 = ax.bar(
        x + width,
        v30,
        width,
        label=LABEL_0730,
        color=COLOR_0730,
    )
    ax.axhline(base_v, color=COLOR_BASELINE, linestyle="--", linewidth=1.2, alpha=0.8)

    def annotate(bars, color):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                (bar.get_x() + bar.get_width() / 2, h),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=7,
                color=color,
            )

    annotate(bars_b, COLOR_BASELINE)
    annotate(bars_19, COLOR_0619)
    annotate(bars_30, COLOR_0730)

    all_vals = [base_v, *v19, *v30]
    pad = 2.0 if as_percent else 0.02
    ax.set_ylim(min(all_vals) - pad, max(all_vals) + pad * 2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BETAS])
    ax.set_xlabel("beta")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_layer_magnitudes(
    out_path: Path,
    *,
    norms_0619: np.ndarray,
    norms_0730_meandiff: np.ndarray,
    norms_control_pc1: np.ndarray,
) -> None:
    layers = np.arange(len(norms_0619))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(
        layers,
        norms_0619,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        color=COLOR_0619,
        label="06-19 author nd70 PC1+mean (eval'd)",
    )
    ax.plot(
        layers,
        norms_0730_meandiff,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        color=COLOR_0730,
        label="07-30 demos850 nd500 meandiff (eval'd)",
    )
    ax.plot(
        layers,
        norms_control_pc1,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        color=COLOR_CONTROL,
        label="demos850 nd500 PC1+mean (control; not in these POPE cells)",
    )
    ax.set_xlabel("decoder layer index")
    ax.set_ylabel("direction L2 norm")
    ax.set_title(
        "Per-layer steering-direction magnitude\n"
        "LLaVA-1.5-7B — matched POPE comparison arms + demos850 PC1+mean control"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Absolute difference of per-layer norms: |‖d0619‖ − ‖d0730‖|
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    diff_eval = np.abs(norms_0619 - norms_0730_meandiff)
    diff_ctrl = np.abs(norms_control_pc1 - norms_0730_meandiff)
    ax.plot(
        layers,
        diff_eval,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        color="#8172B3",
        label="|‖06-19 PC1+mean‖ − ‖07-30 meandiff‖|",
    )
    ax.plot(
        layers,
        diff_ctrl,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        color=COLOR_CONTROL,
        label="|‖demos850 PC1+mean‖ − ‖07-30 meandiff‖| (same demos)",
    )
    ax.set_xlabel("decoder layer index")
    ax.set_ylabel("|Δ L2 norm|")
    ax.set_title(
        "Absolute per-layer magnitude difference\n"
        "eval arms vs same-demo meandiff/PC1+mean control gap"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(
        out_path.with_name("direction_magnitude_abs_diff_per_layer.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_pairwise_cosine(
    out_path: Path,
    *,
    cosines: np.ndarray,
    title: str,
    series_label: str,
    color: str,
) -> None:
    layers = np.arange(len(cosines))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(
        layers,
        cosines,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        color=color,
        label=series_label,
    )
    ax.axhline(1.0, color="#BBBBBB", linestyle=":", linewidth=1.0, alpha=0.9)
    ax.axhline(0.0, color="#BBBBBB", linestyle="--", linewidth=1.0, alpha=0.7)
    finite = cosines[np.isfinite(cosines)]
    if finite.size:
        ymin = min(-0.05, float(finite.min()) - 0.05)
        ymax = max(1.05, float(finite.max()) + 0.05)
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel("decoder layer index")
    ax.set_ylabel("cosine similarity")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else results_root / "2026-08-06" / "_analysis_pope_0619_vs_0730_matched"
    )
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"

    baselines: dict[str, dict[str, Any]] = {}
    per_split: dict[str, dict[str, dict[str, Any]]] = {}

    for split in SPLITS:
        # Baselines must match across dates.
        b19 = load_cell_metrics(cell_dir_0619(results_root, split, None))
        b30 = load_cell_metrics(cell_dir_0730(results_root, split, None))
        for m in METRICS:
            if abs(float(b19[m]) - float(b30[m])) > 1e-12:
                raise RuntimeError(
                    f"Baseline mismatch on {split} {m}: "
                    f"0619={b19[m]} 0730={b30[m]}"
                )
        baselines[split] = b19
        per_split[split] = {"0619": {}, "0730": {}}
        for beta in BETAS:
            per_split[split]["0619"][beta] = load_cell_metrics(
                cell_dir_0619(results_root, split, beta)
            )
            per_split[split]["0730"][beta] = load_cell_metrics(
                cell_dir_0730(results_root, split, beta)
            )

    per_rows, avg_rows, md = build_tables(per_split, baselines)

    fieldnames_per = list(per_rows[0].keys())
    write_csv(tables_dir / "per_split_metrics.csv", per_rows, fieldnames_per)
    # avg rows omit some count fields
    fieldnames_avg = list(avg_rows[0].keys())
    write_csv(tables_dir / "split_averaged_metrics.csv", avg_rows, fieldnames_avg)
    (out_dir / "comparison_tables.md").write_text(md)

    # JSON dump of recomputed cells for provenance
    dump_json(
        out_dir / "cells.json",
        {
            "frame": {
                "model": MODEL,
                "intervention": "vti_textual_additive_mlp",
                "layers": "all",
                "betas": list(BETAS),
                "splits": list(SPLITS),
                "arm_0619": {
                    "run_date": RUN_0619,
                    "demos": "data/vti/demos.jsonl",
                    "num_demos": 70,
                    "steer_reconstruction": "live_pc1_plus_mean (legacy obtain_textual_vti)",
                    "dir_pattern": "vti_textual_additive_mlp__b{beta}",
                },
                "arm_0730": {
                    "run_date": RUN_0730,
                    "demos": "data/vti/demos_850.jsonl",
                    "num_demos": 500,
                    "steer_reconstruction": "raw_mean_difference",
                    "dir_pattern": (
                        "vti_textual_additive_mlp__b{beta}__dall__nd500__"
                        "meandiff__layers_all"
                    ),
                },
            },
            "baselines": {
                SPLIT_SHORT[s]: {k: baselines[s][k] for k in METRICS + ("n_total", "n_gold_yes", "n_gold_no", "tp", "fp", "tn", "fn", "n_unparsed")}
                for s in SPLITS
            },
            "steered": {
                SPLIT_SHORT[s]: {
                    str(beta): {
                        "0619": {
                            k: per_split[s]["0619"][beta][k]
                            for k in METRICS
                            + ("n_total", "tp", "fp", "tn", "fn", "n_unparsed")
                        },
                        "0730": {
                            k: per_split[s]["0730"][beta][k]
                            for k in METRICS
                            + ("n_total", "tp", "fp", "tn", "fn", "n_unparsed")
                        },
                    }
                    for beta in BETAS
                }
                for s in SPLITS
            },
        },
    )

    # Split-averaged bar plots (one per metric)
    for m in METRICS:
        base_avg = sum(baselines[s][m] for s in SPLITS) / len(SPLITS)
        v19 = [next(r for r in avg_rows if r["beta"] == b)[f"0619_{m}"] for b in BETAS]
        v30 = [next(r for r in avg_rows if r["beta"] == b)[f"0730_{m}"] for b in BETAS]
        plot_metric_bars(
            plots_dir / "split_averaged" / f"{m}_vs_beta_bars.png",
            title=(
                f"LLaVA POPE {METRIC_LABELS[m]} vs beta (split-averaged)\n"
                f"additive MLP, all layers — matched 06-19 vs 07-30 arms"
            ),
            ylabel=f"{METRIC_LABELS[m]} (%)",
            baseline=base_avg,
            values_0619=v19,
            values_0730=v30,
        )

    # Per-split bar plots for accuracy (primary) + all metrics under split dirs
    for split in SPLITS:
        short = SPLIT_SHORT[split]
        for m in METRICS:
            plot_metric_bars(
                plots_dir / "per_split" / short / f"{m}_vs_beta_bars.png",
                title=(
                    f"LLaVA POPE {short}: {METRIC_LABELS[m]} vs beta\n"
                    f"additive MLP, all layers — matched 06-19 vs 07-30 arms"
                ),
                ylabel=f"{METRIC_LABELS[m]} (%)",
                baseline=baselines[split][m],
                values_0619=[per_split[split]["0619"][b][m] for b in BETAS],
                values_0730=[per_split[split]["0730"][b][m] for b in BETAS],
            )

    # Direction geometry (magnitudes + pairwise cosines)
    art = PROJECT_ROOT / "experiment_artifacts" / "vti" / MODEL
    path_0619 = art / "textual_directions_nd70_rank1_seed42.npz"
    path_0730 = (
        art
        / "textual_v2"
        / "demos850_ba05bd96_all_nd500_s42_meandiff_partition"
        / "directions.npz"
    )
    path_ctrl = (
        art
        / "textual_v2"
        / "demos850_ba05bd96_all_nd500_s42_r2_partition"
        / "directions.npz"
    )
    d_0619 = load_directions(path_0619)
    d_0730 = load_directions(path_0730)
    d_ctrl = load_directions(path_ctrl)
    if not (d_0619.shape == d_0730.shape == d_ctrl.shape):
        raise RuntimeError(
            f"Direction shape mismatch: 0619={d_0619.shape} "
            f"0730={d_0730.shape} ctrl={d_ctrl.shape}"
        )
    norms_0619 = np.linalg.norm(d_0619, axis=-1)
    norms_0730 = np.linalg.norm(d_0730, axis=-1)
    norms_ctrl = np.linalg.norm(d_ctrl, axis=-1)
    plot_layer_magnitudes(
        plots_dir / "direction_magnitude_per_layer.png",
        norms_0619=norms_0619,
        norms_0730_meandiff=norms_0730,
        norms_control_pc1=norms_ctrl,
    )

    cos_0619_0730 = per_layer_cosine(d_0619, d_0730)
    cos_0619_ctrl = per_layer_cosine(d_0619, d_ctrl)
    cos_0730_ctrl = per_layer_cosine(d_0730, d_ctrl)
    cosine_dir = plots_dir / "direction_cosine"
    plot_pairwise_cosine(
        cosine_dir / "cosine_0619_pc1mean_vs_0730_meandiff.png",
        cosines=cos_0619_0730,
        title=(
            "Per-layer cosine similarity\n"
            "06-19 author nd70 PC1+mean  vs  07-30 demos850 nd500 meandiff"
        ),
        series_label="cos(06-19 PC1+mean, 07-30 meandiff)",
        color="#8172B3",
    )
    plot_pairwise_cosine(
        cosine_dir / "cosine_0619_pc1mean_vs_demos850_pc1mean.png",
        cosines=cos_0619_ctrl,
        title=(
            "Per-layer cosine similarity\n"
            "06-19 author nd70 PC1+mean  vs  demos850 nd500 PC1+mean (control)"
        ),
        series_label="cos(06-19 PC1+mean, demos850 PC1+mean)",
        color=COLOR_0619,
    )
    plot_pairwise_cosine(
        cosine_dir / "cosine_0730_meandiff_vs_demos850_pc1mean.png",
        cosines=cos_0730_ctrl,
        title=(
            "Per-layer cosine similarity\n"
            "07-30 demos850 nd500 meandiff  vs  demos850 nd500 PC1+mean (control)"
        ),
        series_label="cos(07-30 meandiff, demos850 PC1+mean)",
        color=COLOR_CONTROL,
    )

    dump_json(
        out_dir / "direction_magnitudes.json",
        {
            "0619_author_nd70_pc1_plus_mean": {
                "path": str(path_0619),
                "per_layer_l2": norms_0619.tolist(),
                "flat_l2": float(np.linalg.norm(d_0619)),
            },
            "0730_demos850_nd500_meandiff": {
                "path": str(path_0730),
                "per_layer_l2": norms_0730.tolist(),
                "flat_l2": float(np.linalg.norm(d_0730)),
            },
            "control_demos850_nd500_pc1_plus_mean": {
                "path": str(path_ctrl),
                "per_layer_l2": norms_ctrl.tolist(),
                "flat_l2": float(np.linalg.norm(d_ctrl)),
            },
        },
    )
    dump_json(
        out_dir / "direction_cosines.json",
        {
            "0619_pc1mean_vs_0730_meandiff": {
                "per_layer_cosine": cos_0619_0730.tolist(),
                "mean_over_layers": float(np.nanmean(cos_0619_0730)),
            },
            "0619_pc1mean_vs_demos850_pc1mean": {
                "per_layer_cosine": cos_0619_ctrl.tolist(),
                "mean_over_layers": float(np.nanmean(cos_0619_ctrl)),
            },
            "0730_meandiff_vs_demos850_pc1mean": {
                "per_layer_cosine": cos_0730_ctrl.tolist(),
                "mean_over_layers": float(np.nanmean(cos_0730_ctrl)),
            },
        },
    )

    readme = f"""# Matched LLaVA POPE comparison: 2026-06-19 vs 2026-07-30

Built by `evaluation/pope_0619_vs_0730_matched/build_comparison.py`.

## Frame

| Knob | Value |
|------|-------|
| Model | `{MODEL}` |
| Intervention | `vti_textual_additive_mlp` |
| Layers | all |
| β | 0.2, 0.5, 0.9 |
| POPE | random / popular / adversarial, n=200 each (same pin; baselines identical) |

| Arm | Demos | nd | Direction |
|-----|-------|----|-----------|
| 06-19 | author `demos.jsonl` | 70 | PC1 + mean |
| 07-30 | `demos_850` partition | 500 | raw mean-difference |
| control | `demos_850` same nd500 block | 500 | PC1 + mean (`*_r2_partition`; not an eval cell here) |

## Outputs

- `comparison_tables.md` — human-readable tables (rates ×100, deltas in pp)
- `tables/per_split_metrics.csv`, `tables/split_averaged_metrics.csv`
- `cells.json` — recomputed counts
- `plots/split_averaged/{{metric}}_vs_beta_bars.png`
- `plots/per_split/{{split}}/{{metric}}_vs_beta_bars.png`
- `plots/direction_magnitude_per_layer.png`
- `plots/direction_magnitude_abs_diff_per_layer.png`
- `plots/direction_cosine/cosine_0619_pc1mean_vs_0730_meandiff.png`
- `plots/direction_cosine/cosine_0619_pc1mean_vs_demos850_pc1mean.png`
- `plots/direction_cosine/cosine_0730_meandiff_vs_demos850_pc1mean.png`
- `direction_magnitudes.json`, `direction_cosines.json`
"""
    (out_dir / "README.md").write_text(readme)
    print(f"Wrote comparison under {out_dir}")


if __name__ == "__main__":
    main()
