"""Offline gate readout and plots from per_item_p_yes.json (+ metric_summary.json)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Core metrics (unchanged API for run_smoke / summarize_cell)
# ---------------------------------------------------------------------------


def _roc_auc(y_true: List[int], scores: List[float]) -> float:
    """Binary ROC AUC (positive class = 1)."""
    pairs = sorted(zip(scores, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        pos_in_tie = sum(pairs[k][1] for k in range(i, j + 1))
        rank_sum += avg_rank * pos_in_tie
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_per_item_p_yes(path: Path) -> List[dict]:
    return json.loads(path.read_text())


def _labels_from_records(records: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.array(
        [1 if (r.get("ground_truth") or "").lower() == "yes" else 0 for r in records],
        dtype=int,
    )
    scores = np.array([float(r["p_yes"]) for r in records], dtype=float)
    return y, scores


def c_auc_from_records(records: List[dict]) -> float:
    y, scores = _labels_from_records(records)
    return _roc_auc(y.tolist(), scores.tolist())


def roc_curve_steppy(
    records: List[dict],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (fpr, tpr, auc) with one step per distinct p_yes threshold (N≤26)."""
    y, scores = _labels_from_records(records)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    auc = _roc_auc(y.tolist(), scores.tolist())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), auc

    thresholds = np.unique(scores)
    thresholds = np.sort(thresholds)[::-1]  # high → low

    fpr_pts = [0.0]
    tpr_pts = [0.0]
    for tau in thresholds:
        pred = (scores >= tau).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tpr_pts.append(tp / n_pos)
        fpr_pts.append(fp / n_neg)
    fpr_pts.append(1.0)
    tpr_pts.append(1.0)
    return np.array(fpr_pts), np.array(tpr_pts), auc


def threshold_curve(
    baseline_records: List[dict],
    n_thresholds: int = 101,
) -> List[Tuple[float, float, float]]:
    """Return list of (tau, accuracy, yes_ratio) from baseline p_yes sweep."""
    scores = [float(r["p_yes"]) for r in baseline_records]
    if not scores:
        return []
    taus = [i / (n_thresholds - 1) for i in range(n_thresholds)]
    curve = []
    for tau in taus:
        preds = ["yes" if s >= tau else "no" for s in scores]
        yes_ratio = sum(p == "yes" for p in preds) / len(preds)
        correct = sum(
            p == (r.get("ground_truth") or "").lower()
            for p, r in zip(preds, baseline_records)
        )
        acc = correct / len(preds)
        curve.append((tau, acc, yes_ratio))
    return curve


def threshold_curve_steppy(
    baseline_records: List[dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Accuracy vs yes_ratio using only thresholds that change predictions (steppy)."""
    scores = np.array([float(r["p_yes"]) for r in baseline_records])
    gold = [(r.get("ground_truth") or "").lower() for r in baseline_records]
    thresholds = np.unique(np.concatenate([[0.0], scores, [1.0]]))
    thresholds = np.sort(thresholds)[::-1]

    acc_pts: List[float] = []
    yr_pts: List[float] = []
    for tau in thresholds:
        preds = ["yes" if s >= tau else "no" for s in scores]
        yes_ratio = sum(p == "yes" for p in preds) / len(preds)
        correct = sum(p == g for p, g in zip(preds, gold))
        acc_pts.append(correct / len(preds))
        yr_pts.append(yes_ratio)
    return np.array(acc_pts), np.array(yr_pts)


def operating_point(records: List[dict], threshold: float = 0.5) -> Tuple[float, float]:
    preds = [
        "yes" if float(r["p_yes"]) >= threshold else "no"
        for r in records
    ]
    yes_ratio = sum(p == "yes" for p in preds) / len(preds)
    correct = sum(
        p == (r.get("ground_truth") or "").lower()
        for p, r in zip(preds, records)
    )
    return correct / len(preds), yes_ratio


def decoded_operating_point(cell_dir: Path) -> Tuple[float, float]:
    """Accuracy and yes_ratio from parsed responses (metric_summary.json)."""
    path = cell_dir / "metric_summary.json"
    if not path.is_file():
        return float("nan"), float("nan")
    m = json.loads(path.read_text())
    return float(m.get("accuracy_overall", float("nan"))), float(
        m.get("yes_ratio", float("nan"))
    )


def flip_decomposition(
    baseline_records: List[dict],
    steered_records: List[dict],
    threshold: float = 0.5,
) -> Dict[str, int]:
    by_id_b = {r["id"]: r for r in baseline_records}
    out = {
        "n_decision_flips": 0,
        "flip_tn_to_fp": 0,
        "flip_fp_to_tn": 0,
        "flip_tp_to_fn": 0,
        "flip_fn_to_tp": 0,
    }
    for r in steered_records:
        b = by_id_b.get(r["id"])
        if not b:
            continue
        pb = "yes" if float(b["p_yes"]) >= threshold else "no"
        ps = "yes" if float(r["p_yes"]) >= threshold else "no"
        if pb == ps:
            continue
        out["n_decision_flips"] += 1
        gb = (b.get("ground_truth") or "").lower()
        if gb == "no" and pb == "no" and ps == "yes":
            out["flip_tn_to_fp"] += 1
        elif gb == "no" and pb == "yes" and ps == "no":
            out["flip_fp_to_tn"] += 1
        elif gb == "yes" and pb == "yes" and ps == "no":
            out["flip_tp_to_fn"] += 1
        elif gb == "yes" and pb == "no" and ps == "yes":
            out["flip_fn_to_tp"] += 1
    return out


def summarize_cell(
    cell_dir: Path,
    baseline_dir: Optional[Path] = None,
) -> dict:
    p_path = cell_dir / "per_item_p_yes.json"
    records = load_per_item_p_yes(p_path)
    acc, yes_r = operating_point(records)
    summary = {
        "cell_dir": str(cell_dir),
        "c_auc": c_auc_from_records(records),
        "accuracy_at_0.5": acc,
        "yes_ratio_at_0.5": yes_r,
        "n_items": len(records),
    }
    if baseline_dir is not None:
        base_recs = load_per_item_p_yes(baseline_dir / "per_item_p_yes.json")
        flips = flip_decomposition(base_recs, records)
        summary.update(flips)
        summary["h_plus"] = flips["flip_tn_to_fp"]
        summary["h_minus"] = flips["flip_fp_to_tn"]
    dec_acc, dec_yr = decoded_operating_point(cell_dir)
    summary["decoded_accuracy"] = dec_acc
    summary["decoded_yes_ratio"] = dec_yr
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

SMOKE_CELL_ORDER = [
    "no_intervention",
    "vti_textual_uniform_rotation_layer__b0.2",
    "vti_textual_uniform_rotation_layer__b0.4",
    "vti_textual_additive_layer__b0.9",
    "vti_visual_additive_layer__a0.2",
    "vti_visual_additive_layer__a0.4",
    "vti_visual_additive_layer__a0.9",
    "vti_visual_uniform_rotation_mlp__a0.2",
    "vti_visual_uniform_rotation_mlp__a0.4",
    "vti_visual_uniform_rotation_mlp__a0.9",
    "vti_visual_additive_mlp__a0.2",
    "vti_visual_additive_mlp__a0.4",
    "vti_visual_additive_mlp__a0.9",
    "vti_visual_uniform_rotation_layer__a0.2",
    "vti_visual_uniform_rotation_layer__a0.4",
    "vti_visual_uniform_rotation_layer__a0.9",
]

COMPARISON_SETS = {
    "textual": lambda n: n == "no_intervention" or n.startswith("vti_textual_"),
    "visual": lambda n: n == "no_intervention" or n.startswith("vti_visual_"),
    "all": lambda n: True,
}

COEFF_MARKERS = {0.2: "o", 0.4: "s", 0.9: "^"}
VARIANT_COLORS = {
    "uniform_rotation_layer": "#1f77b4",
    "additive_layer": "#ff7f0e",
    "uniform_rotation_mlp": "#2ca02c",
    "additive_mlp": "#d62728",
}


def _cell_sort_key(name: str) -> tuple:
    try:
        return (0, SMOKE_CELL_ORDER.index(name))
    except ValueError:
        return (1, name)


def _parse_cell_style(cell_name: str) -> Tuple[Optional[str], Optional[float], str]:
    if cell_name == "no_intervention":
        return None, None, "baseline"
    if cell_name.startswith("vti_textual_"):
        rest = cell_name[len("vti_textual_"):]
        if "__b" in rest:
            variant, coeff_s = rest.rsplit("__b", 1)
            return variant, float(coeff_s), "textual"
    if cell_name.startswith("vti_visual_"):
        rest = cell_name[len("vti_visual_"):]
        if "__a" in rest:
            variant, coeff_s = rest.rsplit("__a", 1)
            return variant, float(coeff_s), "visual"
    return cell_name, None, "other"


def _short_label(cell_name: str) -> str:
    variant, coeff, arm = _parse_cell_style(cell_name)
    if arm == "baseline":
        return "baseline"
    sym = "β" if arm == "textual" else "α"
    return f"{variant} {sym}={coeff}"


def _discover_cells(amber_dir: Path) -> List[Path]:
    cells = [
        d for d in amber_dir.iterdir()
        if d.is_dir() and (d / "per_item_p_yes.json").is_file()
    ]
    return sorted(cells, key=lambda p: _cell_sort_key(p.name))


def _cells_for_set(cells: Sequence[Path], set_name: str) -> List[Path]:
    pred = COMPARISON_SETS[set_name]
    return [c for c in cells if pred(c.name)]


def plot_roc_overlay(
    baseline_dir: Path,
    steered_dirs: Sequence[Path],
    out_path: Path,
    *,
    title: str,
) -> Path:
    base_recs = load_per_item_p_yes(baseline_dir / "per_item_p_yes.json")
    fpr, tpr, auc = roc_curve_steppy(base_recs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        fpr, tpr, drawstyle="steps-post", color="black", linewidth=2,
        label=f"baseline (AUC={auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle=":", color="#999", linewidth=1)

    for cell in steered_dirs:
        recs = load_per_item_p_yes(cell / "per_item_p_yes.json")
        fpr_s, tpr_s, auc_s = roc_curve_steppy(recs)
        variant, coeff, _ = _parse_cell_style(cell.name)
        color = VARIANT_COLORS.get(variant or "", "#888888")
        ax.plot(
            fpr_s, tpr_s, drawstyle="steps-post", color=color, linewidth=1.5,
            alpha=0.85, label=f"{_short_label(cell.name)} (AUC={auc_s:.3f})",
        )

    ax.set_xlabel("1 − specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_c_curve(
    baseline_dir: Path,
    steered_dirs: Sequence[Path],
    out_path: Path,
    *,
    title: str,
) -> Path:
    base_recs = load_per_item_p_yes(baseline_dir / "per_item_p_yes.json")
    acc_curve, yr_curve = threshold_curve_steppy(base_recs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        yr_curve, acc_curve, drawstyle="steps-post", color="black",
        linewidth=2, label="baseline p_yes sweep",
    )

    base_acc, base_yr = decoded_operating_point(baseline_dir)
    ax.scatter(
        [base_yr], [base_acc], s=120, c="black", marker="*", zorder=5,
        label=f"baseline decoded (acc={base_acc:.2f}, yr={base_yr:.2f})",
    )
    ax.annotate(
        "baseline decoded",
        (base_yr, base_acc),
        textcoords="offset points", xytext=(8, 8), fontsize=8,
    )

    for cell in steered_dirs:
        acc, yr = decoded_operating_point(cell)
        variant, coeff, _ = _parse_cell_style(cell.name)
        color = VARIANT_COLORS.get(variant or "", "#888888")
        marker = COEFF_MARKERS.get(coeff, "D") if coeff is not None else "D"
        ax.scatter(
            [yr], [acc], s=70, c=color, marker=marker, edgecolors="white",
            linewidths=0.5, zorder=4, label=_short_label(cell.name),
        )

    ax.set_xlabel("yes_ratio (decoded)")
    ax.set_ylabel("accuracy (decoded)")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6, loc="best", ncol=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_p_yes_histogram_pair(
    baseline_dir: Path,
    steered_dir: Path,
    out_path: Path,
    *,
    title: str,
) -> Path:
    base_scores = [float(r["p_yes"]) for r in load_per_item_p_yes(
        baseline_dir / "per_item_p_yes.json")]
    steer_scores = [float(r["p_yes"]) for r in load_per_item_p_yes(
        steered_dir / "per_item_p_yes.json")]

    bins = np.linspace(0.0, 1.0, 14)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        base_scores, bins=bins, alpha=0.55, color="black",
        label="baseline", edgecolor="white", linewidth=0.5,
    )
    ax.hist(
        steer_scores, bins=bins, alpha=0.55,
        color=VARIANT_COLORS.get(_parse_cell_style(steered_dir.name)[0] or "", "#888"),
        label=_short_label(steered_dir.name),
        edgecolor="white", linewidth=0.5,
    )
    ax.set_xlabel("p_yes")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_smoke_gate_plots(
    amber_dir: Path,
    plots_dir: Path,
    *,
    run_date: str = "",
    model_short: str = "",
) -> List[Path]:
    """Write ROC, c-curve, and p_yes histogram PNGs for AMBER smoke cells."""
    baseline_dir = amber_dir / "no_intervention"
    if not (baseline_dir / "per_item_p_yes.json").is_file():
        raise FileNotFoundError(f"Missing baseline: {baseline_dir}")

    all_cells = _discover_cells(amber_dir)
    steered_all = [c for c in all_cells if c.name != "no_intervention"]
    written: List[Path] = []

    tag = f"{model_short} " if model_short else ""
    date_tag = f"({run_date}) " if run_date else ""

    for set_name in ("textual", "visual", "all"):
        set_cells = _cells_for_set(all_cells, set_name)
        steered = [c for c in set_cells if c.name != "no_intervention"]
        if not steered:
            continue

        roc_path = plots_dir / f"roc_overlay_{set_name}.png"
        plot_roc_overlay(
            baseline_dir, steered, roc_path,
            title=f"{tag}ROC overlay {date_tag}— {set_name}",
        )
        written.append(roc_path)

        c_path = plots_dir / f"c_curve_{set_name}.png"
        plot_c_curve(
            baseline_dir, steered, c_path,
            title=f"{tag}c-curve {date_tag}— {set_name}",
        )
        written.append(c_path)

    hist_dir = plots_dir / "p_yes_histograms"
    for cell in steered_all:
        safe = re.sub(r"[^\w.-]+", "_", cell.name)
        h_path = hist_dir / f"p_yes_hist_baseline_vs_{safe}.png"
        plot_p_yes_histogram_pair(
            baseline_dir, cell, h_path,
            title=f"{tag}p_yes histogram: baseline vs {_short_label(cell.name)}",
        )
        written.append(h_path)

    return written


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-date", required=True)
    p.add_argument("--model-short", default="llava-1.5-7b-hf")
    p.add_argument("--output-dir", default="evaluation/results")
    p.add_argument("--plots-dir", type=Path, default=None,
                   help="default: {output_dir}/{run_date}/_diagnostics/gate_plots")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    amber_dir = (
        Path(args.output_dir) / args.run_date / args.model_short / "amber"
    )
    plots_dir = args.plots_dir or (
        Path(args.output_dir) / args.run_date / "_diagnostics" / "gate_plots"
    )
    paths = write_smoke_gate_plots(
        amber_dir, plots_dir,
        run_date=args.run_date, model_short=args.model_short,
    )
    for p in paths:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
