#!/usr/bin/env python3
"""Plot Cross-Direction Cosine: 5×5 cosine matrix per layer + summary.

Reads `cross_direction_cosine.json` produced by
`compare_compositional_directions.py` and renders:

  - cross_direction_summary.png   — line plot: cos(semantic, c^l_*) per layer
                                    for each compositional source.
  - cross_direction_layer_{l}.png — full 5×5 |cos| heatmap at the layer that
                                    maximizes the spread between sources
                                    (most informative single-layer figure).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_DIAGNOSTIC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DIAGNOSTIC_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from src.model import _normalize_model_name

_EXPERIMENT_NAME = "compositional_safety"

_NAME_LABELS = {
    "semantic":     "Semantic (CatQA)",
    "holisafe_tt":  "HoliSafe TT",
    "holisafe_vl":  "HoliSafe VL",
    "mssbench_tt":  "MSSBench TT",
    "mssbench_vl":  "MSSBench VL",
}
_SOURCE_COLORS = {
    "holisafe_tt":  "#dc2626",
    "holisafe_vl":  "#ea580c",
    "mssbench_tt":  "#059669",
    "mssbench_vl":  "#7c3aed",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    return p.parse_args()


def _full_matrix(row, names):
    """Materialise a symmetric |cos| matrix from a triangular cosine_matrix dict."""
    n = len(names)
    M = np.full((n, n), float("nan"))
    cm = row["cosine_matrix"]
    for i, a in enumerate(names):
        for j in range(i + 1, n):
            b = names[j]
            v = cm.get(a, {}).get(b)
            if v is not None and not np.isnan(v):
                M[i, j] = M[j, i] = abs(v)
        M[i, i] = 1.0
    return M


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    results_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME / "outputs" / "results"
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = results_dir / "cross_direction_cosine.json"
    if not path.exists():
        print(f"  [warn] {path} not found; run compare_compositional_directions.py first.")
        return
    with open(path) as f:
        rows = json.load(f)
    rows = [r for r in rows if r["layer"] > 0]
    if not rows:
        print(f"  [warn] no rows in {path}")
        return

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
    })

    def _save(fig, name):
        p = out_dir / name
        fig.savefig(str(p), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {p}")
        plt.close(fig)

    # ── Summary line plot: cos(semantic, c^l_*) per layer ────────────────
    layers = np.array([r["layer"] for r in rows])
    fig, ax = plt.subplots(figsize=(14, 5))
    plotted_any = False
    for source, color in _SOURCE_COLORS.items():
        ys = []
        for r in rows:
            v = r["cosine_matrix"].get("semantic", {}).get(source)
            ys.append(abs(v) if v is not None and not np.isnan(v) else float("nan"))
        ys = np.array(ys, dtype=float)
        valid = ~np.isnan(ys)
        if not valid.any():
            continue
        ax.plot(layers[valid], ys[valid], marker="o", linewidth=1.8,
                color=color, label=_NAME_LABELS[source])
        plotted_any = True
    if plotted_any:
        ax.set_ylabel("|cos(semantic, c^l)|")
        ax.set_title(f"Cross-Source Alignment to Semantic Direction\n{model_name}")
        ax.set_xlabel("Transformer Layer")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        fig.tight_layout()
        _save(fig, "cross_direction_summary.png")
    else:
        plt.close(fig)

    # ── Pick the layer with the largest semantic-vs-source spread ─────────
    spreads = []
    for r in rows:
        sem_row = r["cosine_matrix"].get("semantic", {})
        vals = [abs(v) for v in sem_row.values() if v is not None and not np.isnan(v)]
        spread = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
        spreads.append((r["layer"], spread, r))
    if not spreads:
        return
    best_layer, _, best_row = max(spreads, key=lambda t: t[1])

    names = best_row.get("available", [])
    if not names:
        return
    M = _full_matrix(best_row, names)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([_NAME_LABELS.get(n, n) for n in names],
                       rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([_NAME_LABELS.get(n, n) for n in names], fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            v = M[i, j]
            if not np.isnan(v):
                color = "white" if v > 0.7 or v < 0.3 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, color=color)
    ax.set_title(
        f"|cos| Matrix at Layer {best_layer} (max spread)\n{model_name}"
    )
    fig.colorbar(im, ax=ax, label="|Cosine|", shrink=0.8)
    fig.tight_layout()
    _save(fig, f"cross_direction_layer_{best_layer}.png")


if __name__ == "__main__":
    main()
