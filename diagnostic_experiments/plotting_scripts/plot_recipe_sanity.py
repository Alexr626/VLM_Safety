#!/usr/bin/env python3
"""Plot Recipe Sanity: cos(s^l_pair, s^l_joint) per layer.

Reads `recipe_sanity.json` written by `vl_activation_shift.py` and renders
a per-layer comparison of the two semantic-safety-direction estimators:

  s^l_pair   — top-1 PC of the centered per-pair difference matrix
                D = H_safe[i] - H_unsafe[i] (CatQA harmless/harmful pairs).
                The principled estimator for minimal-edit paired data:
                topic/style residual cancels per pair.

  s^l_joint  — top-1 right singular vector of the joint centered matrix
                [H_safe - mu; H_unsafe - mu]. CAST-style joint PCA — the
                legacy recipe; absorbs within-class topic variance.

A high cos(s^l_pair, s^l_joint) at a layer means the recipe choice does
not move the top-1 direction much (pairs really are minimal-edit there).
Lower values mean joint PCA was absorbing topic/style residual that
pairwise PCA discards — these are the layers where the refactor shifts
the canonical direction the most.

Output
------
  diagnostic_experiments/{model}/shift_dc/outputs/results/vl_activation_shift/plots/recipe_sanity.png
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

_EXPERIMENT_NAME = "shift_dc"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    results_dir = (_DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME
                   / "outputs" / "results" / "vl_activation_shift")
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    sanity_path = results_dir / "recipe_sanity.json"
    if not sanity_path.exists():
        print(f"  [warn] {sanity_path} not found. "
              f"Re-run vl_activation_shift.py to generate it.")
        return
    with open(sanity_path) as f:
        rows = json.load(f)
    if not rows:
        print(f"  [warn] no rows in {sanity_path}")
        return

    layers = np.array([r["layer"] for r in rows])
    cos = np.array([r.get("cos_pair_vs_joint", float("nan")) for r in rows])
    n_pairs = int(rows[0].get("n_pairs", 0))
    valid = ~np.isnan(cos)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
    })

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(layers))
    # Bars colored by sign — negative cosine means joint and pairwise s^l point
    # in OPPOSITE directions (they disagree even on which side is "safe").
    pos_mask = valid & (cos >= 0)
    neg_mask = valid & (cos < 0)
    ax.bar(x[pos_mask], cos[pos_mask], width=0.6, color="#7c3aed", alpha=0.85,
           edgecolor="#4c1d95", linewidth=0.6, label="aligned (cos ≥ 0)")
    if neg_mask.any():
        ax.bar(x[neg_mask], cos[neg_mask], width=0.6, color="#dc2626",
               alpha=0.85, edgecolor="#7f1d1d", linewidth=0.6,
               label="anti-aligned (cos < 0)")
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.axhline(1.0, color="#888", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axhline(0.0, color="#444", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("cos(s^l_pair, s^l_joint)")
    ax.set_xlabel("Transformer Layer")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)

    # Y-range: include the full data range plus a small margin; never clip
    # negative bars even if they fall outside [0, 1].
    if valid.any():
        ymin = min(0.0, float(np.nanmin(cos[valid])) - 0.05)
        ymax = max(1.005, float(np.nanmax(cos[valid])) + 0.02)
    else:
        ymin, ymax = 0.0, 1.005
    ax.set_ylim(max(-1.05, ymin), min(1.05, ymax))

    mean = float(np.nanmean(cos[valid])) if valid.any() else float("nan")
    mn = float(np.nanmin(cos[valid])) if valid.any() else float("nan")
    mx = float(np.nanmax(cos[valid])) if valid.any() else float("nan")
    ax.set_title(
        f"Recipe Sanity: cos(pairwise s^l, joint s^l)  ·  {model_name}\n"
        f"n_pairs={n_pairs} · mean={mean:.4f}  min={mn:.4f}  max={mx:.4f}"
    )

    # Annotate the layers with the lowest cosine (where the recipe matters most).
    if valid.any():
        order = np.argsort(cos)
        lowest = [i for i in order if valid[i]][:3]
        for i in lowest:
            ax.annotate(f"{cos[i]:.3f}",
                        xy=(x[i], cos[i]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8, color="#4c1d95")

    fig.tight_layout()
    out_path = out_dir / "recipe_sanity.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
