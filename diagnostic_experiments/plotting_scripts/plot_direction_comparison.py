#!/usr/bin/env python3
"""Plot Direction Comparison — Compositional vs Semantic Safety Direction.

Reads `direction_comparison_{source}_{representation}.json` produced by the
multi-source `compositional_safety_direction.py` and renders three plots:
  - cosine_similarity.png    — |cos(c^l, s^l)| per layer
  - effective_rank.png       — effective rank of the SSU-vs-SSS scatter
  - subspace_overlap.png     — top-5 subspace overlap (when present)

Pass `--all_sources` to scan the results directory and produce one set of
plots per (source, representation) it finds.
"""

import argparse
import json
import re
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
_FNAME_RE = re.compile(r"^direction_comparison_(.+)\.json$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--source", choices=["holisafe", "mssbench"], default=None,
                   help="Restrict to one source. Default: plot every "
                        "(source, representation) found in results/.")
    p.add_argument("--representation", choices=["tt", "vl"], default=None)
    return p.parse_args()


def _plot_one(data, label: str, out_dir: Path, model_name: str):
    data = [d for d in data if d["layer"] > 0]
    if not data:
        print(f"  [warn] no rows for {label}; skipping")
        return
    layers = np.array([d["layer"] for d in data])
    x = np.arange(len(layers))

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
    })
    sub = out_dir / label
    sub.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        path = sub / name
        fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {path}")
        plt.close(fig)

    cos_vals = np.array([d["cosine_sim_compositional_vs_semantic"] for d in data])
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, np.abs(cos_vals), width=0.6, color="#7c3aed", alpha=0.8)
    ax.set_ylabel("|Cosine Similarity|")
    ax.set_title(
        f"|cos(c^l, s^l)|  ·  {label}\n{model_name}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, "cosine_similarity.png")

    ranks = np.array([d["effective_rank"] for d in data])
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, ranks, width=0.6, color="#059669", alpha=0.8)
    ax.set_ylabel("Effective Rank (tau=0.9)")
    ax.set_title(f"Effective Rank of SSU-vs-SSS Activation Space  ·  {label}\n{model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    fig.tight_layout()
    _save(fig, "effective_rank.png")

    if any("subspace_overlap_top5" in d for d in data):
        overlap = np.array([d.get("subspace_overlap_top5", float("nan")) for d in data])
        valid = ~np.isnan(overlap)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(x[valid], overlap[valid], "o-", color="#2563eb", markersize=6, linewidth=1.8)
        ax.set_ylabel("Subspace Overlap (top-5 PCs)")
        ax.set_title(
            f"Subspace Overlap: Compositional vs Semantic  ·  {label}\n{model_name}"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers], fontsize=8)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        _save(fig, "subspace_overlap.png")


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    results_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME / "outputs" / "results"
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover available (source, representation) JSONs.
    all_files = list(results_dir.glob("direction_comparison_*.json"))
    if not all_files:
        print(f"  [warn] no direction_comparison_*.json under {results_dir}")
        return

    selected = []
    for path in sorted(all_files):
        m = _FNAME_RE.match(path.name)
        if not m:
            continue
        label = m.group(1)        # e.g. "holisafe_tt"
        source, _, rep = label.rpartition("_")
        if args.source and source != args.source:
            continue
        if args.representation and rep != args.representation:
            continue
        selected.append((label, path))
    if not selected:
        print(f"  [warn] no matching JSONs under {results_dir} for "
              f"source={args.source} representation={args.representation}")
        return

    for label, path in selected:
        with open(path) as f:
            data = json.load(f)
        _plot_one(data, label, out_dir, model_name)


if __name__ == "__main__":
    main()
