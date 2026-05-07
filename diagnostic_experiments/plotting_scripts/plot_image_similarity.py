#!/usr/bin/env python3
"""
Plot DINOv2 image-similarity distributions for MSSBench chat stems.

Inputs
------
    data/mssbench/image_similarity/dinov2_similarity_scores.json
        (output of compute_image_similarity.py)

Outputs (PNG, 150 dpi)
----------------------
    image_similarity_distribution.png   Single pooled histogram, p25/p50/p75 lines.
    image_similarity_by_type.png        Small-multiples per `Type`.

Also prints a summary table to stdout.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default=None,
                   help="Path to dinov2_similarity_scores.json. "
                        "Default: data/mssbench/image_similarity/...")
    p.add_argument("--output_dir", default=None,
                   help="Output dir for plots. Default: <scores parent>/plots/")
    p.add_argument("--bin_width", type=float, default=0.02)
    p.add_argument("--train_only", action="store_true",
                   help="Restrict plots to stems flagged in_train_split.")
    return p.parse_args()


def _summarize(values, label):
    if len(values) == 0:
        return f"{label:<24} n=0"
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return (f"{label:<24} n={len(values):<5} mean={values.mean():.3f}  "
            f"median={q50:.3f}  p25={q25:.3f}  p75={q75:.3f}")


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    if args.scores is None:
        scores_path = project_root / "data" / "mssbench" / "image_similarity" / \
                      "dinov2_similarity_scores.json"
    else:
        scores_path = Path(args.scores)

    if not scores_path.exists():
        raise FileNotFoundError(
            f"{scores_path} not found. Run compute_image_similarity.py first.")

    with open(scores_path) as f:
        data = json.load(f)
    stems = data["stems"]
    if args.train_only:
        stems = [s for s in stems if s.get("in_train_split")]
        print(f"Restricting to {len(stems)} train-split stems.")

    if args.output_dir is None:
        out_dir = scores_path.parent / "plots"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sims = np.array([s["cosine_similarity"] for s in stems], dtype=np.float64)
    types = np.array([s.get("type", "unknown") for s in stems])

    bins = np.arange(min(0.0, sims.min()), 1.0 + args.bin_width, args.bin_width)

    # ── Pooled histogram ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(sims, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.95)
    q25, q50, q75 = np.percentile(sims, [25, 50, 75])
    for q, label in zip([q25, q50, q75], ["p25", "median", "p75"]):
        ax.axvline(q, color="black", linestyle="--", linewidth=1)
        ax.text(q, ax.get_ylim()[1] * 0.98, f"{label}\n{q:.3f}",
                ha="center", va="top", fontsize=8)
    ax.set_xlabel("DINOv2 cosine similarity (SSS image vs SSU image)")
    ax.set_ylabel("# MSSBench stems")
    title_suffix = " (train split)" if args.train_only else ""
    ax.set_title(f"MSSBench chat — image-pair similarity distribution{title_suffix}\n"
                 f"n={len(sims)}, model={data.get('model', 'dinov2')}")
    fig.tight_layout()
    pooled_path = out_dir / ("image_similarity_distribution_train.png"
                             if args.train_only else
                             "image_similarity_distribution.png")
    fig.savefig(pooled_path)
    plt.close(fig)
    print(f"  → {pooled_path}")

    # ── Stratified by Type ───────────────────────────────────────────────
    unique_types = sorted(set(types.tolist()))
    n_types = len(unique_types)
    ncols = min(n_types, 2)
    nrows = (n_types + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows),
                             dpi=150, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, t in zip(axes, unique_types):
        mask = types == t
        ax.hist(sims[mask], bins=bins, color="#72B7B2",
                edgecolor="white", alpha=0.95)
        ax.set_title(f"{t}  (n={int(mask.sum())})")
        ax.set_xlabel("cosine similarity")
        ax.set_ylabel("count")
    for ax in axes[len(unique_types):]:
        ax.set_visible(False)
    fig.suptitle(f"MSSBench chat similarity by Type{title_suffix}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    by_type_path = out_dir / ("image_similarity_by_type_train.png"
                              if args.train_only else
                              "image_similarity_by_type.png")
    fig.savefig(by_type_path)
    plt.close(fig)
    print(f"  → {by_type_path}")

    # ── Stdout summary table ─────────────────────────────────────────────
    print()
    print("Type                     n     mean     median   p25      p75")
    print("-" * 72)
    print(_summarize(sims, "ALL"))
    for t in unique_types:
        mask = types == t
        print(_summarize(sims[mask], f"chat:{t}"))


if __name__ == "__main__":
    main()
