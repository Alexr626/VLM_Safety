#!/usr/bin/env python3
"""
Scatter plot of P(yes | safe) vs. P(yes | unsafe) per question pair from
the causal-mediation preflight check.

Inputs
------
    preflight_yes_prob_check_*.json files produced by
    causal_mediation_*.py. Each file contains a `per_pair` list with
    {question_id, P_safe, P_unsafe, gap}.

Outputs
-------
    {output_dir}/preflight_yes_prob_{tag}.png   (one per input file)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True,
                   help="One or more preflight_yes_prob_check_*.json files.")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def _tag_from_path(path: Path) -> str:
    stem = path.stem
    prefix = "preflight_yes_prob_check_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def _plot_one(data: dict, out_path: Path, tag: str):
    pairs = data.get("per_pair", [])
    if not pairs:
        print(f"  [skip] {out_path.name}: per_pair is empty")
        return

    p_safe = np.asarray([p["P_safe"] for p in pairs], dtype=float)
    p_unsafe = np.asarray([p["P_unsafe"] for p in pairs], dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)

    lo, hi = 0.0, max(1.0, float(np.max(np.concatenate([p_safe, p_unsafe]))) * 1.02)
    ax.plot([lo, hi], [lo, hi], color="gray", linewidth=0.8,
            linestyle="--", label="y = x")

    above = p_unsafe > p_safe
    ax.scatter(p_safe[above], p_unsafe[above], s=28, color="#E45756",
               alpha=0.75, edgecolor="white", linewidth=0.5,
               label=f"P_unsafe > P_safe (n={int(above.sum())})")
    ax.scatter(p_safe[~above], p_unsafe[~above], s=28, color="#4C78A8",
               alpha=0.75, edgecolor="white", linewidth=0.5,
               label=f"P_unsafe ≤ P_safe (n={int((~above).sum())})")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("P(yes | safe run)")
    ax.set_ylabel("P(yes | unsafe run)")

    model = data.get("model", "")
    direction = data.get("patch_direction", "")
    n_used = data.get("n_used", len(pairs))
    median_gap = data.get("median_gap_abs")
    mean_gap = data.get("mean_gap_signed")
    parts = [model, f"tag={tag}", f"direction={direction}", f"n={n_used}"]
    subtitle_bits = []
    if median_gap is not None:
        subtitle_bits.append(f"median |gap|={median_gap:.3f}")
    if mean_gap is not None:
        subtitle_bits.append(f"mean signed gap={mean_gap:+.3f}")
    title = " — ".join([p for p in parts if p])
    if subtitle_bits:
        title += "\n" + ", ".join(subtitle_bits)
    ax.set_title(title, fontsize=10)

    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in args.results:
        path = Path(r)
        with open(path) as f:
            data = json.load(f)
        tag = _tag_from_path(path)
        out_path = out_dir / f"preflight_yes_prob_{tag}.png"
        _plot_one(data, out_path, tag)


if __name__ == "__main__":
    main()
