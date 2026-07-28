#!/usr/bin/env python3
"""Geometric comparison: deployed all/nd200 vs shuffled-image control direction.

CPU-only. Loads ``directions.npz`` + ``components.npz`` / metadata PC1 norms for
each model; writes per-layer cosine, magnitude overlay plots, CSV, and a facts-only
summary under this directory. No model loading, no steering, no W&B.

Usage::

    python diagnostic_experiments/perception_diag/control/geometric_comparison/\\
        compare_deployed_vs_shuffled_control.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.interventions.vti.directions_v2 import (  # noqa: E402
    load_textual_v2_directions,
)
from src.paths import experiment_artifacts_dir, project_root  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DEPLOYED_SLUG = "demosv2_9a44f4af_all_nd200_s42_r2_prefix"

# Display names / expected decoder shapes from the plan.
MODEL_SPECS = {
    "llava-1.5-7b-hf": {
        "display_name": "LLaVA-1.5-7B",
        "expected_shape": (32, 4096),
    },
    "qwen2.5-vl-7b-instruct": {
        "display_name": "Qwen2.5-VL-7B",
        "expected_shape": (28, 3584),
    },
}

# Soft annotation: mark layers whose max PC1/direction fraction is at or above
# this model's 90th percentile across decoder layers (either direction). No hard
# mean-dominance cutoff — the plan leaves that to the readout.
PC1_FRAC_ANNOTATE_PERCENTILE = 90.0


def deployed_dir(model_short: str) -> Path:
    return (
        experiment_artifacts_dir("vti", model_short)
        / "textual_v2"
        / DEPLOYED_SLUG
    )


def shuffled_dir(model_short: str) -> Path:
    return (
        experiment_artifacts_dir("vti", model_short)
        / "shuffled_control"
        / "all_nd200"
    )


def load_decoder_pc1_norms(cache_dir: Path, n_decoder: int) -> np.ndarray:
    """PC1 L2 norms for decoder layers only (drop embedding row).

    Prefer ``components.npz`` key ``pc0`` (PC1); fall back to metadata
    ``pc1_layer_norms`` sliced ``[1:]``.
    """
    comp_path = cache_dir / "components.npz"
    if comp_path.exists():
        c = np.load(comp_path)
        if "pc0" not in c.files:
            raise RuntimeError(f"{comp_path} missing pc0 (PC1)")
        pc0 = np.asarray(c["pc0"], dtype=np.float64)
        n_plus = int(c["n_layers_plus"]) if "n_layers_plus" in c.files else pc0.shape[0]
        if pc0.shape[0] != n_plus:
            raise RuntimeError(
                f"{comp_path}: pc0 rows {pc0.shape[0]} != n_layers_plus {n_plus}"
            )
        if n_plus != n_decoder + 1:
            raise RuntimeError(
                f"{comp_path}: n_layers_plus={n_plus} but directions have "
                f"{n_decoder} decoder layers (expect n_decoder+1)"
            )
        return np.linalg.norm(pc0[1:], axis=1)

    meta = load_textual_v2_directions(cache_dir)[1]
    norms = meta.get("pc1_layer_norms")
    if not norms:
        raise RuntimeError(f"No pc1_layer_norms in {cache_dir}/metadata.json")
    arr = np.asarray(norms, dtype=np.float64)
    if len(arr) != n_decoder + 1:
        raise RuntimeError(
            f"pc1_layer_norms length {len(arr)} != {n_decoder + 1} "
            f"(embedding + decoder) in {cache_dir}"
        )
    return arr[1:]


def per_layer_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise cosine; ``a``, ``b`` shaped ``(n_layers, hidden_dim)``."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    dots = np.sum(a.astype(np.float64) * b.astype(np.float64), axis=1)
    na = np.linalg.norm(a.astype(np.float64), axis=1)
    nb = np.linalg.norm(b.astype(np.float64), axis=1)
    return dots / np.maximum(na * nb, eps)


def compare_model(model_short: str) -> Dict:
    spec = MODEL_SPECS[model_short]
    expected = spec["expected_shape"]
    dep_path = deployed_dir(model_short)
    shuf_path = shuffled_dir(model_short)

    deployed, deployed_meta = load_textual_v2_directions(dep_path)
    shuffled, shuffled_meta = load_textual_v2_directions(shuf_path)

    if deployed.shape != expected:
        raise RuntimeError(
            f"{model_short} deployed shape {deployed.shape} != expected {expected}"
        )
    if shuffled.shape != expected:
        raise RuntimeError(
            f"{model_short} shuffled shape {shuffled.shape} != expected {expected}"
        )
    if deployed.shape != shuffled.shape:
        raise RuntimeError(
            f"{model_short} shape mismatch deployed {deployed.shape} vs "
            f"shuffled {shuffled.shape}"
        )

    n_layers, hidden_dim = deployed.shape
    deployed_norm = np.linalg.norm(deployed.astype(np.float64), axis=1)
    shuffled_norm = np.linalg.norm(shuffled.astype(np.float64), axis=1)
    cosine = per_layer_cosine(deployed, shuffled)

    # Cross-check against metadata direction_layer_norms[1:] when present.
    for label, arr, meta in (
        ("deployed", deployed_norm, deployed_meta),
        ("shuffled", shuffled_norm, shuffled_meta),
    ):
        meta_norms = meta.get("direction_layer_norms")
        if meta_norms and len(meta_norms) == n_layers + 1:
            meta_dec = np.asarray(meta_norms[1:], dtype=np.float64)
            max_abs = float(np.max(np.abs(arr - meta_dec)))
            if max_abs > 1e-4:
                raise RuntimeError(
                    f"{model_short} {label}: recomputed norms disagree with "
                    f"metadata[1:] (max abs {max_abs})"
                )

    deployed_pc1 = load_decoder_pc1_norms(dep_path, n_layers)
    shuffled_pc1 = load_decoder_pc1_norms(shuf_path, n_layers)
    if deployed_pc1.shape != (n_layers,) or shuffled_pc1.shape != (n_layers,):
        raise RuntimeError(f"{model_short}: PC1 norm length mismatch")

    dep_frac = deployed_pc1 / np.maximum(deployed_norm, 1e-12)
    shuf_frac = shuffled_pc1 / np.maximum(shuffled_norm, 1e-12)
    max_frac = np.maximum(dep_frac, shuf_frac)
    annotate_thresh = float(np.percentile(max_frac, PC1_FRAC_ANNOTATE_PERCENTILE))
    annotate_mask = max_frac >= annotate_thresh

    rows = []
    for i in range(n_layers):
        rows.append(
            {
                "layer": i,
                "cosine": float(cosine[i]),
                "deployed_norm": float(deployed_norm[i]),
                "shuffled_norm": float(shuffled_norm[i]),
                "deployed_pc1_norm": float(deployed_pc1[i]),
                "shuffled_pc1_norm": float(shuffled_pc1[i]),
                "deployed_pc1_fraction": float(dep_frac[i]),
                "shuffled_pc1_fraction": float(shuf_frac[i]),
                "pc1_fraction_annotated": bool(annotate_mask[i]),
            }
        )

    return {
        "model_short": model_short,
        "display_name": spec["display_name"],
        "deployed_path": str(dep_path.relative_to(project_root())),
        "shuffled_path": str(shuf_path.relative_to(project_root())),
        "shape": [n_layers, hidden_dim],
        "rows": rows,
        "annotate_thresh": annotate_thresh,
        "annotate_percentile": PC1_FRAC_ANNOTATE_PERCENTILE,
        "n_annotated": int(annotate_mask.sum()),
        "cosine_min": float(cosine.min()),
        "cosine_max": float(cosine.max()),
        "cosine_median": float(np.median(cosine)),
        "max_pc1_frac_deployed": float(dep_frac.max()),
        "max_pc1_frac_shuffled": float(shuf_frac.max()),
    }


def write_csv(result: Dict, out_path: Path) -> None:
    fieldnames = [
        "layer",
        "cosine",
        "deployed_norm",
        "shuffled_norm",
        "deployed_pc1_norm",
        "shuffled_pc1_norm",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in result["rows"]:
            w.writerow({k: row[k] for k in fieldnames})


def plot_cosine(result: Dict, out_path: Path) -> None:
    layers = [r["layer"] for r in result["rows"]]
    cos = [r["cosine"] for r in result["rows"]]
    annotated = [r for r in result["rows"] if r["pc1_fraction_annotated"]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(layers, cos, color="#1f4e79", linewidth=1.8, marker="o", markersize=3.5)
    if annotated:
        ax.scatter(
            [r["layer"] for r in annotated],
            [r["cosine"] for r in annotated],
            s=64,
            facecolors="none",
            edgecolors="#c45c26",
            linewidths=1.6,
            zorder=5,
            label=(
                f"≥{int(result['annotate_percentile'])}th pct PC1/direction "
                f"fraction (either direction)"
            ),
        )
        ax.legend(loc="best", fontsize=8)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-0.5, result["shape"][0] - 0.5)
    ax.set_xlabel("Decoder layer index")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(
        f"{result['display_name']}: deployed all/nd200 direction vs "
        "shuffled-image control"
    )
    ax.set_xticks(layers[:: max(1, len(layers) // 16)])
    caption = (
        "Open markers: layers at/above the "
        f"{int(result['annotate_percentile'])}th percentile of "
        "max(PC1 norm / direction norm) across decoder layers "
        f"(threshold={result['annotate_thresh']:.4f}). "
        "High PC1 fraction means the cosine sign is less trustworthy "
        "(PC1 sign is svd_flip–dependent)."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7.5, wrap=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_magnitude(result: Dict, out_path: Path) -> None:
    layers = [r["layer"] for r in result["rows"]]
    dep = [r["deployed_norm"] for r in result["rows"]]
    shuf = [r["shuffled_norm"] for r in result["rows"]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        layers,
        dep,
        color="#1f4e79",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        label="Deployed direction",
    )
    ax.plot(
        layers,
        shuf,
        color="#c45c26",
        linewidth=1.8,
        marker="s",
        markersize=3.5,
        label="Shuffled-image control direction",
    )
    ax.set_xlabel("Decoder layer index")
    ax.set_ylabel("L2 norm")
    ax.set_title(
        f"{result['display_name']}: per-layer magnitude — deployed all/nd200 "
        "vs shuffled-image control"
    )
    ax.set_xlim(-0.5, result["shape"][0] - 0.5)
    ax.set_xticks(layers[:: max(1, len(layers) // 16)])
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def format_pc1_table(result: Dict) -> List[str]:
    lines = [
        "| Layer | Deployed PC1 norm | Deployed direction norm | "
        "Deployed PC1/dir | Shuffled PC1 norm | Shuffled direction norm | "
        "Shuffled PC1/dir | Annotated |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in result["rows"]:
        lines.append(
            f"| {r['layer']} | {r['deployed_pc1_norm']:.6g} | "
            f"{r['deployed_norm']:.6g} | {r['deployed_pc1_fraction']:.6g} | "
            f"{r['shuffled_pc1_norm']:.6g} | {r['shuffled_norm']:.6g} | "
            f"{r['shuffled_pc1_fraction']:.6g} | "
            f"{'yes' if r['pc1_fraction_annotated'] else 'no'} |"
        )
    return lines


def write_summary(results: Sequence[Dict], out_path: Path) -> None:
    lines = [
        "# Geometric comparison: deployed vs shuffled-image control",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Control for: `{DEPLOYED_SLUG}` (demos_v2 `all` @ nd200)",
        "Stage: geometric only (no behavioral runs).",
        "",
        "## Preconditions",
        "",
    ]
    for r in results:
        lines.append(f"### {r['display_name']} (`{r['model_short']}`)")
        lines.append("")
        lines.append(
            f"- **Shape match:** deployed and shuffled both "
            f"`{tuple(r['shape'])}` (PASS)."
        )
        lines.append(
            f"- **Inputs:** `{r['deployed_path']}` vs `{r['shuffled_path']}`."
        )
        lines.append(
            f"- **PC1-fraction (decoder layers):** max deployed "
            f"{r['max_pc1_frac_deployed']:.6g}; max shuffled "
            f"{r['max_pc1_frac_shuffled']:.6g}. Layers annotated at/above "
            f"the {int(r['annotate_percentile'])}th percentile of "
            f"max(PC1/direction) across layers: {r['n_annotated']} "
            f"(threshold={r['annotate_thresh']:.6g}). "
            "No hard mean-dominance cutoff; fractions are the readout."
        )
        lines.append("")

    lines.extend(
        [
            "## Per-layer cosine",
            "",
        ]
    )
    for r in results:
        lines.append(f"### {r['display_name']}")
        lines.append("")
        lines.append(
            f"Cosine over decoder layers: min={r['cosine_min']:.6g}, "
            f"median={r['cosine_median']:.6g}, max={r['cosine_max']:.6g}."
        )
        lines.append("")
        lines.append("| Layer | Cosine | Deployed L2 | Shuffled L2 |")
        lines.append("|---:|---:|---:|---:|")
        for row in r["rows"]:
            lines.append(
                f"| {row['layer']} | {row['cosine']:.6g} | "
                f"{row['deployed_norm']:.6g} | {row['shuffled_norm']:.6g} |"
            )
        lines.append("")

    lines.extend(
        [
            "## PC1-fraction diagnostic (precondition 2)",
            "",
            "PC1 norms from `components.npz` key `pc0` (decoder rows only; "
            "embedding row dropped to align with `directions.npz`).",
            "",
        ]
    )
    for r in results:
        lines.append(f"### {r['display_name']}")
        lines.append("")
        lines.extend(format_pc1_table(r))
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
        ]
    )
    for r in results:
        ms = r["model_short"]
        lines.append(f"- `{ms}`:")
        lines.append(
            f"  - `cosine_deployed_vs_shuffled_control_by_layer_{ms}.png`"
        )
        lines.append(
            f"  - `magnitude_deployed_vs_shuffled_control_by_layer_{ms}.png`"
        )
        lines.append(f"  - `geometric_comparison_{ms}.csv`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SPECS.keys()),
        choices=list(MODEL_SPECS.keys()),
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory (default: this script's directory).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    for model_short in args.models:
        print(f"=== {model_short}")
        result = compare_model(model_short)
        results.append(result)

        csv_path = out_dir / f"geometric_comparison_{model_short}.csv"
        cos_path = (
            out_dir / f"cosine_deployed_vs_shuffled_control_by_layer_{model_short}.png"
        )
        mag_path = (
            out_dir
            / f"magnitude_deployed_vs_shuffled_control_by_layer_{model_short}.png"
        )
        write_csv(result, csv_path)
        plot_cosine(result, cos_path)
        plot_magnitude(result, mag_path)
        print(
            f"  shape={tuple(result['shape'])}  "
            f"cosine median={result['cosine_median']:.4f}  "
            f"range=[{result['cosine_min']:.4f}, {result['cosine_max']:.4f}]"
        )
        print(f"  wrote {csv_path.name}, {cos_path.name}, {mag_path.name}")

    summary_path = out_dir / "geometric_comparison_summary.md"
    write_summary(results, summary_path)
    print(f"  wrote {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
