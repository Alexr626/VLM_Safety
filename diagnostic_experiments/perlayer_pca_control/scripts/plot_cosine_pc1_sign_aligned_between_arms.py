#!/usr/bin/env python3
"""Diagnostic replot: deployed-vs-control cosine after aligning PC1 signs.

Additive only — does not modify primary plots or direction artifacts.

Rule (align control to deployed via Check 9 signs): at each decoder layer, if
``sign(cos(PC1, mean))`` differs between arms, multiply the **control** PC1 by
``-1``. Layers where signs already agree are left unchanged (including both
``-1``). Directions are then rebuilt as ``PC1 + mean`` and the per-layer cosine
is recomputed.

This is a diagnostic readout of one deferred correction from the plan; it does
not adopt a sign convention for the primary measurement.

Example::

    python diagnostic_experiments/perlayer_pca_control/scripts/\\
        plot_cosine_pc1_sign_aligned_between_arms.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.interventions.vti.directions_partition import (  # noqa: E402
    PARTITION_SIZES,
    partition_cache_dir,
    partition_slug,
)
from evaluation.interventions.vti.perlayer_pca import (  # noqa: E402
    EXPECTED_DECODER_SHAPES,
    EXPECTED_DEMOS_HASH,
    perlayer_partition_cache_dir,
    perlayer_shuffled_control_dir,
)
from evaluation.interventions.vti.shuffled_control_partition import (  # noqa: E402
    shuffled_demos850_direction_dir,
)
from src.paths import project_root  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parents[1]
DEMOS_HASH = EXPECTED_DEMOS_HASH

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

_GEO_PATH = (
    _PROJECT_ROOT
    / "diagnostic_experiments"
    / "perception_diag"
    / "control"
    / "geometric_comparison"
    / "compare_deployed_vs_shuffled_control.py"
)
_spec = importlib.util.spec_from_file_location(
    "compare_deployed_vs_shuffled_control", _GEO_PATH,
)
_geo = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_geo)
per_layer_cosine = _geo.per_layer_cosine


def _sign_int(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _dirs(model_short: str, scheme: str, arm: str, num_demos: int) -> Path:
    if scheme == "global":
        if arm == "deployed":
            slug = partition_slug(DEMOS_HASH, "all", num_demos, seed=42, rank=2)
            return partition_cache_dir(model_short, slug)
        return shuffled_demos850_direction_dir(model_short, num_demos)
    slug = partition_slug(
        DEMOS_HASH, "all", num_demos, seed=42, rank=2, fit_locus="perlayer",
    )
    if arm == "deployed":
        return perlayer_partition_cache_dir(model_short, slug)
    return perlayer_shuffled_control_dir(model_short, num_demos)


def _load_pc_mean(cache_dir: Path):
    c = np.load(cache_dir / "components.npz")
    pc0 = np.asarray(c["pc0"], dtype=np.float64)
    mean = np.asarray(c["pca_mean_flat"], dtype=np.float64).reshape(pc0.shape)
    return pc0, mean


def cosine_as_is_and_sign_aligned(
    model_short: str, num_demos: int, scheme: str,
) -> List[dict]:
    expected = MODEL_SPECS[model_short]["expected_shape"]
    dep_path = _dirs(model_short, scheme, "deployed", num_demos)
    ctrl_path = _dirs(model_short, scheme, "control", num_demos)
    dep_pc0, dep_mean = _load_pc_mean(dep_path)
    ctrl_pc0, ctrl_mean = _load_pc_mean(ctrl_path)
    n_plus, hidden = dep_pc0.shape
    n_dec = n_plus - 1
    if (n_dec, hidden) != expected:
        raise RuntimeError(
            f"{model_short} {scheme} nd{num_demos}: pc0 decoder shape "
            f"{(n_dec, hidden)} != {expected}"
        )

    dep_dir_asis = (dep_pc0 + dep_mean)[1:]
    ctrl_dir_asis = (ctrl_pc0 + ctrl_mean)[1:]
    cos_asis = per_layer_cosine(dep_dir_asis, ctrl_dir_asis)

    ctrl_pc0_aligned = ctrl_pc0.copy()
    flipped = np.zeros(n_dec, dtype=bool)
    rows = []
    for li in range(n_dec):
        r = li + 1
        d_cos = float(
            np.dot(dep_pc0[r], dep_mean[r])
            / max(np.linalg.norm(dep_pc0[r]) * np.linalg.norm(dep_mean[r]), 1e-12)
        )
        c_cos = float(
            np.dot(ctrl_pc0[r], ctrl_mean[r])
            / max(np.linalg.norm(ctrl_pc0[r]) * np.linalg.norm(ctrl_mean[r]), 1e-12)
        )
        d_sgn = _sign_int(d_cos)
        c_sgn = _sign_int(c_cos)
        agreed = d_sgn == c_sgn
        if not agreed:
            ctrl_pc0_aligned[r] = -ctrl_pc0[r]
            flipped[li] = True
        dep_dir = dep_pc0[r] + dep_mean[r]
        ctrl_dir = ctrl_pc0_aligned[r] + ctrl_mean[r]
        cos_aligned = float(
            np.dot(dep_dir, ctrl_dir)
            / max(np.linalg.norm(dep_dir) * np.linalg.norm(ctrl_dir), 1e-12)
        )
        rows.append({
            "fit_scheme": scheme,
            "num_demos": num_demos,
            "layer": li,
            "cosine_as_is": float(cos_asis[li]),
            "cosine_pc1_sign_aligned": cos_aligned,
            "deployed_pc1_sign": d_sgn,
            "control_pc1_sign_as_is": c_sgn,
            "control_pc1_flipped": bool(flipped[li]),
            "pc1_sign_agreed_as_is": agreed,
        })
    return rows


def write_csv(rows: Sequence[dict], path: Path) -> None:
    fields = [
        "fit_scheme", "num_demos", "layer",
        "cosine_as_is", "cosine_pc1_sign_aligned",
        "deployed_pc1_sign", "control_pc1_sign_as_is",
        "control_pc1_flipped", "pc1_sign_agreed_as_is",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def plot_model(
    model_short: str,
    rows: Sequence[dict],
    path: Path,
    *,
    show_disagreement_bars: bool,
) -> None:
    display = MODEL_SPECS[model_short]["display_name"]
    ns = sorted({r["num_demos"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    axes = axes.ravel()
    titles = {
        50: "50 demo pairs", 100: "100 demo pairs",
        200: "200 demo pairs", 500: "500 demo pairs",
    }
    for ax, n in zip(axes, ns):
        # Only sign-aligned per-layer + global as-is reference (no as-is
        # per-layer overlay — that lives in the primary cosine figure).
        for scheme, key, label, color, ls in (
            ("perlayer", "cosine_pc1_sign_aligned",
             "Per-layer PCA (PC1 signs aligned)", "#2e7d32", "-"),
            ("global", "cosine_as_is",
             "Global PCA (as-is)", "#c45c26", "--"),
        ):
            sub = [
                r for r in rows
                if r["num_demos"] == n and r["fit_scheme"] == scheme
            ]
            sub = sorted(sub, key=lambda r: r["layer"])
            ax.plot(
                [r["layer"] for r in sub],
                [r[key] for r in sub],
                color=color,
                linestyle=ls,
                linewidth=1.6,
                label=label,
            )
        if show_disagreement_bars:
            flips = [
                r for r in rows
                if r["num_demos"] == n
                and r["fit_scheme"] == "perlayer"
                and r["control_pc1_flipped"]
            ]
            for r in flips:
                ax.axvspan(
                    r["layer"] - 0.5, r["layer"] + 0.5,
                    color="#f4a261", alpha=0.25,
                )
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(titles[n])
        ax.set_xlabel("Decoder layer index")
        ax.set_ylabel("Cosine similarity, deployed vs\nshuffled-image control")
        ax.legend(fontsize=7, loc="best")
    bar_note = (
        "Shaded layers: control PC1 was multiplied by −1 so "
        "sign(cos(PC1, mean)) matches deployed. "
        if show_disagreement_bars
        else "Disagreement-layer shading omitted in this variant; "
        "flip flags are in the companion CSV. "
    )
    fig.suptitle(
        f"{display}: deployed vs shuffled-image control cosine — "
        "PC1 sign-aligned between arms (control flipped on disagreement)",
        fontsize=10,
    )
    fig.text(
        0.5, 0.01,
        bar_note
        + "Global as-is for reference. Diagnostic only.",
        ha="center", va="bottom", fontsize=7.5,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SPECS.keys()),
        choices=list(MODEL_SPECS.keys()),
    )
    p.add_argument(
        "--num_demos",
        nargs="+",
        type=int,
        default=list(PARTITION_SIZES),
    )
    p.add_argument(
        "--exp_root",
        type=Path,
        default=EXP_ROOT,
        help="perlayer_pca_control experiment root",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    plots_dir = args.exp_root / "plots" / "pc1_sign_aligned"
    tables_dir = args.exp_root / "tables" / "pc1_sign_aligned"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        print(f"=== {model}")
        all_rows: List[dict] = []
        for scheme in ("global", "perlayer"):
            for n in args.num_demos:
                rows = cosine_as_is_and_sign_aligned(model, n, scheme)
                n_flip = sum(1 for r in rows if r["control_pc1_flipped"])
                print(f"  {scheme} nd{n}: control PC1 flips={n_flip}/{len(rows)}")
                all_rows.extend(rows)
        csv_path = (
            tables_dir
            / f"cosine_deployed_vs_control_pc1_sign_aligned_between_arms_{model}.csv"
        )
        write_csv(all_rows, csv_path)
        # Two plot variants: with and without disagreement-layer shading.
        for shaded, tag in (
            (True, "with_disagreement_bars"),
            (False, "without_disagreement_bars"),
        ):
            png_path = (
                plots_dir
                / (
                    "cosine_deployed_vs_control_by_layer_pc1_sign_aligned_"
                    f"between_arms_{tag}_{model}.png"
                )
            )
            plot_model(
                model, all_rows, png_path, show_disagreement_bars=shaded,
            )
            print(f"  wrote {png_path}")
        # Remove any older single-variant filename if present.
        legacy = (
            plots_dir
            / f"cosine_deployed_vs_control_by_layer_pc1_sign_aligned_between_arms_{model}.png"
        )
        if legacy.is_file():
            legacy.unlink()
            print(f"  removed legacy {legacy.name}")
        print(f"  wrote {csv_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
