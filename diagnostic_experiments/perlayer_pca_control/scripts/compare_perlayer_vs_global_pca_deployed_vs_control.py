#!/usr/bin/env python3
"""Compare per-layer vs global PCA: deployed vs shuffled-image control cosine.

CPU-only. Loads 32 direction objects (2 models × 4 N × 2 arms × 2 schemes),
writes per-layer CSV/plots and a facts-only summary. Refuses to start unless
the byte-identity report exists and records all-match.

Example::

    python diagnostic_experiments/perlayer_pca_control/scripts/\\
        compare_perlayer_vs_global_pca_deployed_vs_control.py \\
      --models llava-1.5-7b-hf qwen2.5-vl-7b-instruct \\
      --num_demos 50 100 200 500
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from evaluation.interventions.vti.directions_v2 import (  # noqa: E402
    load_textual_v2_directions,
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
VERIFICATION_DIR = EXP_ROOT / "verification"
PLOTS_DIR = EXP_ROOT / "plots"
TABLES_DIR = EXP_ROOT / "tables"
SUMMARIES_DIR = EXP_ROOT / "summaries"
BYTE_IDENTITY_REPORT = (
    VERIFICATION_DIR / "global_fit_directions_byte_identity_report.md"
)
DEMOS_HASH = EXPECTED_DEMOS_HASH

MODEL_SPECS = {
    "llava-1.5-7b-hf": {
        "display_name": "LLaVA-1.5-7B",
        "expected_shape": (32, 4096),
        "bands": (("0-10", range(0, 11)), ("18-31", range(18, 32))),
    },
    "qwen2.5-vl-7b-instruct": {
        "display_name": "Qwen2.5-VL-7B",
        "expected_shape": (28, 3584),
        "bands": (("0-10", range(0, 11)), ("18-27", range(18, 28))),
    },
}

# Import shared cosine helper from the demos_v2 geometric comparison script.
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


def _require_byte_identity() -> None:
    if not BYTE_IDENTITY_REPORT.is_file():
        raise SystemExit(
            f"Refusing to start: missing {BYTE_IDENTITY_REPORT}. "
            "Run helper_scripts/verify_perlayer_pca_control_extraction.py "
            "--mode verify first."
        )
    text = BYTE_IDENTITY_REPORT.read_text()
    if "**All global-fit SHA-256 match:** yes" not in text:
        raise SystemExit(
            f"Refusing to start: {BYTE_IDENTITY_REPORT.name} does not record "
            "all-match. Do not compute cosines against possibly overwritten "
            "artifacts."
        )


def _dirs_for(
    model_short: str, scheme: str, arm: str, num_demos: int,
) -> Path:
    if scheme == "global":
        if arm == "deployed":
            slug = partition_slug(DEMOS_HASH, "all", num_demos, seed=42, rank=2)
            return partition_cache_dir(model_short, slug)
        return shuffled_demos850_direction_dir(model_short, num_demos)
    if scheme == "perlayer":
        if arm == "deployed":
            slug = partition_slug(
                DEMOS_HASH, "all", num_demos, seed=42, rank=2, fit_locus="perlayer",
            )
            return perlayer_partition_cache_dir(model_short, slug)
        return perlayer_shuffled_control_dir(model_short, num_demos)
    raise ValueError(scheme)


def _sign_int(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _pc1_explained_variance(meta: dict, scheme: str, layer: int) -> float:
    if scheme == "perlayer":
        vals = meta.get("pc1_explained_variance_per_layer") or []
        # metadata stores embedding + decoder; layer is decoder index
        if len(vals) == EXPECTED_DECODER_SHAPES[meta["model_short"]][0] + 1:
            return float(vals[layer + 1])
        if len(vals) > layer + 1:
            return float(vals[layer + 1])
        return float("nan")
    # global: single scalar for the whole fit
    v = meta.get("pc1_explained_variance")
    return float(v) if v is not None else float("nan")


def analyze_model_nd(
    model_short: str, num_demos: int, scheme: str,
) -> Dict:
    expected = MODEL_SPECS[model_short]["expected_shape"]
    dep_path = _dirs_for(model_short, scheme, "deployed", num_demos)
    ctrl_path = _dirs_for(model_short, scheme, "control", num_demos)
    deployed, dep_meta = load_textual_v2_directions(dep_path)
    control, ctrl_meta = load_textual_v2_directions(ctrl_path)
    if deployed.shape != expected or control.shape != expected:
        raise RuntimeError(
            f"{model_short} {scheme} nd{num_demos}: shapes "
            f"{deployed.shape}/{control.shape} != {expected}"
        )

    n_layers = deployed.shape[0]
    dep_norm = np.linalg.norm(deployed.astype(np.float64), axis=1)
    ctrl_norm = np.linalg.norm(control.astype(np.float64), axis=1)
    cosine = per_layer_cosine(deployed, control)

    for label, arr, meta in (
        ("deployed", dep_norm, dep_meta),
        ("control", ctrl_norm, ctrl_meta),
    ):
        meta_norms = meta.get("direction_layer_norms")
        if meta_norms and len(meta_norms) == n_layers + 1:
            meta_dec = np.asarray(meta_norms[1:], dtype=np.float64)
            max_abs = float(np.max(np.abs(arr - meta_dec)))
            if max_abs > 1e-4:
                raise RuntimeError(
                    f"{model_short} {scheme} nd{num_demos} {label}: "
                    f"norm metadata mismatch max_abs={max_abs}"
                )

    dep_c = np.load(dep_path / "components.npz")
    ctrl_c = np.load(ctrl_path / "components.npz")
    dep_pc0 = np.asarray(dep_c["pc0"], dtype=np.float64)
    ctrl_pc0 = np.asarray(ctrl_c["pc0"], dtype=np.float64)
    dep_mean = np.asarray(dep_c["pca_mean_flat"], dtype=np.float64).reshape(dep_pc0.shape)
    ctrl_mean = np.asarray(ctrl_c["pca_mean_flat"], dtype=np.float64).reshape(ctrl_pc0.shape)
    dep_pc1 = np.linalg.norm(dep_pc0[1:], axis=1)
    ctrl_pc1 = np.linalg.norm(ctrl_pc0[1:], axis=1)
    dep_share = dep_pc1 / np.maximum(dep_norm, 1e-12)
    ctrl_share = ctrl_pc1 / np.maximum(ctrl_norm, 1e-12)

    rows = []
    for li in range(n_layers):
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
        rows.append({
            "fit_scheme": scheme,
            "num_demos": num_demos,
            "layer": li,
            "cosine_deployed_vs_control": float(cosine[li]),
            "deployed_direction_l2": float(dep_norm[li]),
            "control_direction_l2": float(ctrl_norm[li]),
            "deployed_pc1_l2": float(dep_pc1[li]),
            "control_pc1_l2": float(ctrl_pc1[li]),
            "deployed_pc1_share_of_direction_norm": float(dep_share[li]),
            "control_pc1_share_of_direction_norm": float(ctrl_share[li]),
            "deployed_pc1_explained_variance": _pc1_explained_variance(
                dep_meta, scheme, li,
            ),
            "control_pc1_explained_variance": _pc1_explained_variance(
                ctrl_meta, scheme, li,
            ),
            "deployed_cosine_pc1_with_mean": d_cos,
            "control_cosine_pc1_with_mean": c_cos,
            "deployed_pc1_sign": d_sgn,
            "control_pc1_sign": c_sgn,
            "pc1_sign_agrees_between_arms": bool(d_sgn == c_sgn),
        })
    return {
        "model_short": model_short,
        "scheme": scheme,
        "num_demos": num_demos,
        "deployed_path": str(dep_path.relative_to(project_root())),
        "control_path": str(ctrl_path.relative_to(project_root())),
        "rows": rows,
        "dep_meta": dep_meta,
        "ctrl_meta": ctrl_meta,
    }


def band_summary(rows: Sequence[dict], model_short: str) -> List[dict]:
    out = []
    by_key: Dict[Tuple[str, int], List[dict]] = {}
    for r in rows:
        by_key.setdefault((r["fit_scheme"], r["num_demos"]), []).append(r)
    for (scheme, n), group in sorted(by_key.items()):
        for band_name, band_range in MODEL_SPECS[model_short]["bands"]:
            vals = [
                g["cosine_deployed_vs_control"]
                for g in group
                if g["layer"] in band_range
            ]
            arr = np.asarray(vals, dtype=np.float64)
            out.append({
                "fit_scheme": scheme,
                "num_demos": n,
                "layer_band": band_name,
                "median_cosine": float(np.median(arr)),
                "mean_cosine": float(np.mean(arr)),
                "min_cosine": float(np.min(arr)),
                "max_cosine": float(np.max(arr)),
                "n_layers_in_band": len(arr),
            })
    return out


def write_layer_csv(rows: Sequence[dict], path: Path) -> None:
    fields = [
        "fit_scheme", "num_demos", "layer", "cosine_deployed_vs_control",
        "deployed_direction_l2", "control_direction_l2",
        "deployed_pc1_l2", "control_pc1_l2",
        "deployed_pc1_share_of_direction_norm",
        "control_pc1_share_of_direction_norm",
        "deployed_pc1_explained_variance", "control_pc1_explained_variance",
        "deployed_cosine_pc1_with_mean", "control_cosine_pc1_with_mean",
        "deployed_pc1_sign", "control_pc1_sign", "pc1_sign_agrees_between_arms",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def write_band_csv(rows: Sequence[dict], path: Path) -> None:
    fields = [
        "fit_scheme", "num_demos", "layer_band",
        "median_cosine", "mean_cosine", "min_cosine", "max_cosine",
        "n_layers_in_band",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def plot_cosine(model_short: str, rows: Sequence[dict], path: Path) -> None:
    display = MODEL_SPECS[model_short]["display_name"]
    ns = sorted({r["num_demos"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    axes = axes.ravel()
    panel_titles = {
        50: "50 demo pairs",
        100: "100 demo pairs",
        200: "200 demo pairs",
        500: "500 demo pairs",
    }
    for ax, n in zip(axes, ns):
        for scheme, label, color, ls in (
            ("perlayer", "Per-layer PCA fit", "#1f4e79", "-"),
            ("global", "Global PCA fit", "#c45c26", "--"),
        ):
            sub = [r for r in rows if r["num_demos"] == n and r["fit_scheme"] == scheme]
            sub = sorted(sub, key=lambda r: r["layer"])
            ax.plot(
                [r["layer"] for r in sub],
                [r["cosine_deployed_vs_control"] for r in sub],
                color=color,
                linestyle=ls,
                linewidth=1.6,
                label=label,
            )
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(panel_titles[n])
        ax.set_xlabel("Decoder layer index")
        ax.set_ylabel("Cosine similarity, deployed vs\nshuffled-image control")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        f"{display}: deployed direction vs shuffled-image control, "
        "per-layer PCA fit and global PCA fit",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_norms(model_short: str, rows: Sequence[dict], path: Path) -> None:
    display = MODEL_SPECS[model_short]["display_name"]
    ns = sorted({r["num_demos"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    panel_titles = {
        50: "50 demo pairs", 100: "100 demo pairs",
        200: "200 demo pairs", 500: "500 demo pairs",
    }
    series = [
        ("perlayer", "deployed_direction_l2", "Per-layer deployed", "#1f4e79", "-"),
        ("perlayer", "control_direction_l2", "Per-layer control", "#1f4e79", "--"),
        ("global", "deployed_direction_l2", "Global deployed", "#c45c26", "-"),
        ("global", "control_direction_l2", "Global control", "#c45c26", "--"),
    ]
    for ax, n in zip(axes, ns):
        for scheme, key, label, color, ls in series:
            sub = [r for r in rows if r["num_demos"] == n and r["fit_scheme"] == scheme]
            sub = sorted(sub, key=lambda r: r["layer"])
            ax.plot(
                [r["layer"] for r in sub],
                [r[key] for r in sub],
                color=color,
                linestyle=ls,
                linewidth=1.4,
                label=label,
            )
        ax.set_title(panel_titles[n])
        ax.set_xlabel("Decoder layer index")
        ax.set_ylabel("Direction L2 norm")
        ax.legend(fontsize=6, loc="best")
    fig.suptitle(
        f"{display}: per-layer direction magnitude, deployed and "
        "shuffled-image control, under both PCA fit schemes",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pc1_share(model_short: str, rows: Sequence[dict], path: Path) -> None:
    display = MODEL_SPECS[model_short]["display_name"]
    ns = sorted({r["num_demos"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    panel_titles = {
        50: "50 demo pairs", 100: "100 demo pairs",
        200: "200 demo pairs", 500: "500 demo pairs",
    }
    series = [
        ("perlayer", "deployed_pc1_share_of_direction_norm",
         "Per-layer deployed", "#1f4e79", "-"),
        ("perlayer", "control_pc1_share_of_direction_norm",
         "Per-layer control", "#1f4e79", "--"),
        ("global", "deployed_pc1_share_of_direction_norm",
         "Global deployed", "#c45c26", "-"),
        ("global", "control_pc1_share_of_direction_norm",
         "Global control", "#c45c26", "--"),
    ]
    for ax, n in zip(axes, ns):
        for scheme, key, label, color, ls in series:
            sub = [r for r in rows if r["num_demos"] == n and r["fit_scheme"] == scheme]
            sub = sorted(sub, key=lambda r: r["layer"])
            ax.plot(
                [r["layer"] for r in sub],
                [r[key] for r in sub],
                color=color,
                linestyle=ls,
                linewidth=1.4,
                label=label,
            )
        ax.set_title(panel_titles[n])
        ax.set_xlabel("Decoder layer index")
        ax.set_ylabel("PC1 L2 norm divided by direction L2 norm")
        ax.legend(fontsize=6, loc="best")
    fig.suptitle(
        f"{display}: PC1 share of direction norm by layer, both PCA fit schemes",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sign_agreement(model_short: str, rows: Sequence[dict], path: Path) -> None:
    display = MODEL_SPECS[model_short]["display_name"]
    ns = sorted({r["num_demos"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    axes = axes.ravel()
    panel_titles = {
        50: "50 demo pairs", 100: "100 demo pairs",
        200: "200 demo pairs", 500: "500 demo pairs",
    }
    for ax, n in zip(axes, ns):
        pl = [r for r in rows if r["num_demos"] == n and r["fit_scheme"] == "perlayer"]
        gl = [r for r in rows if r["num_demos"] == n and r["fit_scheme"] == "global"]
        pl = sorted(pl, key=lambda r: r["layer"])
        gl = sorted(gl, key=lambda r: r["layer"])
        layers = [r["layer"] for r in pl]
        # Global signs are constant across layers by construction; draw flat refs.
        if gl:
            ax.axhline(
                gl[0]["deployed_pc1_sign"], color="#c45c26", linestyle="-",
                linewidth=1.0, alpha=0.7, label="Global deployed sign",
            )
            ax.axhline(
                gl[0]["control_pc1_sign"], color="#c45c26", linestyle="--",
                linewidth=1.0, alpha=0.7, label="Global control sign",
            )
        ax.step(
            layers,
            [r["deployed_pc1_sign"] for r in pl],
            where="mid",
            color="#1f4e79",
            linewidth=1.5,
            label="Per-layer deployed sign",
        )
        ax.step(
            layers,
            [r["control_pc1_sign"] for r in pl],
            where="mid",
            color="#2e7d32",
            linewidth=1.5,
            label="Per-layer control sign",
        )
        disagree = [r for r in pl if not r["pc1_sign_agrees_between_arms"]]
        for r in disagree:
            ax.axvspan(r["layer"] - 0.5, r["layer"] + 0.5, color="#f4a261", alpha=0.35)
        if disagree:
            ax.scatter(
                [r["layer"] for r in disagree],
                [r["deployed_pc1_sign"] for r in disagree],
                s=36,
                facecolors="#f4a261",
                edgecolors="#9a3412",
                zorder=5,
                label="Arms disagree",
            )
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim(-1.25, 1.25)
        ax.set_title(panel_titles[n])
        ax.set_xlabel("Decoder layer index")
        ax.set_ylabel("sign(cos(PC1, mean))")
        ax.legend(fontsize=6, loc="best")
    fig.suptitle(
        f"{display}: PC1 orientation per layer, deployed and shuffled-image "
        "control, per-layer PCA fit against global PCA fit",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(
    model_results: Dict[str, dict],
    gating_manifest: Path,
    out_path: Path,
) -> None:
    lines = [
        "# Per-layer vs global PCA: deployed vs shuffled-image control",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        "Item set: demos850 `all`, disjoint partition blocks nd∈{50,100,200,500}.",
        "Stage: geometric only (no model load, no steering, no benchmark).",
        "",
        "## Input paths",
        "",
    ]
    for model, res in model_results.items():
        lines.append(f"### `{model}`")
        lines.append("")
        for key in sorted(res["path_sha"].keys()):
            lines.append(f"- `{key}`: `{res['path_sha'][key]}`")
        lines.append("")

    lines += [
        "## Gating checks (1–7)",
        "",
    ]
    if gating_manifest.is_file():
        man = json.loads(gating_manifest.read_text())
        lines += [
            "| Check | Pass | Detail |",
            "|---|---|---|",
        ]
        for g in man.get("gating", []):
            lines.append(
                f"| {g['check']} | {'PASS' if g['pass'] else 'FAIL'} | "
                f"{g['detail'][:120]} |"
            )
        lines.append("")
        # Check 8 summary
        lines += ["## Check 8 — PC1 share of direction norm (report)", ""]
        lines += [
            "| Model | Scheme | Arm | N | min | median | max | "
            "descending steps | largest descent |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in man.get("check8_pc1_share", []):
            lines.append(
                f"| {row['model']} | {row['scheme']} | {row['arm']} | "
                f"{row['num_demos']} | {row['pc1_share_min']:.6g} | "
                f"{row['pc1_share_median']:.6g} | {row['pc1_share_max']:.6g} | "
                f"{row['direction_norm_descending_steps']} | "
                f"{row['largest_descending_step']:.6g} |"
            )
        lines.append("")
        lines += [
            "Spec stated expectation (not a gate): LLaVA PC1 share ~9% global → "
            "20–72% per-layer; Qwen ~0.8% → 1.5–11%. Observed values are in the "
            "table above and the per-layer CSV.",
            "",
            "## Check 9 — PC1 sign agreement (report)",
            "",
            "Per-layer `sign(cos(PC1, mean))` for each arm and whether arms agree. "
            "No correction applied.",
            "",
        ]
        # Compact: count disagreements per (model, scheme, N)
        disagree_counts: Dict[Tuple, int] = {}
        for row in man.get("check9_pc1_sign", []):
            key = (row["model"], row["scheme"], row["num_demos"])
            if not row["pc1_sign_agrees_between_arms"]:
                disagree_counts[key] = disagree_counts.get(key, 0) + 1
        lines += [
            "| Model | Scheme | N | Layers where arms disagree |",
            "|---|---|---:|---:|",
        ]
        for model in MODEL_SPECS:
            for scheme in ("global", "perlayer"):
                for n in PARTITION_SIZES:
                    key = (model, scheme, n)
                    lines.append(
                        f"| {model} | {scheme} | {n} | "
                        f"{disagree_counts.get(key, 0)} |"
                    )
        lines.append("")
    else:
        lines.append(f"(gating manifest missing: `{gating_manifest}`)")
        lines.append("")

    lines += ["## Per-layer cosine tables", ""]
    for model, res in model_results.items():
        display = MODEL_SPECS[model]["display_name"]
        lines.append(f"### {display}")
        lines.append("")
        for scheme in ("perlayer", "global"):
            for n in PARTITION_SIZES:
                sub = [
                    r for r in res["rows"]
                    if r["fit_scheme"] == scheme and r["num_demos"] == n
                ]
                sub = sorted(sub, key=lambda r: r["layer"])
                lines.append(f"#### {scheme}, nd={n}")
                lines.append("")
                lines.append("| Layer | Cosine | Deployed L2 | Control L2 |")
                lines.append("|---:|---:|---:|---:|")
                for r in sub:
                    lines.append(
                        f"| {r['layer']} | {r['cosine_deployed_vs_control']:.6g} | "
                        f"{r['deployed_direction_l2']:.6g} | "
                        f"{r['control_direction_l2']:.6g} |"
                    )
                lines.append("")

    lines += ["## Layer-band summary", ""]
    for model, res in model_results.items():
        display = MODEL_SPECS[model]["display_name"]
        lines.append(f"### {display}")
        lines.append("")
        lines.append(
            "| Scheme | N | Band | Median | Mean | Min | Max | n_layers |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
        for b in res["bands"]:
            lines.append(
                f"| {b['fit_scheme']} | {b['num_demos']} | {b['layer_band']} | "
                f"{b['median_cosine']:.6g} | {b['mean_cosine']:.6g} | "
                f"{b['min_cosine']:.6g} | {b['max_cosine']:.6g} | "
                f"{b['n_layers_in_band']} |"
            )
        lines.append("")

    lines += ["## Artifacts", ""]
    for model in model_results:
        lines.append(f"- `{model}`:")
        lines.append(
            f"  - `tables/cosine_and_norms_deployed_vs_control_by_layer_{model}.csv`"
        )
        lines.append(
            f"  - `tables/cosine_by_layer_band_and_sample_size_{model}.csv`"
        )
        lines.append(
            f"  - `plots/cosine_deployed_vs_control_by_layer_per_layer_and_global_pca_{model}.png`"
        )
        lines.append(
            f"  - `plots/direction_l2_norm_by_layer_deployed_and_control_per_layer_and_global_pca_{model}.png`"
        )
        lines.append(
            f"  - `plots/pc1_share_of_direction_norm_by_layer_per_layer_and_global_pca_{model}.png`"
        )
        lines.append(
            f"  - `plots/pc1_sign_agreement_by_layer_per_layer_and_global_pca_{model}.png`"
        )
    lines.append("- `summaries/perlayer_vs_global_pca_control_summary.md` (this file)")
    lines.append("- `verification/` — manifests and byte-identity / cache reports")
    lines.append("- `scripts/` — analysis entry points")
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
        "--num_demos",
        nargs="+",
        type=int,
        default=list(PARTITION_SIZES),
    )
    p.add_argument(
        "--exp_root",
        type=Path,
        default=EXP_ROOT,
        help="perlayer_pca_control experiment root (contains plots/tables/…)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _require_byte_identity()
    exp_root = args.exp_root
    plots_dir = exp_root / "plots"
    tables_dir = exp_root / "tables"
    summaries_dir = exp_root / "summaries"
    verification_dir = exp_root / "verification"
    for d in (plots_dir, tables_dir, summaries_dir, verification_dir):
        d.mkdir(parents=True, exist_ok=True)

    import hashlib

    model_results: Dict[str, dict] = {}
    for model in args.models:
        print(f"=== {model}")
        all_rows: List[dict] = []
        path_sha: Dict[str, str] = {}
        for scheme in ("global", "perlayer"):
            for n in args.num_demos:
                result = analyze_model_nd(model, n, scheme)
                all_rows.extend(result["rows"])
                for label, p in (
                    ("deployed", result["deployed_path"]),
                    ("control", result["control_path"]),
                ):
                    full = project_root() / p / "directions.npz"
                    key = f"{scheme}/{label}/nd{n}"
                    path_sha[key] = (
                        f"{p}/directions.npz "
                        f"sha256={hashlib.sha256(full.read_bytes()).hexdigest()}"
                    )
                print(
                    f"  {scheme} nd{n}: "
                    f"cosine median="
                    f"{np.median([r['cosine_deployed_vs_control'] for r in result['rows']]):.4f}"
                )
        bands = band_summary(all_rows, model)
        csv_path = tables_dir / f"cosine_and_norms_deployed_vs_control_by_layer_{model}.csv"
        band_path = tables_dir / f"cosine_by_layer_band_and_sample_size_{model}.csv"
        write_layer_csv(all_rows, csv_path)
        write_band_csv(bands, band_path)
        plot_cosine(
            model, all_rows,
            plots_dir / f"cosine_deployed_vs_control_by_layer_per_layer_and_global_pca_{model}.png",
        )
        plot_norms(
            model, all_rows,
            plots_dir
            / f"direction_l2_norm_by_layer_deployed_and_control_per_layer_and_global_pca_{model}.png",
        )
        plot_pc1_share(
            model, all_rows,
            plots_dir
            / f"pc1_share_of_direction_norm_by_layer_per_layer_and_global_pca_{model}.png",
        )
        plot_sign_agreement(
            model, all_rows,
            plots_dir
            / f"pc1_sign_agreement_by_layer_per_layer_and_global_pca_{model}.png",
        )
        model_results[model] = {
            "rows": all_rows,
            "bands": bands,
            "path_sha": path_sha,
        }
        print(f"  wrote CSVs and plots for {model}")

    gating_manifest = verification_dir / "perlayer_pca_extraction_manifest_2026-07-29.json"
    summary_path = summaries_dir / "perlayer_vs_global_pca_control_summary.md"
    write_summary(model_results, gating_manifest, summary_path)
    print(f"Wrote {summary_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
