#!/usr/bin/env python3
"""
Sanity Check: Text-Only Baseline Positions
==========================================
Test whether SSU text-only (caption + text) activations already sit closer
to the "unsafe" side of the safety boundary than SSS text-only activations.

When --compositional_safety_dir is passed, also projects TT activations onto
the compositional safety direction c^l and reports SSS-vs-SSU baseline
statistics for that direction.

Outputs (under diagnostic_experiments/{model}/shift_dc/outputs/results/
          sanity_check_tt_baseline/)
  - tt_baseline_projections.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
_DIAGNOSTIC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DIAGNOSTIC_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction import (
    ActivationCache, load_activation_matrix, load_json, save_json,
)
from src.model import _normalize_model_name

_EXPERIMENT_NAME = "shift_dc"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--compositional_safety_dir", action="store_true",
                   help="Also compute TT projections onto compositional safety direction c^l")
    return p.parse_args()


def _cosine_rows(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1)
    v_norm = np.linalg.norm(v)
    mask = (norms > 1e-12) & (v_norm > 1e-12)
    return np.where(mask, (X @ v) / (norms * v_norm), np.nan)


def _compute_layer_row(l, s_l, X_sss_tt, X_ssu_tt, prefix):
    """Projection + cosine stats for one direction. prefix='' or prefix='comp_'."""
    s_norm_sq = float(np.dot(s_l, s_l))
    row = {}
    if s_norm_sq < 1e-12:
        return row
    proj_sss = (X_sss_tt @ s_l) / s_norm_sq
    proj_ssu = (X_ssu_tt @ s_l) / s_norm_sq
    t_p, p_p = stats.ttest_ind(proj_sss, proj_ssu, equal_var=False)

    cos_sss = _cosine_rows(X_sss_tt, s_l)
    cos_ssu = _cosine_rows(X_ssu_tt, s_l)
    cos_sss_v = cos_sss[~np.isnan(cos_sss)]
    cos_ssu_v = cos_ssu[~np.isnan(cos_ssu)]
    t_c, p_c = stats.ttest_ind(cos_sss_v, cos_ssu_v, equal_var=False)

    interp_p = "SSU_more_unsafe" if proj_ssu.mean() < proj_sss.mean() else "SSU_not_more_unsafe"
    interp_c = "SSU_more_unsafe" if cos_ssu_v.mean() < cos_sss_v.mean() else "SSU_not_more_unsafe"

    row.update({
        f"SSS_mean_tt_{prefix}projection": float(proj_sss.mean()),
        f"SSU_mean_tt_{prefix}projection": float(proj_ssu.mean()),
        f"SSS_std_{prefix}projection": float(proj_sss.std()),
        f"SSU_std_{prefix}projection": float(proj_ssu.std()),
        f"t_statistic_{prefix}projection": float(t_p),
        f"p_value_{prefix}projection": float(p_p),
        f"interpretation_{prefix}projection": interp_p,
        f"SSS_mean_tt_{prefix}cosine": float(cos_sss_v.mean()),
        f"SSU_mean_tt_{prefix}cosine": float(cos_ssu_v.mean()),
        f"SSS_std_{prefix}cosine": float(cos_sss_v.std()),
        f"SSU_std_{prefix}cosine": float(cos_ssu_v.std()),
        f"t_statistic_{prefix}cosine": float(t_c),
        f"p_value_{prefix}cosine": float(p_c),
        f"interpretation_{prefix}cosine": interp_c,
    })
    return row


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)

    _DATA = _PROJECT_ROOT / "data"
    experiment_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME
    experiment_artifacts = _PROJECT_ROOT / "experiment_artifacts" / model_name / "vl_activation_shift"

    cache = ActivationCache(str(_DATA / "holisafe-bench" / model_name / "activations"))

    safety_vecs = np.load(experiment_artifacts / "safety_direction_vectors.npz")
    layers = sorted(int(k.replace("layer_", "")) for k in safety_vecs.files)

    comp_vecs = None
    if args.compositional_safety_dir:
        comp_path = (_PROJECT_ROOT / "experiment_artifacts" / model_name /
                     "compositional_safety" /
                     "compositional_safety_direction_vectors.npz")
        if not comp_path.exists():
            print(f"ERROR: --compositional_safety_dir requested but "
                  f"{comp_path} does not exist.")
            sys.exit(1)
        comp_vecs = np.load(comp_path)
        print(f"Loaded compositional safety direction for {len(comp_vecs.files)} layers")

    metadata = load_json(str(_DATA / "holisafe-bench" / model_name / "activations" / "sample_metadata.json"))
    sss_ids_all = [s["id"] for s in metadata if s["label"] == "SSS"]
    ssu_ids_all = [s["id"] for s in metadata if s["label"] == "SSU"]
    sss_ids = [sid for sid in sss_ids_all if cache.exists(sid, "tt")]
    ssu_ids = [sid for sid in ssu_ids_all if cache.exists(sid, "tt")]
    n_missing = (len(sss_ids_all) - len(sss_ids)) + (len(ssu_ids_all) - len(ssu_ids))
    if n_missing:
        print(f"  Warning: {n_missing} samples skipped (TT activations not cached)")

    print(f"SSS samples: {len(sss_ids)}, SSU samples: {len(ssu_ids)}")
    print(f"Layers: {len(layers)} (0–{max(layers)})")

    out_dir = experiment_dir / "outputs" / "results" / "sanity_check_tt_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for l in layers:
        s_l = safety_vecs[f"layer_{l}"].astype(np.float64)
        X_sss_tt = load_activation_matrix(cache, sss_ids, l, suffix="tt").astype(np.float64)
        X_ssu_tt = load_activation_matrix(cache, ssu_ids, l, suffix="tt").astype(np.float64)

        row = {"layer": l}
        row.update(_compute_layer_row(l, s_l, X_sss_tt, X_ssu_tt, prefix=""))

        if comp_vecs is not None and f"layer_{l}" in comp_vecs.files:
            c_l = comp_vecs[f"layer_{l}"].astype(np.float64)
            row.update(_compute_layer_row(l, c_l, X_sss_tt, X_ssu_tt, prefix="comp_"))

        if "SSS_mean_tt_projection" in row:
            print(f"  Layer {l:2d}: "
                  f"proj SSS={row['SSS_mean_tt_projection']:.4f} "
                  f"SSU={row['SSU_mean_tt_projection']:.4f} "
                  f"p={row['p_value_projection']:.2e} [{row['interpretation_projection']}]")
        results.append(row)

    save_json(results, str(out_dir / "tt_baseline_projections.json"))

    safety_layers = [r for r in results if 6 <= r["layer"] <= 14]
    n_unsafe_proj = sum(1 for r in safety_layers
                        if r.get("interpretation_projection") == "SSU_more_unsafe")
    print(f"\nSummary (layers 6–14):")
    print(f"  Projection: {n_unsafe_proj}/{len(safety_layers)} layers show SSU more unsafe at baseline")


if __name__ == "__main__":
    main()
