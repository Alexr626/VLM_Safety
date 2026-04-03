#!/usr/bin/env python3
"""
Sanity Check: Text-Only Baseline Positions
==========================================
Test whether SSU text-only (caption + text) activations already sit closer
to the "unsafe" side of the safety boundary than SSS text-only activations.

If yes → the larger modality shift toward "safe" for SSU may be a
         regression-to-mean effect (Hypothesis 1).
If no  → the distortion is caused by the image integration process itself.

Dependencies:
  - shift_dc/outputs/activations/sample_{id}_tt.npz
  - shift_dc/outputs/artifacts/safety_direction_vectors.npz
  - shift_dc/outputs/results/sample_metadata.json

Outputs (under shift_dc/outputs/results/):
  - tt_baseline_projections.json

Usage:
  python diagnostic_experiments/llava-1.5-7b-hf/shift_dc/experiment_scripts/sanity_check_tt_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent                          # shift_dc/
_PROJECT_ROOT = _EXPERIMENT_DIR.parent.parent.parent          # VLM_Safety/
_MODEL_NAME = _EXPERIMENT_DIR.parent.name                     # llava-1.5-7b-hf
_EXPERIMENT_ARTIFACTS = _PROJECT_ROOT / "experiment_artifacts" / _MODEL_NAME / "vl_activation_shift"
sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction import (
    ActivationCache, load_activation_matrix,
    load_json, save_json,
)


def main():
    _DATA = _PROJECT_ROOT / "data"
    cache = ActivationCache(str(_DATA / "holisafe-bench" / "activations" / _MODEL_NAME))

    safety_vecs = np.load(_EXPERIMENT_ARTIFACTS / "safety_direction_vectors.npz")
    layers = sorted(int(k.replace("layer_", "")) for k in safety_vecs.files)

    metadata = load_json(str(_DATA / "holisafe-bench" / "activations" / _MODEL_NAME / "sample_metadata.json"))
    sss_ids_all = [s["id"] for s in metadata if s["label"] == "SSS"]
    ssu_ids_all = [s["id"] for s in metadata if s["label"] == "SSU"]

    # Keep only samples that have TT activations cached
    sss_ids = [sid for sid in sss_ids_all if cache.exists(sid, "tt")]
    ssu_ids = [sid for sid in ssu_ids_all if cache.exists(sid, "tt")]
    n_missing = (len(sss_ids_all) - len(sss_ids)) + (len(ssu_ids_all) - len(ssu_ids))
    if n_missing:
        print(f"  Warning: {n_missing} samples skipped (TT activations not cached)")

    print(f"SSS samples: {len(sss_ids)}, SSU samples: {len(ssu_ids)}")
    print(f"Layers: {len(layers)} (0–{max(layers)})")

    out_dir = _EXPERIMENT_DIR / "outputs" / "results" / "sanity_check_tt_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _cosine_rows(X: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Cosine similarity between each row of X and vector v. Shape: (N,)"""
        norms = np.linalg.norm(X, axis=1)          # (N,)
        v_norm = np.linalg.norm(v)
        mask = (norms > 1e-12) & (v_norm > 1e-12)
        cos = np.where(mask, (X @ v) / (norms * v_norm), np.nan)
        return cos

    results = []
    for l in layers:
        s_l = safety_vecs[f"layer_{l}"].astype(np.float64)
        s_norm_sq = np.dot(s_l, s_l)

        if s_norm_sq < 1e-12:
            continue

        X_sss_tt = load_activation_matrix(cache, sss_ids, l, suffix="tt").astype(np.float64)
        X_ssu_tt = load_activation_matrix(cache, ssu_ids, l, suffix="tt").astype(np.float64)

        # Projection onto safety direction (positive = closer to "safe" centroid)
        proj_sss = (X_sss_tt @ s_l) / s_norm_sq
        proj_ssu = (X_ssu_tt @ s_l) / s_norm_sq

        t_stat_proj, p_val_proj = stats.ttest_ind(proj_sss, proj_ssu, equal_var=False)

        # Cosine similarity with safety direction
        cos_sss = _cosine_rows(X_sss_tt, s_l)
        cos_ssu = _cosine_rows(X_ssu_tt, s_l)

        cos_sss_valid = cos_sss[~np.isnan(cos_sss)]
        cos_ssu_valid = cos_ssu[~np.isnan(cos_ssu)]
        t_stat_cos, p_val_cos = stats.ttest_ind(cos_sss_valid, cos_ssu_valid, equal_var=False)

        interp_proj = "SSU_more_unsafe" if proj_ssu.mean() < proj_sss.mean() else "SSU_not_more_unsafe"
        interp_cos  = "SSU_more_unsafe" if cos_ssu_valid.mean() < cos_sss_valid.mean() else "SSU_not_more_unsafe"

        results.append({
            "layer": l,
            # Projection
            "SSS_mean_tt_projection":   float(proj_sss.mean()),
            "SSU_mean_tt_projection":   float(proj_ssu.mean()),
            "SSS_std_projection":       float(proj_sss.std()),
            "SSU_std_projection":       float(proj_ssu.std()),
            "t_statistic_projection":   float(t_stat_proj),
            "p_value_projection":       float(p_val_proj),
            "interpretation_projection": interp_proj,
            # Cosine similarity
            "SSS_mean_tt_cosine":       float(cos_sss_valid.mean()),
            "SSU_mean_tt_cosine":       float(cos_ssu_valid.mean()),
            "SSS_std_cosine":           float(cos_sss_valid.std()),
            "SSU_std_cosine":           float(cos_ssu_valid.std()),
            "t_statistic_cosine":       float(t_stat_cos),
            "p_value_cosine":           float(p_val_cos),
            "interpretation_cosine":    interp_cos,
        })

        agree = "✓" if interp_proj == interp_cos else "✗"
        print(f"  Layer {l:2d}: "
              f"proj SSS={proj_sss.mean():.4f} SSU={proj_ssu.mean():.4f} p={p_val_proj:.2e} [{interp_proj}]  |  "
              f"cos SSS={cos_sss_valid.mean():.4f} SSU={cos_ssu_valid.mean():.4f} p={p_val_cos:.2e} [{interp_cos}]  "
              f"{agree}")

    save_json(results, str(out_dir / "tt_baseline_projections.json"))

    # Summary for safety-critical layers
    safety_layers = [r for r in results if 6 <= r["layer"] <= 14]
    n_unsafe_proj = sum(1 for r in safety_layers if r["interpretation_projection"] == "SSU_more_unsafe")
    n_unsafe_cos  = sum(1 for r in safety_layers if r["interpretation_cosine"]     == "SSU_more_unsafe")
    n_agree       = sum(1 for r in safety_layers
                        if r["interpretation_projection"] == r["interpretation_cosine"])
    print(f"\nSummary (layers 6–14):")
    print(f"  Projection : {n_unsafe_proj}/{len(safety_layers)} layers show SSU more unsafe at baseline")
    print(f"  Cosine     : {n_unsafe_cos}/{len(safety_layers)} layers show SSU more unsafe at baseline")
    print(f"  Agreement  : {n_agree}/{len(safety_layers)} layers agree between the two metrics")


if __name__ == "__main__":
    main()
