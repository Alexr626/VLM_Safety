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
  - activations/sample_{id}_tt.npz
  - method2_activation_shift/safety_direction_vectors.npz
  - method2_activation_shift/sample_metadata.json

Outputs (under outputs/{model_name}/sanity_check_tt_baseline/):
  - tt_baseline_projections.json

Usage:
  cd ~/dev/VLM_Safety
  python diagnostic_exploratory_tests/followup_subspace_analysis/sanity_check_tt_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction import (
    ActivationCache, load_activation_matrix,
    load_json, save_json,
)


def main():
    OUTPUT_ROOT = _PROJECT_ROOT / "diagnostic_exploratory_tests" / "outputs" / "llava-1.5-7b-hf"
    cache = ActivationCache(str(OUTPUT_ROOT / "activations"))

    # Load safety directions
    safety_vecs = np.load(OUTPUT_ROOT / "method2_activation_shift" / "safety_direction_vectors.npz")
    layers = sorted(int(k.replace("layer_", "")) for k in safety_vecs.files)

    # Load sample metadata
    metadata = load_json(str(OUTPUT_ROOT / "method2_activation_shift" / "sample_metadata.json"))
    sss_ids = [s["id"] for s in metadata if s["label"] == "SSS"]
    ssu_ids = [s["id"] for s in metadata if s["label"] == "SSU"]

    print(f"SSS samples: {len(sss_ids)}, SSU samples: {len(ssu_ids)}")
    print(f"Layers: {len(layers)} (0–{max(layers)})")

    out_dir = OUTPUT_ROOT / "sanity_check_tt_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for l in layers:
        s_l = safety_vecs[f"layer_{l}"].astype(np.float64)
        s_norm_sq = np.dot(s_l, s_l)

        if s_norm_sq < 1e-12:
            continue

        X_sss_tt = load_activation_matrix(cache, sss_ids, l, suffix="tt")
        X_ssu_tt = load_activation_matrix(cache, ssu_ids, l, suffix="tt")

        # Project onto safety direction (positive = closer to "safe" centroid)
        proj_sss = (X_sss_tt.astype(np.float64) @ s_l) / s_norm_sq
        proj_ssu = (X_ssu_tt.astype(np.float64) @ s_l) / s_norm_sq

        t_stat, p_val = stats.ttest_ind(proj_sss, proj_ssu, equal_var=False)

        interp = "SSU_more_unsafe" if proj_ssu.mean() < proj_sss.mean() else "SSU_not_more_unsafe"
        results.append({
            "layer": l,
            "SSS_mean_tt_projection": float(proj_sss.mean()),
            "SSU_mean_tt_projection": float(proj_ssu.mean()),
            "SSS_std": float(proj_sss.std()),
            "SSU_std": float(proj_ssu.std()),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "interpretation": interp,
        })

        print(f"  Layer {l:2d}: SSS={proj_sss.mean():.4f}  SSU={proj_ssu.mean():.4f}  "
              f"p={p_val:.2e}  → {interp}")

    save_json(results, str(out_dir / "tt_baseline_projections.json"))

    # Summary for safety-critical layers
    safety_layers = [r for r in results if 6 <= r["layer"] <= 14]
    n_unsafe = sum(1 for r in safety_layers if r["interpretation"] == "SSU_more_unsafe")
    print(f"\nSummary (layers 6–14): {n_unsafe}/{len(safety_layers)} layers show "
          f"SSU more unsafe at baseline")


if __name__ == "__main__":
    main()
