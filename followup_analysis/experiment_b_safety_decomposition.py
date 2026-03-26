#!/usr/bin/env python3
"""
Experiment B: Decompose Modality Shift Along Multiple Safety Dimensions
=======================================================================
Extract a multi-dimensional safety subspace via SVD and decompose
modality shifts along each principal component. Tests whether the
SSU-vs-SSS distortion is concentrated in the dominant safety direction
or spread across non-dominant dimensions.

Methodology:
  For each layer, compute D_safe = X_safe - mu_unsafe (how each safe
  reference sample deviates from the unsafe centroid), mean-center, then
  SVD. The top-k right singular vectors form the safety subspace basis.
  Component 0 should closely match Method 2's single safety direction s^l.

Dependencies:
  - activations/sample_{id}_vl.npz and sample_{id}_tt.npz
  - method2_activation_shift/sample_metadata.json
  - method2_activation_shift/reference_activation_matrices.npz
  - method2_activation_shift/safety_direction_vectors.npz

Outputs (under outputs/{model_name}/experiment_b_safety_decomposition/):
  - safety_subspace_bases.npz
  - safety_subspace_spectra.npz
  - per_component_projections.json
  - dominant_vs_method2_cosine.json

Usage:
  cd ~/dev/VLM_Safety
  python diagnostic_exploratory_tests/followup_subspace_analysis/experiment_b_safety_decomposition.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction import (
    ActivationCache, load_modality_shift_matrix,
    load_json, save_json, save_npz,
)

K_SAFETY = 10


def extract_safety_subspace(ref_mats, layer, k):
    """Extract safety subspace at a given layer from reference activation matrices.

    Uses the deviation-from-unsafe-centroid approach:
      D_safe = X_safe - mu_unsafe, then SVD of mean-centered D_safe.

    Returns:
        Vt_k: (k, d) orthonormal basis
        sigmas_k: (k,) singular values
    """
    X_safe = ref_mats[f"safe_layer_{layer}"].astype(np.float64)
    X_unsafe = ref_mats[f"unsafe_layer_{layer}"].astype(np.float64)

    mu_unsafe = X_unsafe.mean(axis=0)
    D_safe = X_safe - mu_unsafe
    D_safe_centered = D_safe - D_safe.mean(axis=0, keepdims=True)

    _, sigmas, Vt = np.linalg.svd(D_safe_centered, full_matrices=False)
    k_actual = min(k, len(sigmas))
    return Vt[:k_actual], sigmas[:k_actual]


def main():
    OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "llava-1.5-7b-hf"
    cache = ActivationCache(str(OUTPUT_ROOT / "activations" / "holisafe"))

    metadata = load_json(str(OUTPUT_ROOT / "method2_shiftdc" / "sample_metadata.json"))
    sss_ids = [s["id"] for s in metadata if s["label"] == "SSS"]
    ssu_ids = [s["id"] for s in metadata if s["label"] == "SSU"]

    ref_mats = {
        **dict(np.load(OUTPUT_ROOT / "reference" / "llava-instruct"  / "activation_matrices.npz")),
        **dict(np.load(OUTPUT_ROOT / "reference" / "mm-safetybench" / "activation_matrices.npz")),
    }
    safety_vecs = np.load(OUTPUT_ROOT / "method2_shiftdc" / "safety_direction_vectors.npz")
    layers = sorted(int(k.replace("layer_", "")) for k in safety_vecs.files)

    print(f"SSS: {len(sss_ids)}, SSU: {len(ssu_ids)}, Layers: {len(layers)}")

    out_dir = OUTPUT_ROOT / "experiment_b_safety_decomposition"
    out_dir.mkdir(parents=True, exist_ok=True)

    subspace_bases = {}
    subspace_spectra = {}
    dominant_cosine = []
    projection_results = []

    for l in layers:
        safe_key = f"safe_layer_{l}"
        if safe_key not in ref_mats:
            continue

        print(f"  Layer {l:2d}...", end=" ", flush=True)

        Vt_k, sigmas_k = extract_safety_subspace(ref_mats, l, K_SAFETY)
        k = Vt_k.shape[0]

        subspace_bases[f"layer_{l}"] = Vt_k.astype(np.float32)
        subspace_spectra[f"layer_{l}"] = sigmas_k.astype(np.float32)

        # Sanity check: cosine between dominant component and Method 2's s^l
        s_l = safety_vecs[f"layer_{l}"].astype(np.float64)
        v0 = Vt_k[0]
        norm_prod = np.linalg.norm(v0) * np.linalg.norm(s_l)
        cos_dom = float(np.dot(v0, s_l) / norm_prod) if norm_prod > 1e-12 else 0.0
        dominant_cosine.append({"layer": l, "cosine_sim_dominant_vs_s_l": cos_dom})

        # Project modality shifts onto each safety dimension
        M_sss = load_modality_shift_matrix(cache, sss_ids, l).astype(np.float64)
        M_ssu = load_modality_shift_matrix(cache, ssu_ids, l).astype(np.float64)

        total_var = float((sigmas_k ** 2).sum())

        for ki in range(k):
            v_ki = Vt_k[ki].astype(np.float64)

            proj_sss = M_sss @ v_ki
            proj_ssu = M_ssu @ v_ki

            t_stat, p_val = stats.ttest_ind(proj_sss, proj_ssu, equal_var=False)

            projection_results.append({
                "layer": l,
                "component_k": ki,
                "variance_explained_frac": float(sigmas_k[ki] ** 2 / total_var) if total_var > 0 else 0.0,
                "SSS_mean_proj": float(proj_sss.mean()),
                "SSU_mean_proj": float(proj_ssu.mean()),
                "SSS_std": float(proj_sss.std()),
                "SSU_std": float(proj_ssu.std()),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
            })

        var_pcts = [f"{sigmas_k[i]**2/total_var:.1%}" for i in range(min(3, k))]
        print(f"cos(v0,s^l)={cos_dom:.4f}  top-3 var%: {', '.join(var_pcts)}")

    save_npz(subspace_bases, str(out_dir / "safety_subspace_bases.npz"))
    save_npz(subspace_spectra, str(out_dir / "safety_subspace_spectra.npz"))
    save_json(projection_results, str(out_dir / "per_component_projections.json"))
    save_json(dominant_cosine, str(out_dir / "dominant_vs_method2_cosine.json"))

    # Summary for safety-critical layers, dominant component
    print(f"\nSummary for safety-critical layers (6–14), component k=0:")
    for r in projection_results:
        if 6 <= r["layer"] <= 14 and r["component_k"] == 0:
            print(f"  Layer {r['layer']:2d}: SSS={r['SSS_mean_proj']:.4f}  "
                  f"SSU={r['SSU_mean_proj']:.4f}  p={r['p_value']:.2e}  "
                  f"var={r['variance_explained_frac']:.1%}")

    # Check non-dominant components for significant differences
    print(f"\nSignificant non-dominant components (k>0, p<0.01) at layers 6–14:")
    for r in projection_results:
        if 6 <= r["layer"] <= 14 and r["component_k"] > 0 and r["p_value"] < 0.01:
            print(f"  Layer {r['layer']:2d} k={r['component_k']}: "
                  f"SSS={r['SSS_mean_proj']:.4f}  SSU={r['SSU_mean_proj']:.4f}  "
                  f"p={r['p_value']:.2e}")


if __name__ == "__main__":
    main()
