#!/usr/bin/env python3
"""
Experiment C: Subspace Overlap — Integration Subspace vs. Safety Subspace
=========================================================================
Extract the principal directions of modality integration (the "integration
subspace") and measure its geometric overlap with the safety subspace.

High overlap → the model's fusion mechanism inherently interferes with
safety-relevant dimensions of activation space.

Also performs sensitivity analysis over subspace dimensionality k to
ensure robustness of overlap measurements.

Dependencies:
  - activations/sample_{id}_vl.npz and sample_{id}_tt.npz
  - method2_activation_shift/sample_metadata.json
  - method2_activation_shift/reference_activation_matrices.npz

Outputs (under outputs/{model_name}/experiment_c_subspace_overlap/):
  - subspace_overlap_results.json
  - sensitivity_analysis.json
  - integration_subspace_bases.npz
  - integration_spectra.npz

Usage:
  cd ~/dev/VLM_Safety
  python diagnostic_exploratory_tests/followup_subspace_analysis/experiment_c_subspace_overlap.py
"""

import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction import (
    ActivationCache, load_modality_shift_matrix,
    subspace_overlap, principal_angles,
    load_json, save_json, save_npz,
)

K_MODALITY = 10
K_MAX = 20  # for sensitivity analysis


def extract_safety_subspace(ref_mats, layer, k):
    """Extract safety subspace from reference activation matrices."""
    X_safe = ref_mats[f"safe_layer_{layer}"].astype(np.float64)
    X_unsafe = ref_mats[f"unsafe_layer_{layer}"].astype(np.float64)

    mu_unsafe = X_unsafe.mean(axis=0)
    D_safe = X_safe - mu_unsafe
    D_safe_centered = D_safe - D_safe.mean(axis=0, keepdims=True)

    _, sigmas, Vt = np.linalg.svd(D_safe_centered, full_matrices=False)
    k_actual = min(k, len(sigmas))
    return Vt[:k_actual], sigmas[:k_actual]


def extract_integration_subspace(M, k):
    """Extract integration subspace from a modality shift matrix."""
    centered = M - M.mean(axis=0, keepdims=True)
    _, sigmas, Vt = np.linalg.svd(centered, full_matrices=False)
    k_actual = min(k, len(sigmas))
    return Vt[:k_actual], sigmas[:k_actual]


def main():
    OUTPUT_ROOT = _PROJECT_ROOT / "diagnostic_exploratory_tests" / "outputs" / "llava-1.5-7b-hf"
    cache = ActivationCache(str(OUTPUT_ROOT / "activations"))

    metadata = load_json(str(OUTPUT_ROOT / "method2_activation_shift" / "sample_metadata.json"))
    sss_ids = [s["id"] for s in metadata if s["label"] == "SSS"]
    ssu_ids = [s["id"] for s in metadata if s["label"] == "SSU"]

    ref_mats = np.load(OUTPUT_ROOT / "method2_activation_shift" / "reference_activation_matrices.npz")

    safety_vecs = np.load(OUTPUT_ROOT / "method2_activation_shift" / "safety_direction_vectors.npz")
    layers = sorted(int(k.replace("layer_", "")) for k in safety_vecs.files)

    print(f"SSS: {len(sss_ids)}, SSU: {len(ssu_ids)}, Layers: {len(layers)}")

    out_dir = OUTPUT_ROOT / "experiment_c_subspace_overlap"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Precompute all subspaces at K_MAX for sensitivity analysis
    safety_subs_full = {}
    integ_subs_all_full = {}
    integ_subs_sss_full = {}
    integ_subs_ssu_full = {}

    integ_bases = {}
    integ_spectra_dict = {}
    overlap_results = []

    for l in layers:
        safe_key = f"safe_layer_{l}"
        if safe_key not in ref_mats:
            continue

        print(f"  Layer {l:2d}...", end=" ", flush=True)

        # Safety subspace (K_MAX components for sensitivity sweep)
        V_safety, _ = extract_safety_subspace(ref_mats, l, K_MAX)
        safety_subs_full[l] = V_safety

        # Integration subspaces
        M_sss = load_modality_shift_matrix(cache, sss_ids, l)
        M_ssu = load_modality_shift_matrix(cache, ssu_ids, l)
        M_all = np.concatenate([M_sss, M_ssu], axis=0)

        V_all, s_all = extract_integration_subspace(M_all, K_MAX)
        V_sss, s_sss = extract_integration_subspace(M_sss, K_MAX)
        V_ssu, s_ssu = extract_integration_subspace(M_ssu, K_MAX)

        integ_subs_all_full[l] = V_all
        integ_subs_sss_full[l] = V_sss
        integ_subs_ssu_full[l] = V_ssu

        # Save bases and spectra at default K_MODALITY
        k_m = min(K_MODALITY, len(s_all))
        integ_bases[f"all_layer_{l}"] = V_all[:k_m].astype(np.float32)
        integ_bases[f"sss_layer_{l}"] = V_sss[:k_m].astype(np.float32)
        integ_bases[f"ssu_layer_{l}"] = V_ssu[:k_m].astype(np.float32)
        integ_spectra_dict[f"all_layer_{l}"] = s_all[:k_m].astype(np.float32)
        integ_spectra_dict[f"sss_layer_{l}"] = s_sss[:k_m].astype(np.float32)
        integ_spectra_dict[f"ssu_layer_{l}"] = s_ssu[:k_m].astype(np.float32)

        # Compute overlaps at default K_MODALITY
        V_safety_k = V_safety[:min(K_MODALITY, V_safety.shape[0])]
        V_all_k = V_all[:min(K_MODALITY, V_all.shape[0])]
        V_sss_k = V_sss[:min(K_MODALITY, V_sss.shape[0])]
        V_ssu_k = V_ssu[:min(K_MODALITY, V_ssu.shape[0])]

        result = {
            "layer": l,
            "overlap_all_vs_safety": subspace_overlap(V_all_k, V_safety_k),
            "principal_angles_all_vs_safety": principal_angles(V_all_k, V_safety_k).tolist(),
            "overlap_sss_vs_safety": subspace_overlap(V_sss_k, V_safety_k),
            "overlap_ssu_vs_safety": subspace_overlap(V_ssu_k, V_safety_k),
            "overlap_sss_vs_ssu": subspace_overlap(V_sss_k, V_ssu_k),
        }
        overlap_results.append(result)

        print(f"overlap(all,safety)={result['overlap_all_vs_safety']:.4f}  "
              f"SSS={result['overlap_sss_vs_safety']:.4f}  "
              f"SSU={result['overlap_ssu_vs_safety']:.4f}  "
              f"SSS-vs-SSU={result['overlap_sss_vs_ssu']:.4f}")

    # Sensitivity analysis: sweep over k values at focus layers
    k_values = [1, 2, 3, 5, 10, 20]
    focus_layers = [l for l in [7, 8, 10, 12, 14, 20, 25] if l in safety_subs_full]
    sensitivity_results = []

    print(f"\nSensitivity analysis over k={k_values} at layers {focus_layers}...")
    for l in focus_layers:
        for k in k_values:
            V_safety_k = safety_subs_full[l][:min(k, safety_subs_full[l].shape[0])]
            V_all_k = integ_subs_all_full[l][:min(k, integ_subs_all_full[l].shape[0])]
            V_sss_k = integ_subs_sss_full[l][:min(k, integ_subs_sss_full[l].shape[0])]
            V_ssu_k = integ_subs_ssu_full[l][:min(k, integ_subs_ssu_full[l].shape[0])]

            if V_safety_k.shape[0] == 0 or V_all_k.shape[0] == 0:
                continue

            sensitivity_results.append({
                "layer": l, "k": k,
                "overlap_all_vs_safety": subspace_overlap(V_all_k, V_safety_k),
                "overlap_ssu_vs_safety": subspace_overlap(V_ssu_k, V_safety_k),
                "overlap_sss_vs_safety": subspace_overlap(V_sss_k, V_safety_k),
            })

        # Print sensitivity for this layer
        layer_sens = [r for r in sensitivity_results if r["layer"] == l]
        print(f"  Layer {l:2d}: " +
              "  ".join(f"k={r['k']}→{r['overlap_all_vs_safety']:.3f}" for r in layer_sens))

    save_json(overlap_results, str(out_dir / "subspace_overlap_results.json"))
    save_json(sensitivity_results, str(out_dir / "sensitivity_analysis.json"))
    save_npz(integ_bases, str(out_dir / "integration_subspace_bases.npz"))
    save_npz(integ_spectra_dict, str(out_dir / "integration_spectra.npz"))

    # Summary
    print(f"\nOverlap summary for safety-critical layers (6–14):")
    for r in overlap_results:
        if 6 <= r["layer"] <= 14:
            diff = r["overlap_ssu_vs_safety"] - r["overlap_sss_vs_safety"]
            print(f"  Layer {r['layer']:2d}: SSU-safety={r['overlap_ssu_vs_safety']:.4f}  "
                  f"SSS-safety={r['overlap_sss_vs_safety']:.4f}  "
                  f"diff(SSU-SSS)={diff:+.4f}  "
                  f"SSS-vs-SSU={r['overlap_sss_vs_ssu']:.4f}")


if __name__ == "__main__":
    main()
