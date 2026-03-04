#!/usr/bin/env python3
"""
Experiment D: Per-Category Safety Dimension Interaction
=======================================================
Decompose SSU modality shifts by HoliSafe-Bench harm category and test
whether different types of cross-modal harm project onto different
safety dimensions.

Category-agnostic profiles → a single correction method suffices.
Category-specific profiles → category-aware correction needed.

Dependencies:
  - activations/sample_{id}_vl.npz and sample_{id}_tt.npz
  - method2_activation_shift/sample_metadata.json
  - method2_activation_shift/reference_activation_matrices.npz

Outputs (under outputs/{model_name}/experiment_d_category_analysis/):
  - per_category_projections.json
  - category_subspace_overlap.json
  - anova_across_categories.json
  - category_sample_counts.json

Usage:
  cd ~/dev/VLM_Safety
  python diagnostic_exploratory_tests/followup_subspace_analysis/experiment_d_category_analysis.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import f_oneway

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction import (
    ActivationCache, load_modality_shift_matrix,
    extract_subspace, effective_rank, subspace_overlap,
    load_json, save_json,
)

K_SAFETY = 10
K_CAT = 5
MIN_CATEGORY_SIZE = 20


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


def main():
    OUTPUT_ROOT = _PROJECT_ROOT / "diagnostic_exploratory_tests" / "outputs" / "llava-1.5-7b-hf"
    cache = ActivationCache(str(OUTPUT_ROOT / "activations"))

    metadata = load_json(str(OUTPUT_ROOT / "method2_activation_shift" / "sample_metadata.json"))

    # Group SSU samples by category
    ssu_by_category = defaultdict(list)
    for s in metadata:
        if s["label"] == "SSU":
            ssu_by_category[s["category"]].append(s["id"])

    # Filter small categories
    ssu_by_category = {cat: ids for cat, ids in ssu_by_category.items()
                       if len(ids) >= MIN_CATEGORY_SIZE}

    category_counts = {cat: len(ids) for cat, ids in ssu_by_category.items()}
    print(f"Categories with >= {MIN_CATEGORY_SIZE} SSU samples:")
    for cat, n in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")

    # Load reference data
    ref_mats = np.load(OUTPUT_ROOT / "method2_activation_shift" / "reference_activation_matrices.npz")
    safety_vecs = np.load(OUTPUT_ROOT / "method2_activation_shift" / "safety_direction_vectors.npz")
    layers = sorted(int(k.replace("layer_", "")) for k in safety_vecs.files)

    out_dir = OUTPUT_ROOT / "experiment_d_category_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Steps 2 & 4 combined: per-category projections + ANOVA
    # Process one layer at a time to avoid redundant I/O
    category_projections = []
    anova_results = []
    focus_layers = [l for l in [7, 8, 10, 12, 14, 20] if f"safe_layer_{l}" in ref_mats]
    category_overlap_results = []

    for l in layers:
        safe_key = f"safe_layer_{l}"
        if safe_key not in ref_mats:
            continue

        print(f"  Layer {l:2d}...", end=" ", flush=True)

        V_k, sigmas_k = extract_safety_subspace(ref_mats, l, K_SAFETY)
        k = V_k.shape[0]

        # Load all category shift matrices once for this layer
        cat_shifts = {}
        for cat, cat_ids in ssu_by_category.items():
            cat_shifts[cat] = load_modality_shift_matrix(cache, cat_ids, l)

        # Per-category projections onto each safety dimension
        for cat, M_cat in cat_shifts.items():
            M_cat_f64 = M_cat.astype(np.float64)
            for ki in range(k):
                v_ki = V_k[ki].astype(np.float64)
                projections = M_cat_f64 @ v_ki

                category_projections.append({
                    "layer": l,
                    "category": cat,
                    "component_k": ki,
                    "n_samples": len(ssu_by_category[cat]),
                    "mean_projection": float(projections.mean()),
                    "std_projection": float(projections.std()),
                    "median_projection": float(np.median(projections)),
                })

        # ANOVA: do categories differ on each safety dimension?
        for ki in range(min(K_SAFETY, 5)):
            v_ki = V_k[ki].astype(np.float64)
            groups = []
            cat_names = []
            for cat in sorted(cat_shifts.keys()):
                groups.append(cat_shifts[cat].astype(np.float64) @ v_ki)
                cat_names.append(cat)

            if len(groups) >= 2:
                f_stat, p_val = f_oneway(*groups)
                anova_results.append({
                    "layer": l,
                    "component_k": ki,
                    "f_statistic": float(f_stat),
                    "p_value": float(p_val),
                    "categories_tested": cat_names,
                })

        # Category-specific integration subspace overlap (focus layers only)
        if l in focus_layers:
            V_safety_trunc = V_k[:K_CAT]
            for cat, M_cat in cat_shifts.items():
                if M_cat.shape[0] < K_CAT + 2:
                    continue

                V_cat = extract_subspace(M_cat, K_CAT, center=True)

                category_overlap_results.append({
                    "layer": l,
                    "category": cat,
                    "n_samples": len(ssu_by_category[cat]),
                    "overlap_with_safety": subspace_overlap(V_cat, V_safety_trunc),
                    "effective_rank_tau90": effective_rank(M_cat, 0.9),
                })

        print("done")

    save_json(category_projections, str(out_dir / "per_category_projections.json"))
    save_json(category_overlap_results, str(out_dir / "category_subspace_overlap.json"))
    save_json(anova_results, str(out_dir / "anova_across_categories.json"))
    save_json(category_counts, str(out_dir / "category_sample_counts.json"))

    # Summary
    print(f"\nANOVA summary for k=0 at safety-critical layers (6–14):")
    for r in anova_results:
        if r["component_k"] == 0 and 6 <= r["layer"] <= 14:
            sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else ""))
            print(f"  Layer {r['layer']:2d}: F={r['f_statistic']:.2f}  "
                  f"p={r['p_value']:.2e} {sig}")

    if category_overlap_results:
        print(f"\nCategory integration-safety overlap at focus layers:")
        for l in focus_layers:
            layer_overlaps = [r for r in category_overlap_results if r["layer"] == l]
            if layer_overlaps:
                overlaps_str = "  ".join(
                    f"{r['category'][:12]}={r['overlap_with_safety']:.3f}"
                    for r in sorted(layer_overlaps, key=lambda x: -x["overlap_with_safety"])
                )
                print(f"  Layer {l:2d}: {overlaps_str}")


if __name__ == "__main__":
    main()
