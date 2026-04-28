#!/usr/bin/env python3
"""
Compositional Safety Direction
===============================
Extract a direction from SSU-vs-SSS contrastive data (CAST-style PCA on
TT activations), then compare it to the CatQA-derived semantic safety
direction.

Uses the full HoliSafe SSS+SSU pool — no train/eval split, since the
direction itself is not evaluated against held-out HoliSafe samples here.
The held-out evaluation lives in safety_probes.py.

Outputs
-------
  experiment_artifacts/{model}/compositional_safety/compositional_safety_direction_vectors.npz
  diagnostic_experiments/{model}/compositional_safety/outputs/results/direction_comparison.json
  diagnostic_experiments/{model}/compositional_safety/outputs/artifacts/compositional_safety_direction_vectors.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_DIAGNOSTIC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DIAGNOSTIC_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction import (
    ActivationCache, load_activation_matrix, effective_rank,
    extract_subspace, subspace_overlap, pairwise_difference_matrix,
    load_json, save_json, save_npz,
)
from src.dataset import load_holisafe, filter_subsets
from src.model import _normalize_model_name

_EXPERIMENT_NAME = "compositional_safety"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    return p.parse_args()


def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)

    _DATA = _PROJECT_ROOT / "data"
    shared_artifacts = _PROJECT_ROOT / "experiment_artifacts" / model_name
    experiment_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME
    results_dir = experiment_dir / "outputs" / "results"
    local_artifacts = experiment_dir / "outputs" / "artifacts"
    for d in [results_dir, local_artifacts,
              shared_artifacts / "compositional_safety"]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load ALL HoliSafe SSS + SSU samples (no train/eval split) ──────────
    entries, images_base = load_holisafe()
    sss_samples, ssu_samples = filter_subsets(entries, images_base)
    sss_ids = [s["id"] for s in sss_samples]
    ssu_ids = [s["id"] for s in ssu_samples]
    print(f"Loaded full HoliSafe pool: {len(sss_ids)} SSS, {len(ssu_ids)} SSU")

    # ── Load semantic safety direction (CatQA-derived) ─────────────────────
    sd_path = shared_artifacts / "vl_activation_shift" / "safety_direction_vectors.npz"
    if not sd_path.exists():
        raise FileNotFoundError(f"Semantic safety direction not found: {sd_path}")
    safety_dir = dict(np.load(sd_path))
    layers = sorted(int(k.replace("layer_", "")) for k in safety_dir)
    print(f"Loaded semantic safety direction for {len(layers)} layers")

    # ── Load CatQA reference activations (for subspace-overlap metric) ─────
    ref_base = _DATA / "catqa-contrastive" / "activations" / model_name
    ref_safe_npz, ref_unsafe_npz = None, None
    ref_safe_ids, ref_unsafe_ids = [], []
    ref_safe_name, ref_unsafe_name = None, None
    for subdir in ref_base.iterdir():
        if subdir.is_dir():
            meta_path = subdir / "metadata.json"
            if meta_path.exists():
                meta = load_json(str(meta_path))
                npz_path = subdir / "activation_matrices.npz"
                if npz_path.exists():
                    if meta["role"] == "safe":
                        ref_safe_npz = np.load(npz_path)
                        ref_safe_ids = meta.get("sample_ids", [])
                        ref_safe_name = meta.get("source", subdir.name)
                    elif meta["role"] == "unsafe":
                        ref_unsafe_npz = np.load(npz_path)
                        ref_unsafe_ids = meta.get("sample_ids", [])
                        ref_unsafe_name = meta.get("source", subdir.name)

    # Decide whether the semantic subspace will be built pairwise or joint.
    semantic_recipe = "joint"
    semantic_safe_prefix = semantic_unsafe_prefix = None
    if ref_safe_ids and ref_unsafe_ids and ref_safe_name and ref_unsafe_name:
        semantic_safe_prefix = f"{ref_safe_name.replace('-', '_')}_"
        semantic_unsafe_prefix = f"{ref_unsafe_name.replace('-', '_')}_"
        # Probe pair structure on layer 0 to set the recipe consistently.
        l0 = layers[0]
        H_safe0 = ref_safe_npz[f"safe_layer_{l0}"].astype(np.float64)
        H_unsafe0 = ref_unsafe_npz[f"unsafe_layer_{l0}"].astype(np.float64)
        D0, _ = pairwise_difference_matrix(
            H_safe0, ref_safe_ids, H_unsafe0, ref_unsafe_ids,
            safe_prefix=semantic_safe_prefix,
            unsafe_prefix=semantic_unsafe_prefix,
        )
        if D0 is not None:
            semantic_recipe = "pairwise"
    print(f"Semantic top-5 subspace recipe: {semantic_recipe.upper()}")

    # ── Compute compositional safety direction per layer ───────────────────
    cache = ActivationCache(str(_DATA / "holisafe-bench" / "activations" / model_name))
    comp_dir = {}
    results = []

    for l in layers:
        try:
            H_sss = load_activation_matrix(cache, sss_ids, l, suffix="tt")
            H_ssu = load_activation_matrix(cache, ssu_ids, l, suffix="tt")
        except FileNotFoundError as e:
            print(f"  Skipping layer {l}: {e}")
            continue

        H_sss = H_sss.astype(np.float64)
        H_ssu = H_ssu.astype(np.float64)

        mu = (H_sss.mean(axis=0) + H_ssu.mean(axis=0)) / 2
        M = np.concatenate([H_sss - mu, H_ssu - mu], axis=0)
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        vector = Vt[0]
        if np.dot(vector, H_sss.mean(axis=0) - H_ssu.mean(axis=0)) < 0:
            vector = -vector

        comp_dir[f"layer_{l}"] = vector.astype(np.float32)

        s_l = safety_dir[f"layer_{l}"].astype(np.float64)
        cos_sim = _cosine_sim(vector, s_l)
        combined = np.concatenate([H_sss - mu, H_ssu - mu], axis=0)
        eff_rank = effective_rank(combined, tau=0.9)

        row = {
            "layer": l,
            "cosine_sim_compositional_vs_semantic": cos_sim,
            "effective_rank": eff_rank,
        }

        k = 5
        try:
            # HoliSafe (compositional) subspace stays joint — c^l is intentionally
            # the joint-PCA estimate, since SSS/SSU are not paired.
            V_comp = extract_subspace(combined, k=k)
            if ref_safe_npz is not None and ref_unsafe_npz is not None:
                safe_key = f"safe_layer_{l}"
                unsafe_key = f"unsafe_layer_{l}"
                if safe_key in ref_safe_npz.files and unsafe_key in ref_unsafe_npz.files:
                    H_ref_safe = ref_safe_npz[safe_key].astype(np.float64)
                    H_ref_unsafe = ref_unsafe_npz[unsafe_key].astype(np.float64)
                    if semantic_recipe == "pairwise":
                        D_ref, _ = pairwise_difference_matrix(
                            H_ref_safe, ref_safe_ids,
                            H_ref_unsafe, ref_unsafe_ids,
                            safe_prefix=semantic_safe_prefix,
                            unsafe_prefix=semantic_unsafe_prefix,
                        )
                        V_semantic = extract_subspace(D_ref, k=k, center=True)
                    else:
                        mu_ref = (H_ref_safe.mean(axis=0) + H_ref_unsafe.mean(axis=0)) / 2
                        M_ref = np.concatenate([H_ref_safe - mu_ref, H_ref_unsafe - mu_ref], axis=0)
                        V_semantic = extract_subspace(M_ref, k=k)
                    row["subspace_overlap_top5"] = subspace_overlap(V_comp, V_semantic)
                    row["semantic_subspace_recipe"] = semantic_recipe
        except Exception:
            pass

        results.append(row)

    # ── Save ───────────────────────────────────────────────────────────────
    shared_path = (shared_artifacts / "compositional_safety"
                   / "compositional_safety_direction_vectors.npz")
    save_npz(comp_dir, str(shared_path))
    save_npz(comp_dir,
             str(local_artifacts / "compositional_safety_direction_vectors.npz"))
    save_json(results, str(results_dir / "direction_comparison.json"))

    # ── Summary ────────────────────────────────────────────────────────────
    cos_vals = [r["cosine_sim_compositional_vs_semantic"] for r in results
                if not np.isnan(r["cosine_sim_compositional_vs_semantic"])]
    if cos_vals:
        print(f"\nCosine similarity (compositional safety vs semantic safety):")
        print(f"  Mean: {np.mean(cos_vals):.4f}")
        print(f"  Min:  {np.min(cos_vals):.4f} | Max: {np.max(cos_vals):.4f}")

    ranks = [r["effective_rank"] for r in results]
    if ranks:
        print(f"Effective rank (SSU-vs-SSS): mean={np.mean(ranks):.1f}")


if __name__ == "__main__":
    main()
