#!/usr/bin/env python3
"""
Combinatorial Safety Direction
===============================
Extract a direction from SSU-vs-SSS contrastive data (CAST-style PCA on
TT activations), then compare it to the CatQA-derived safety direction.

Outputs
-------
  experiment_artifacts/{model}/combinatorial_safety/combinatorial_direction_vectors.npz
  diagnostic_experiments/{model}/combinatorial_safety/outputs/results/direction_comparison.json
  diagnostic_experiments/{model}/combinatorial_safety/outputs/artifacts/combinatorial_direction_vectors.npz
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
    extract_subspace, subspace_overlap,
    load_json, save_json, save_npz,
)
from src.dataset import load_holisafe, filter_subsets, split_holisafe_train_eval
from src.model import _normalize_model_name

_EXPERIMENT_NAME = "combinatorial_safety"


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
              shared_artifacts / "combinatorial_safety"]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load train/eval split ────────────────────────────────────────────────
    split_path = _DATA / "holisafe-bench" / "train_eval_split.json"
    if split_path.exists():
        split = load_json(str(split_path))
        sss_train_ids = split["sss_train_ids"]
        ssu_train_ids = split["ssu_train_ids"]
        print(f"Loaded split: {len(sss_train_ids)} SSS train, {len(ssu_train_ids)} SSU train")
    else:
        print("No saved split found; creating one...")
        entries, images_base = load_holisafe()
        sss, ssu = filter_subsets(entries, images_base)
        sss_train, _, ssu_train, _ = split_holisafe_train_eval(sss, ssu)
        sss_train_ids = [s["id"] for s in sss_train]
        ssu_train_ids = [s["id"] for s in ssu_train]

    # ── Load CatQA safety direction ──────────────────────────────────────────
    sd_path = shared_artifacts / "vl_activation_shift" / "safety_direction_vectors.npz"
    if not sd_path.exists():
        raise FileNotFoundError(f"Safety direction not found: {sd_path}")
    safety_dir = dict(np.load(sd_path))
    layers = sorted(int(k.replace("layer_", "")) for k in safety_dir)
    print(f"Loaded CatQA safety direction for {len(layers)} layers")

    # ── Load CatQA reference activations ─────────────────────────────────────
    ref_base = _DATA / "catqa-contrastive" / "activations" / model_name
    ref_safe_npz, ref_unsafe_npz = None, None
    for subdir in ref_base.iterdir():
        if subdir.is_dir():
            meta_path = subdir / "metadata.json"
            if meta_path.exists():
                meta = load_json(str(meta_path))
                npz_path = subdir / "activation_matrices.npz"
                if npz_path.exists():
                    if meta["role"] == "safe":
                        ref_safe_npz = np.load(npz_path)
                    elif meta["role"] == "unsafe":
                        ref_unsafe_npz = np.load(npz_path)

    # ── Compute combinatorial direction per layer ────────────────────────────
    cache = ActivationCache(str(_DATA / "holisafe-bench" / "activations" / model_name))
    comb_dir = {}
    results = []

    for l in layers:
        try:
            H_sss = load_activation_matrix(cache, sss_train_ids, l, suffix="tt")
            H_ssu = load_activation_matrix(cache, ssu_train_ids, l, suffix="tt")
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

        comb_dir[f"layer_{l}"] = vector.astype(np.float32)

        s_l = safety_dir[f"layer_{l}"].astype(np.float64)
        cos_sim = _cosine_sim(vector, s_l)
        combined = np.concatenate([H_sss - mu, H_ssu - mu], axis=0)
        eff_rank = effective_rank(combined, tau=0.9)

        row = {"layer": l, "cosine_sim_comb_vs_catqa": cos_sim, "effective_rank": eff_rank}

        k = 5
        try:
            V_comb = extract_subspace(combined, k=k)
            if ref_safe_npz is not None and ref_unsafe_npz is not None:
                safe_key = f"safe_layer_{l}"
                unsafe_key = f"unsafe_layer_{l}"
                if safe_key in ref_safe_npz.files and unsafe_key in ref_unsafe_npz.files:
                    H_ref_safe = ref_safe_npz[safe_key].astype(np.float64)
                    H_ref_unsafe = ref_unsafe_npz[unsafe_key].astype(np.float64)
                    mu_ref = (H_ref_safe.mean(axis=0) + H_ref_unsafe.mean(axis=0)) / 2
                    M_ref = np.concatenate([H_ref_safe - mu_ref, H_ref_unsafe - mu_ref], axis=0)
                    V_catqa = extract_subspace(M_ref, k=k)
                    row["subspace_overlap_top5"] = subspace_overlap(V_comb, V_catqa)
        except Exception:
            pass

        results.append(row)

    # ── Save ─────────────────────────────────────────────────────────────────
    shared_path = shared_artifacts / "combinatorial_safety" / "combinatorial_direction_vectors.npz"
    save_npz(comb_dir, str(shared_path))
    save_npz(comb_dir, str(local_artifacts / "combinatorial_direction_vectors.npz"))
    save_json(results, str(results_dir / "direction_comparison.json"))

    # ── Summary ──────────────────────────────────────────────────────────────
    cos_vals = [r["cosine_sim_comb_vs_catqa"] for r in results
                if not np.isnan(r["cosine_sim_comb_vs_catqa"])]
    if cos_vals:
        print(f"\nCosine similarity (combinatorial vs CatQA):")
        print(f"  Mean: {np.mean(cos_vals):.4f}")
        print(f"  Min:  {np.min(cos_vals):.4f} | Max: {np.max(cos_vals):.4f}")

    ranks = [r["effective_rank"] for r in results]
    if ranks:
        print(f"Effective rank (SSU-vs-SSS): mean={np.mean(ranks):.1f}")


if __name__ == "__main__":
    main()
