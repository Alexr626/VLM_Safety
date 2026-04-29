#!/usr/bin/env python3
"""
Compositional Safety Direction (multi-source, multi-representation)
====================================================================
Estimate a per-layer compositional safety direction `c^l` from one of:

  --source holisafe  --representation tt   (joint PCA, no pair structure)
  --source holisafe  --representation vl   (joint PCA, no pair structure)
  --source mssbench  --representation tt   (pairwise PCA on (rec_idx, q_idx) pairs)
  --source mssbench  --representation vl   (pairwise PCA — default for refusal eval)

Each `c^l` is also compared to the CatQA-derived semantic safety direction
`s^l` (cosine + top-5 subspace overlap). The semantic top-5 subspace is
built pairwise on CatQA whenever the legacy CatQA refs are detected
(matching the new pairwise s^l estimator), else falls back to joint PCA.

Output paths
------------
  experiment_artifacts/{model}/compositional_safety/{source}_{representation}/
    ├── compositional_safety_direction_vectors.npz                 # the chosen recipe
    └── compositional_safety_direction_vectors_joint.npz           # MSSBench only

  diagnostic_experiments/{model}/compositional_safety/outputs/
    ├── results/
    │   ├── direction_comparison_{source}_{representation}.json    # per-layer cos to s^l + eff_rank + subspace_overlap
    │   └── pairwise_vs_joint_{source}_{representation}.json       # MSSBench only: cos(c_pair, c_joint)
    └── artifacts/
        └── {source}_{representation}/                             # mirror of experiment_artifacts copy
            └── ...
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
    mssbench_pair_key, load_json, save_json, save_npz,
)
from src.dataset import (
    load_holisafe, filter_subsets, load_mssbench, DATASET_DATA_DIRS,
)
from src.model import _normalize_model_name

_EXPERIMENT_NAME = "compositional_safety"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--source", choices=["holisafe", "mssbench"], default="holisafe",
                   help="Source dataset: 'holisafe' uses all SSS+SSU; 'mssbench' "
                        "uses the train split of paired SSS/SSU image variants.")
    p.add_argument("--representation", choices=["tt", "vl"], default="tt",
                   help="Activation representation: 'tt' (text-only counterpart) "
                        "or 'vl' (full multimodal).")
    p.add_argument("--method", choices=["auto", "joint", "pairwise"], default="auto",
                   help="Force a specific PCA recipe. 'auto' selects 'joint' for "
                        "holisafe (no pair structure) and 'pairwise' for mssbench.")
    return p.parse_args()


def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _joint_pca_direction(H_pos: np.ndarray, H_neg: np.ndarray) -> np.ndarray:
    mu = (H_pos.mean(axis=0) + H_neg.mean(axis=0)) / 2
    M = np.concatenate([H_pos - mu, H_neg - mu], axis=0)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    v = Vt[0]
    if np.dot(v, H_pos.mean(axis=0) - H_neg.mean(axis=0)) < 0:
        v = -v
    return v


def _pairwise_pca_direction(D: np.ndarray, H_pos: np.ndarray,
                            H_neg: np.ndarray) -> np.ndarray:
    Dc = D - D.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Dc, full_matrices=False)
    v = Vt[0]
    if np.dot(v, H_pos.mean(axis=0) - H_neg.mean(axis=0)) < 0:
        v = -v
    return v


def _resolve_method(source: str, requested: str) -> str:
    """Map ('holisafe'|'mssbench', 'auto'|'joint'|'pairwise') -> concrete method."""
    if requested != "auto":
        return requested
    return "pairwise" if source == "mssbench" else "joint"


def _load_holisafe_pool(suffix: str):
    """All HoliSafe SSS + SSU samples; no train/eval split (per Decision 4)."""
    entries, images_base = load_holisafe()
    sss_samples, ssu_samples = filter_subsets(entries, images_base)
    sss_ids = [s["id"] for s in sss_samples]
    ssu_ids = [s["id"] for s in ssu_samples]
    print(f"  HoliSafe pool ({suffix}): {len(sss_ids)} SSS, {len(ssu_ids)} SSU")
    return sss_ids, ssu_ids


def _load_mssbench_train_pool():
    """MSSBench train-split SSS + SSU sample ids (pair structure preserved)."""
    split_path = _PROJECT_ROOT / "data" / "mssbench" / "train_eval_split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"MSSBench split not found: {split_path}\n"
            "Run: python -m src.dataset --mssbench_split"
        )
    split = load_json(str(split_path))
    train_ids = set(split["train_sample_ids"])
    samples = load_mssbench()
    sss = [s["id"] for s in samples if s["id"] in train_ids and s["label"] == "SSS"]
    ssu = [s["id"] for s in samples if s["id"] in train_ids and s["label"] == "SSU"]
    print(f"  MSSBench train pool: {len(sss)} SSS, {len(ssu)} SSU")
    return sss, ssu


def main():
    args = parse_args()
    method = _resolve_method(args.source, args.method)
    rep = args.representation
    source = args.source
    pair_dir = f"{source}_{rep}"

    model_name = _normalize_model_name(args.model)
    print(f"[{model_name}] source={source} representation={rep} method={method}")

    _DATA = _PROJECT_ROOT / "data"
    shared_artifacts = _PROJECT_ROOT / "experiment_artifacts" / model_name
    experiment_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME
    results_dir = experiment_dir / "outputs" / "results"
    local_artifacts = experiment_dir / "outputs" / "artifacts" / pair_dir
    shared_pair_dir = shared_artifacts / "compositional_safety" / pair_dir
    for d in [results_dir, local_artifacts, shared_pair_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load ids for the source pool ────────────────────────────────────────
    if source == "holisafe":
        sss_ids, ssu_ids = _load_holisafe_pool(rep)
    else:  # mssbench
        sss_ids, ssu_ids = _load_mssbench_train_pool()

    # ── Load semantic safety direction (CatQA-derived; pairwise-PCA s^l) ────
    sd_path = shared_artifacts / "vl_activation_shift" / "safety_direction_vectors.npz"
    if not sd_path.exists():
        raise FileNotFoundError(
            f"Semantic safety direction not found: {sd_path}\n"
            "Run vl_activation_shift.py first."
        )
    safety_dir = dict(np.load(sd_path))
    layers = sorted(int(k.replace("layer_", "")) for k in safety_dir)
    print(f"  Loaded semantic safety direction for {len(layers)} layers")

    # ── Load CatQA reference activations + ids (for the semantic top-5 subspace) ─
    ref_base = _DATA / "catqa-contrastive" / "activations" / model_name
    ref_safe_npz, ref_unsafe_npz = None, None
    ref_safe_ids, ref_unsafe_ids = [], []
    ref_safe_name, ref_unsafe_name = None, None
    if ref_base.exists():
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

    # Decide CatQA semantic-subspace recipe (parallels s^l recipe selection).
    semantic_recipe = "joint"
    semantic_safe_prefix = semantic_unsafe_prefix = None
    if ref_safe_ids and ref_unsafe_ids and ref_safe_name and ref_unsafe_name:
        semantic_safe_prefix = f"{ref_safe_name.replace('-', '_')}_"
        semantic_unsafe_prefix = f"{ref_unsafe_name.replace('-', '_')}_"
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
    print(f"  Semantic top-5 subspace recipe: {semantic_recipe.upper()}")

    # ── Per-layer compositional direction extraction ────────────────────────
    cache_dir = _DATA / DATASET_DATA_DIRS[source] / "activations" / model_name
    cache = ActivationCache(str(cache_dir))
    print(f"  Activation cache: {cache_dir}")

    # Probe pair structure once on layer 0 (MSSBench only).
    if method == "pairwise":
        H_sss0 = load_activation_matrix(cache, sss_ids, layers[0], suffix=rep).astype(np.float64)
        H_ssu0 = load_activation_matrix(cache, ssu_ids, layers[0], suffix=rep).astype(np.float64)
        D0, n_pairs0 = pairwise_difference_matrix(
            H_sss0, sss_ids, H_ssu0, ssu_ids, pair_key_fn=mssbench_pair_key,
        )
        if D0 is None:
            raise RuntimeError(
                f"--method=pairwise requested for source={source} but pair structure "
                "could not be detected. Check sample id format / pair coverage."
            )
        smaller = min(len(sss_ids), len(ssu_ids))
        if n_pairs0 < smaller:
            print(f"  WARN: pairwise alignment used {n_pairs0} pairs out of "
                  f"min({len(sss_ids)}, {len(ssu_ids)})={smaller}")
        print(f"  Recipe: PAIRWISE PCA (n_pairs={n_pairs0}) over {len(layers)} layers")
    else:
        n_pairs0 = 0
        print(f"  Recipe: JOINT PCA over {len(layers)} layers")

    comp_dir: dict = {}
    comp_dir_joint: dict = {}
    results = []
    for l in layers:
        try:
            H_sss = load_activation_matrix(cache, sss_ids, l, suffix=rep).astype(np.float64)
            H_ssu = load_activation_matrix(cache, ssu_ids, l, suffix=rep).astype(np.float64)
        except FileNotFoundError as e:
            print(f"  Skipping layer {l}: {e}")
            continue

        v_joint = _joint_pca_direction(H_sss, H_ssu)
        if method == "pairwise":
            D, _ = pairwise_difference_matrix(
                H_sss, sss_ids, H_ssu, ssu_ids, pair_key_fn=mssbench_pair_key,
            )
            v = _pairwise_pca_direction(D, H_sss, H_ssu)
        else:
            v = v_joint
        comp_dir[f"layer_{l}"] = v.astype(np.float32)
        if method == "pairwise":
            comp_dir_joint[f"layer_{l}"] = v_joint.astype(np.float32)

        s_l = safety_dir[f"layer_{l}"].astype(np.float64)
        cos_sim = _cosine_sim(v, s_l)
        # Effective rank of the SSS+SSU joint scatter (recipe-independent).
        mu = (H_sss.mean(axis=0) + H_ssu.mean(axis=0)) / 2
        combined = np.concatenate([H_sss - mu, H_ssu - mu], axis=0)
        eff_rank = effective_rank(combined, tau=0.9)

        row = {
            "layer": l,
            "cosine_sim_compositional_vs_semantic": cos_sim,
            "effective_rank": eff_rank,
            "method": method,
            "source": source,
            "representation": rep,
        }

        # Top-5 subspace overlap. The compositional subspace mirrors the chosen
        # recipe: pairwise -> diff matrix; joint -> joint scatter. The semantic
        # subspace mirrors the semantic_recipe established at the top of main().
        try:
            k = 5
            if method == "pairwise":
                V_comp = extract_subspace(D, k=k, center=True)
            else:
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
        except Exception as e:
            print(f"  layer {l}: subspace overlap skipped ({type(e).__name__}: {e})")

        results.append(row)

    # ── Save per-source artifacts ───────────────────────────────────────────
    chosen_path = shared_pair_dir / "compositional_safety_direction_vectors.npz"
    save_npz(comp_dir, str(chosen_path))
    save_npz(comp_dir, str(local_artifacts / "compositional_safety_direction_vectors.npz"))
    print(f"  → {chosen_path}")

    if method == "pairwise":
        joint_path = shared_pair_dir / "compositional_safety_direction_vectors_joint.npz"
        save_npz(comp_dir_joint, str(joint_path))
        save_npz(comp_dir_joint, str(local_artifacts / "compositional_safety_direction_vectors_joint.npz"))
        # Per-layer pairwise-vs-joint sanity for this source.
        sanity_rows = []
        for l in layers:
            k = f"layer_{l}"
            if k in comp_dir and k in comp_dir_joint:
                vp = comp_dir[k].astype(np.float64)
                vj = comp_dir_joint[k].astype(np.float64)
                sanity_rows.append({
                    "layer": l,
                    "cos_pair_vs_joint": _cosine_sim(vp, vj),
                    "n_pairs": int(n_pairs0),
                })
        save_json(sanity_rows,
                  str(results_dir / f"pairwise_vs_joint_{pair_dir}.json"))
        cos_vals = [r["cos_pair_vs_joint"] for r in sanity_rows
                    if not np.isnan(r["cos_pair_vs_joint"])]
        if cos_vals:
            print(f"  cos(c_pair, c_joint): mean={np.mean(cos_vals):.4f} "
                  f"min={np.min(cos_vals):.4f} max={np.max(cos_vals):.4f}")

    save_json(results, str(results_dir / f"direction_comparison_{pair_dir}.json"))

    # ── Summary ────────────────────────────────────────────────────────────
    cos_vals = [r["cosine_sim_compositional_vs_semantic"] for r in results
                if not np.isnan(r["cosine_sim_compositional_vs_semantic"])]
    if cos_vals:
        print(f"\nCosine similarity (c^l_{pair_dir} vs s^l):")
        print(f"  Mean: {np.mean(cos_vals):.4f}")
        print(f"  Min:  {np.min(cos_vals):.4f} | Max: {np.max(cos_vals):.4f}")

    ranks = [r["effective_rank"] for r in results]
    if ranks:
        print(f"Effective rank (SSU-vs-SSS, {pair_dir}): mean={np.mean(ranks):.1f}")


if __name__ == "__main__":
    main()
