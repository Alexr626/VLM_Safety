#!/usr/bin/env python3
"""
Method 1: Safety Boundary Probing (JailBound-style)
====================================================
From: "JailBound: Jailbreaking Internal Safety Boundaries of
       Vision-Language Models" (Song et al., NeurIPS 2025)

Diagnostic question
-------------------
Can a linear classifier separate SSS from SSU examples in the VLM's
fusion-layer hidden states?

  • If yes  → the model *encodes* integration-based harm but doesn't act on it.
  • If no   → the model fails to represent such harm internally at all.

Steps
-----
  1. Extract last-token hidden states at every transformer layer for all
     SSS and SSU samples from HoliSafe-Bench.
  2. Train a logistic regression classifier at each layer (80/20 split).
  3. Record: accuracy, weight vector, normal vector, per-sample boundary
     distances, predicted probabilities, confusion matrix.
  4. Cross-validation: train classifiers on *conventional* safe/unsafe pairs
     (UiUt=unsafe, SiSt→S=safe) and test transfer to SSU detection.

Outputs (under {output_dir}/{model_name}/method1_boundary_probing/)
-------
  per_layer_results.json     — accuracy + confusion matrix per layer
  classifiers.pkl            — dict {layer_idx: sklearn LogisticRegression}
  per_layer_weights.npz      — weight vectors and biases per layer
  per_layer_distances.npz    — boundary distances per sample per layer
  per_layer_probabilities.npz — predicted P(unsafe) per sample per layer
  sample_metadata.json       — sample list with IDs, labels, categories
  cross_trained_results.json — accuracy when trained on conventional data (optional)

Usage
-----
  python method1_boundary_probing.py
  python method1_boundary_probing.py --model llava-hf/llava-1.5-7b-hf \\
      --output_dir outputs --limit 50 --inspect
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))
from src.dataset import (
    load_holisafe, inspect_schema, filter_subsets,
    filter_reference_subsets, load_image_for_sample,
)
from src.model import VLMWrapper
from src.extraction import (
    ActivationCache, get_last_token_activations,
    cleanup_gpu, save_json, save_pickle, save_npz,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Method 1: Safety Boundary Probing")
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf",
                   help="HuggingFace model ID")
    p.add_argument("--output_dir", default="outputs",
                   help="Root output directory")
    p.add_argument("--cache_dir", default=None,
                   help="HuggingFace download cache directory")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of samples per class (for quick testing)")
    p.add_argument("--inspect", action="store_true",
                   help="Inspect dataset schema and exit")
    p.add_argument("--skip_extraction", action="store_true",
                   help="Skip extraction if all activations are already cached")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Fraction of samples held out for test (default 0.2)")
    return p.parse_args()


# ── Activation extraction ──────────────────────────────────────────────────────

def extract_and_cache(
    samples: list,
    wrapper: VLMWrapper,
    cache: ActivationCache,
    suffix: str = "vl",
    skip_if_cached: bool = True,
) -> None:
    """
    Run forward passes for each sample and save last-token activations.
    Skips samples whose cache file already exists.
    """
    todo = [s for s in samples if not (skip_if_cached and cache.exists(s["id"], suffix))]
    if not todo:
        print(f"  All {len(samples)} samples already cached (suffix='{suffix}').")
        return

    print(f"  Extracting activations for {len(todo)} samples (suffix='{suffix}')...")
    for sample in tqdm(todo, desc="Forward passes"):
        image = load_image_for_sample(sample)
        if image is None:
            print(f"  Warning: skipping sample {sample['id']} — image unavailable.")
            continue

        try:
            hidden, _, _ = wrapper.forward_vl(image, sample["text"])
            acts = get_last_token_activations(hidden)
            cache.save(sample["id"], acts, suffix=suffix)
            del hidden, acts
        except Exception as e:
            print(f"  Warning: sample {sample['id']} failed — {e}")

        cleanup_gpu()


# ── Classifier training ───────────────────────────────────────────────────────

def train_probing_classifiers(
    samples: list,
    cache: ActivationCache,
    num_layers: int,
    test_size: float,
    suffix: str = "vl",
):
    """
    For each layer, collect activations, train logistic regression, record stats.

    Returns
    -------
    per_layer_results : list of dicts (one per layer)
    classifiers       : dict {layer_idx: fitted LogisticRegression}
    weights_dict      : dict {str(layer_idx): np.ndarray}  (for npz)
    biases_dict       : dict {str(layer_idx): np.ndarray}
    distances_dict    : dict {str(layer_idx): np.ndarray}  — per sample, signed
    probs_dict        : dict {str(layer_idx): np.ndarray}  — per sample P(unsafe)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    # ── Pre-load activations for available samples ────────────────────────
    print("  Loading cached activations into memory...")
    available = []
    for s in samples:
        acts = cache.load_or_none(s["id"], suffix=suffix)
        if acts is not None:
            available.append((s, acts))

    if not available:
        raise RuntimeError("No cached activations found. Run extraction first.")

    print(f"  Available: {len(available)} samples "
          f"({sum(1 for s,_ in available if s['label']=='SSS')} SSS, "
          f"{sum(1 for s,_ in available if s['label']=='SSU')} SSU)")

    labels_all = np.array([s["label_idx"] for s, _ in available])
    ids_all = [s["id"] for s, _ in available]

    # Layer indices present in first sample's activation dict
    layer_indices = sorted(available[0][1].keys())

    per_layer_results = []
    classifiers = {}
    weights_dict = {}
    biases_dict = {}
    distances_dict = {}
    probs_dict = {}

    print(f"  Training classifiers across {len(layer_indices)} layers...")
    for l in tqdm(layer_indices, desc="Layers"):
        # Collect activations at this layer
        h_all = np.stack([acts[l] for _, acts in available], axis=0)  # (N, d)
        y_all = labels_all

        if len(np.unique(y_all)) < 2:
            print(f"  Layer {l}: only one class present, skipping.")
            continue

        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            h_all, y_all, list(range(len(y_all))),
            test_size=test_size, random_state=42, stratify=y_all,
        )

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        acc = float(clf.score(X_te, y_te))
        cm = confusion_matrix(y_te, y_pred).tolist()

        # Weight vector, bias, normal vector
        w = clf.coef_[0]           # shape (d,)
        b = float(clf.intercept_[0])
        w_norm = float(np.linalg.norm(w))
        v = w / (w_norm + 1e-12)   # unit normal to boundary

        # Per-sample distances and probabilities (on ALL samples)
        raw_scores = h_all @ w + b
        distances = raw_scores / (w_norm + 1e-12)   # signed distance to boundary
        probs = 1.0 / (1.0 + np.exp(-raw_scores))   # sigmoid = P(unsafe=1)

        per_layer_results.append({
            "layer": l,
            "test_accuracy": acc,
            "confusion_matrix": cm,
            "w_norm": w_norm,
            "bias": b,
            "train_size": len(X_tr),
            "test_size": len(X_te),
        })
        classifiers[l] = clf
        weights_dict[f"w_layer_{l}"] = w.astype(np.float32)
        biases_dict[f"b_layer_{l}"] = np.array([b], dtype=np.float32)
        weights_dict[f"v_layer_{l}"] = v.astype(np.float32)   # unit normal
        distances_dict[f"layer_{l}"] = distances.astype(np.float32)
        probs_dict[f"layer_{l}"] = probs.astype(np.float32)

    return per_layer_results, classifiers, weights_dict, biases_dict, distances_dict, probs_dict


# ── Cross-trained classifiers (conventional → SSS/SSU) ───────────────────────

def train_cross_classifiers(
    ref_safe_samples: list,
    ref_unsafe_samples: list,
    target_ssu_samples: list,
    target_sss_samples: list,
    cache: ActivationCache,
    num_layers: int,
    suffix_ref: str = "ref",
    suffix_target: str = "vl",
):
    """
    Train on conventional (clearly safe vs clearly unsafe) multimodal data,
    then test on SSU vs SSS.

    The reference safe samples come from SiSt→S subset (used as "safe").
    The reference unsafe samples come from UiUt subset (used as "unsafe").

    NOTE: The activations for reference samples must already be cached under
    suffix_ref. Call extract_and_cache() on them first.

    Returns list of per-layer result dicts.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix

    def _load_acts(samples, suf):
        out = []
        for s in samples:
            a = cache.load_or_none(s["id"], suffix=suf)
            if a is not None:
                out.append((s, a))
        return out

    ref_safe_loaded  = _load_acts(ref_safe_samples,   suffix_ref)
    ref_unsafe_loaded = _load_acts(ref_unsafe_samples, suffix_ref)
    tgt_sss_loaded   = _load_acts(target_sss_samples,  suffix_target)
    tgt_ssu_loaded   = _load_acts(target_ssu_samples,  suffix_target)

    if not ref_safe_loaded or not ref_unsafe_loaded:
        print("  Cross-training skipped: reference activations not available.")
        return []
    if not tgt_sss_loaded or not tgt_ssu_loaded:
        print("  Cross-training skipped: target activations not available.")
        return []

    layer_indices = sorted(ref_safe_loaded[0][1].keys())
    results = []

    print(f"  Cross-training on {len(ref_safe_loaded)} safe + {len(ref_unsafe_loaded)} unsafe ref samples")
    print(f"  Testing on {len(tgt_sss_loaded)} SSS + {len(tgt_ssu_loaded)} SSU samples")

    for l in tqdm(layer_indices, desc="Cross-train layers"):
        # Training data: conventional safe (y=0) and unsafe (y=1)
        X_ref_safe   = np.stack([a[l] for _, a in ref_safe_loaded])
        X_ref_unsafe = np.stack([a[l] for _, a in ref_unsafe_loaded])
        X_tr = np.vstack([X_ref_safe, X_ref_unsafe])
        y_tr = np.concatenate([
            np.zeros(len(X_ref_safe)),
            np.ones(len(X_ref_unsafe)),
        ])

        # Test data: SSS (y=0) and SSU (y=1)
        X_tgt_sss = np.stack([a[l] for _, a in tgt_sss_loaded])
        X_tgt_ssu = np.stack([a[l] for _, a in tgt_ssu_loaded])
        X_te = np.vstack([X_tgt_sss, X_tgt_ssu])
        y_te = np.concatenate([
            np.zeros(len(X_tgt_sss)),
            np.ones(len(X_tgt_ssu)),
        ])

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        acc = float(accuracy_score(y_te, y_pred))
        cm = confusion_matrix(y_te, y_pred).tolist()

        results.append({
            "layer": l,
            "test_accuracy": acc,
            "confusion_matrix": cm,
            "train_safe": len(ref_safe_loaded),
            "train_unsafe": len(ref_unsafe_loaded),
            "test_sss": len(tgt_sss_loaded),
            "test_ssu": len(tgt_ssu_loaded),
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Output directories ────────────────────────────────────────────────
    model_name = args.model.split("/")[-1]
    out_base   = Path(args.output_dir) / model_name
    out_m1     = out_base / "method1_boundary_probing"
    act_dir    = out_base / "activations"
    out_m1.mkdir(parents=True, exist_ok=True)

    cache = ActivationCache(str(act_dir))

    # ── Load dataset ──────────────────────────────────────────────────────
    print("\n[1/5] Loading HoliSafe-Bench...")
    entries, images_base = load_holisafe(cache_dir=args.cache_dir)

    if args.inspect:
        inspect_schema(entries)
        sys.exit(0)

    sss_samples, ssu_samples = filter_subsets(entries, images_base)

    # Apply sample limit for quick testing
    if args.limit:
        sss_samples = sss_samples[: args.limit]
        ssu_samples = ssu_samples[: args.limit]
        print(f"  (limit applied: {args.limit} per class)")

    all_samples = sss_samples + ssu_samples

    # Reference subsets for cross-trained classifiers
    ref_buckets = filter_reference_subsets(entries, images_base)

    # Choose conventional safe/unsafe references:
    #   safe   → SiSt→S samples (same as SSS, but we'll use them as training)
    #   unsafe → UiUt samples (unsafe image + unsafe text → clearly unsafe)
    ref_safe_candidates = []
    ref_unsafe_candidates = []
    for k, v in ref_buckets.items():
        k_lower = k.lower().replace("→", "->").replace("_", "->")
        if "uiut" in k_lower or ("ui" in k_lower and "ut" in k_lower):
            ref_unsafe_candidates.extend(v)
        elif "sist" in k_lower and ("->s" in k_lower or "safe" in k_lower.split("->")[-1:]):
            ref_safe_candidates.extend(v)

    if args.limit:
        ref_safe_candidates   = ref_safe_candidates[:args.limit]
        ref_unsafe_candidates = ref_unsafe_candidates[:args.limit]

    # Save sample metadata
    metadata = [
        {
            "id": s["id"],
            "label": s["label"],
            "category": s["category"],
            "text_snippet": s["text"][:120],
        }
        for s in all_samples
    ]
    save_json(metadata, str(out_m1 / "sample_metadata.json"))

    # ── Load model ────────────────────────────────────────────────────────
    print("\n[2/5] Loading model...")
    wrapper = VLMWrapper(args.model).load()

    # ── Extract activations (SSS + SSU) ───────────────────────────────────
    print("\n[3/5] Extracting VL activations for SSS and SSU samples...")
    extract_and_cache(all_samples, wrapper, cache, suffix="vl",
                      skip_if_cached=args.skip_extraction)

    # Extract reference activations for cross-training (if available)
    do_cross = bool(ref_safe_candidates and ref_unsafe_candidates)
    if do_cross:
        print("\n[3b] Extracting activations for reference (conventional safe/unsafe)...")
        extract_and_cache(ref_safe_candidates,   wrapper, cache, suffix="ref")
        extract_and_cache(ref_unsafe_candidates, wrapper, cache, suffix="ref")

    cleanup_gpu()
    del wrapper   # free GPU memory before classification

    # ── Train classifiers ─────────────────────────────────────────────────
    print("\n[4/5] Training logistic regression classifiers per layer...")
    (per_layer_results, classifiers,
     weights_dict, biases_dict,
     distances_dict, probs_dict) = train_probing_classifiers(
        all_samples, cache, num_layers=32,
        test_size=args.test_size,
    )

    # ── Cross-trained classifiers ─────────────────────────────────────────
    cross_results = []
    if do_cross:
        print("\n[4b] Cross-trained classifiers (conventional → SSS/SSU)...")
        cross_results = train_cross_classifiers(
            ref_safe_samples=ref_safe_candidates,
            ref_unsafe_samples=ref_unsafe_candidates,
            target_ssu_samples=ssu_samples,
            target_sss_samples=sss_samples,
            cache=cache,
            num_layers=32,
            suffix_ref="ref",
            suffix_target="vl",
        )

    # ── Save results ──────────────────────────────────────────────────────
    print("\n[5/5] Saving results...")

    save_json(per_layer_results, str(out_m1 / "per_layer_results.json"))

    # Combine weights and biases into a single npz
    combined_weights = {**weights_dict, **biases_dict}
    save_npz(combined_weights, str(out_m1 / "per_layer_weights.npz"))
    save_npz(distances_dict,   str(out_m1 / "per_layer_distances.npz"))
    save_npz(probs_dict,       str(out_m1 / "per_layer_probabilities.npz"))
    save_pickle(classifiers,   str(out_m1 / "classifiers.pkl"))

    if cross_results:
        save_json(cross_results, str(out_m1 / "cross_trained_results.json"))

    # ── Summary ───────────────────────────────────────────────────────────
    if per_layer_results:
        best = max(per_layer_results, key=lambda x: x["test_accuracy"])
        print(f"\n{'='*60}")
        print(f"  Method 1 complete.")
        print(f"  Best layer: {best['layer']}  accuracy: {best['test_accuracy']:.3f}")
        print(f"  Results  → {out_m1}/")
        print(f"{'='*60}")
    else:
        print("\nWarning: no per-layer results were produced.")


if __name__ == "__main__":
    main()
