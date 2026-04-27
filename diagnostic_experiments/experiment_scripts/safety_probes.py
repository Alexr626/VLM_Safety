#!/usr/bin/env python3
"""
Cross-Evaluation Safety Probes
===============================
Train & cross-evaluate linear probes for semantic safety (CatQA) and
compositional safety (SSU-vs-SSS) classification.

Outputs
-------
  diagnostic_experiments/{model}/compositional_safety/outputs/results/probe_results.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

_SCRIPT_DIR = Path(__file__).resolve().parent
_DIAGNOSTIC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DIAGNOSTIC_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction import ActivationCache, load_activation_matrix, load_json, save_json
from src.dataset import (
    load_holisafe,
    filter_subsets,
    split_holisafe_train_eval,
    split_catqa_train_eval,
)
from src.model import _normalize_model_name

_EXPERIMENT_NAME = "compositional_safety"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    return p.parse_args()


def _evaluate(clf, X, y):
    if len(X) == 0 or len(np.unique(y)) < 2:
        return {"accuracy": None, "f1": None, "auc_roc": None, "n": len(X)}
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    try:
        auc = float(roc_auc_score(y, y_prob))
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "auc_roc": auc, "n": len(X),
    }


def _read_refusal_key(label_entry, suffix):
    """Handle both legacy (refused_vl bool) and twoaxis (vl.harmful_content) schemas."""
    if not label_entry:
        return None
    # Legacy schema
    key = f"refused_{suffix}"
    if key in label_entry:
        return bool(label_entry[key])
    # Twoaxis schema: refusal = STRONG safety_awareness OR harmful_content=NO
    sub = label_entry.get(suffix)
    if isinstance(sub, dict):
        awareness = sub.get("safety_awareness")
        if awareness is not None:
            return awareness == "STRONG"
    return None


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)

    _DATA = _PROJECT_ROOT / "data"
    shared_artifacts = _PROJECT_ROOT / "experiment_artifacts" / model_name
    experiment_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME
    results_dir = experiment_dir / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load split ───────────────────────────────────────────────────────────
    split_path = _DATA / "holisafe-bench" / "train_eval_split.json"
    split: dict = {}
    if split_path.exists():
        split = load_json(str(split_path))
        sss_train_ids = split["sss_train_ids"]
        sss_eval_ids = split["sss_eval_ids"]
        ssu_train_ids = split["ssu_train_ids"]
        ssu_eval_ids = split["ssu_eval_ids"]
    else:
        print("No saved split found; creating one...")
        entries, images_base = load_holisafe()
        sss, ssu = filter_subsets(entries, images_base)
        sss_train, sss_eval, ssu_train, ssu_eval = split_holisafe_train_eval(sss, ssu)
        sss_train_ids = [s["id"] for s in sss_train]
        sss_eval_ids = [s["id"] for s in sss_eval]
        ssu_train_ids = [s["id"] for s in ssu_train]
        ssu_eval_ids = [s["id"] for s in ssu_eval]

    print(f"Split: SSS {len(sss_train_ids)} train / {len(sss_eval_ids)} eval, "
          f"SSU {len(ssu_train_ids)} train / {len(ssu_eval_ids)} eval")

    # ── Compositional eval pools (USU / SUU / UUU) ──────────────────────────
    # Populated by `python -m src.dataset` after the SSS/SSU split exists.
    new_eval_ids = {
        "usu": split.get("usu_eval_ids", []),
        "suu": split.get("suu_eval_ids", []),
        "uuu": split.get("uuu_eval_ids", []),
    }
    print(f"Compositional eval pools: "
          f"{ {k: len(v) for k, v in new_eval_ids.items()} }")

    # ── Load CatQA reference activations ─────────────────────────────────────
    ref_base = _DATA / "catqa-contrastive" / "activations" / model_name
    ref_safe_npz, ref_unsafe_npz = None, None
    safe_prefix, unsafe_prefix = "safe", "unsafe"
    safe_meta, unsafe_meta = None, None
    for subdir in ref_base.iterdir():
        if subdir.is_dir():
            meta_path = subdir / "metadata.json"
            if meta_path.exists():
                meta = load_json(str(meta_path))
                npz_path = subdir / "activation_matrices.npz"
                if npz_path.exists():
                    if meta["role"] == "safe":
                        ref_safe_npz = np.load(npz_path)
                        safe_prefix = meta["role"]
                        safe_meta = meta
                    elif meta["role"] == "unsafe":
                        ref_unsafe_npz = np.load(npz_path)
                        unsafe_prefix = meta["role"]
                        unsafe_meta = meta

    if ref_safe_npz is None or ref_unsafe_npz is None:
        raise FileNotFoundError("CatQA reference activations not found.")

    # ── Stratified CatQA train/eval split ───────────────────────────────────
    # Use the same n_samples/seed that extraction used so the split's row
    # indices align with the activation matrix rows.
    if (safe_meta["n_samples"] != unsafe_meta["n_samples"]
            or safe_meta["seed"] != unsafe_meta["seed"]):
        raise ValueError(
            "Mismatched n_samples/seed between safe and unsafe CatQA extractions; "
            "rows are not aligned."
        )
    catqa_train_idx, catqa_eval_idx = split_catqa_train_eval(
        n_samples=safe_meta["n_samples"], seed=safe_meta["seed"]
    )
    print(f"CatQA split: {len(catqa_train_idx)} train / {len(catqa_eval_idx)} eval")

    # ── Load behavioral labels (optional) ────────────────────────────────────
    refusal_path = (_DIAGNOSTIC_ROOT / model_name / "behavioral_ground_truth" /
                    "outputs" / "results" / "holisafe_refusal_labels.json")
    refusal_map = {}
    if refusal_path.exists():
        refusal_data = load_json(str(refusal_path))
        refusal_map = {r["id"]: r for r in refusal_data}
        print(f"Loaded behavioral labels for {len(refusal_map)} samples")

    layers = sorted(int(k.replace(f"{safe_prefix}_layer_", ""))
                    for k in ref_safe_npz.files if k.startswith(f"{safe_prefix}_layer_"))
    print(f"Processing {len(layers)} layers")

    cache = ActivationCache(str(_DATA / "holisafe-bench" / "activations" / model_name))
    all_results = []

    for l in layers:
        row = {"layer": l}

        catqa_safe = ref_safe_npz[f"{safe_prefix}_layer_{l}"].astype(np.float32)
        catqa_unsafe = ref_unsafe_npz[f"{unsafe_prefix}_layer_{l}"].astype(np.float32)

        catqa_safe_train = catqa_safe[catqa_train_idx]
        catqa_unsafe_train = catqa_unsafe[catqa_train_idx]
        catqa_safe_eval = catqa_safe[catqa_eval_idx]
        catqa_unsafe_eval = catqa_unsafe[catqa_eval_idx]

        X_a_train = np.vstack([catqa_safe_train, catqa_unsafe_train])
        y_a_train = np.array(
            [0] * len(catqa_safe_train) + [1] * len(catqa_unsafe_train)
        )
        X_catqa_eval = np.vstack([catqa_safe_eval, catqa_unsafe_eval])
        y_catqa_eval = np.array(
            [0] * len(catqa_safe_eval) + [1] * len(catqa_unsafe_eval)
        )

        try:
            X_sss_train = load_activation_matrix(cache, sss_train_ids, l, suffix="tt")
            X_ssu_train = load_activation_matrix(cache, ssu_train_ids, l, suffix="tt")
        except FileNotFoundError as e:
            print(f"  Skipping layer {l}: {e}")
            continue
        X_b_train = np.vstack([X_sss_train, X_ssu_train])
        y_b_train = np.array([0] * len(X_sss_train) + [1] * len(X_ssu_train))

        probe_a = LogisticRegression(max_iter=1000, solver="lbfgs").fit(X_a_train, y_a_train)
        probe_b = LogisticRegression(max_iter=1000, solver="lbfgs").fit(X_b_train, y_b_train)

        try:
            X_sss_eval_tt = load_activation_matrix(cache, sss_eval_ids, l, suffix="tt")
            X_ssu_eval_tt = load_activation_matrix(cache, ssu_eval_ids, l, suffix="tt")
            X_test1 = np.vstack([X_sss_eval_tt, X_ssu_eval_tt])
            y_test1 = np.array([0] * len(X_sss_eval_tt) + [1] * len(X_ssu_eval_tt))
        except FileNotFoundError:
            X_test1, y_test1 = np.empty((0, X_a_train.shape[1])), np.array([])

        try:
            X_sss_eval_vl = load_activation_matrix(cache, sss_eval_ids, l, suffix="vl")
            X_ssu_eval_vl = load_activation_matrix(cache, ssu_eval_ids, l, suffix="vl")
            X_test2 = np.vstack([X_sss_eval_vl, X_ssu_eval_vl])
            y_test2 = np.array([0] * len(X_sss_eval_vl) + [1] * len(X_ssu_eval_vl))
        except FileNotFoundError:
            X_test2, y_test2 = np.empty((0, X_a_train.shape[1])), np.array([])

        X_test3, y_test3 = X_catqa_eval, y_catqa_eval

        X_test4, y_test4 = [], []
        if refusal_map:
            for sid in ssu_eval_ids:
                if sid in refusal_map:
                    acts = cache.load_or_none(sid, "tt")
                    if acts is not None and l in acts:
                        refused = _read_refusal_key(refusal_map[sid], "vl")
                        if refused is None:
                            continue
                        X_test4.append(acts[l])
                        y_test4.append(0 if refused else 1)
            X_test4 = np.array(X_test4) if X_test4 else np.empty((0, X_a_train.shape[1]))
            y_test4 = np.array(y_test4) if y_test4 else np.array([])

        # Compositional eval pairs at this layer: SSS_eval vs {USU,SUU,UUU}_eval
        # at both TT and VL suffixes. Skip silently when activations are missing.
        new_tests: dict = {}
        for subset_key, ids in new_eval_ids.items():
            if not ids:
                continue
            for suffix in ("tt", "vl"):
                try:
                    X_sss = load_activation_matrix(
                        cache, sss_eval_ids, l, suffix=suffix)
                    X_neg = load_activation_matrix(
                        cache, ids, l, suffix=suffix)
                except FileNotFoundError as e:
                    print(f"  layer {l} {subset_key}/{suffix}: skip ({e})")
                    continue
                X = np.vstack([X_sss, X_neg])
                y = np.array([0] * len(X_sss) + [1] * len(X_neg))
                new_tests[f"holisafe_eval_sss_vs_{subset_key}_{suffix}"] = (X, y)

        test_iter = [
            ("holisafe_eval_tt", X_test1, y_test1),
            ("holisafe_eval_vl", X_test2, y_test2),
            ("catqa_eval", X_test3, y_test3),
            ("ssu_behavioral", X_test4, y_test4),
        ] + [(name, X, y) for name, (X, y) in new_tests.items()]

        for probe_name, probe in [
            ("semantic_safety_probe", probe_a),
            ("compositional_safety_probe", probe_b),
        ]:
            for test_name, X_test, y_test in test_iter:
                metrics = _evaluate(probe, X_test, y_test)
                for k, v in metrics.items():
                    row[f"{probe_name}__{test_name}__{k}"] = v

        all_results.append(row)

    save_json(all_results, str(results_dir / "probe_results.json"))

    if all_results:
        print(f"\nProbe Results Summary (best layer):")
        summary_tests = [
            "holisafe_eval_tt", "holisafe_eval_vl", "catqa_eval", "ssu_behavioral",
        ]
        for subset_key in ("usu", "suu", "uuu"):
            for suffix in ("tt", "vl"):
                summary_tests.append(f"holisafe_eval_sss_vs_{subset_key}_{suffix}")
        for probe in ["semantic_safety_probe", "compositional_safety_probe"]:
            for test in summary_tests:
                key = f"{probe}__{test}__accuracy"
                accs = [(r["layer"], r.get(key)) for r in all_results if r.get(key) is not None]
                if accs:
                    best = max(accs, key=lambda x: x[1])
                    print(f"  {probe} on {test}: best acc={best[1]:.3f} at layer {best[0]}")


if __name__ == "__main__":
    main()
