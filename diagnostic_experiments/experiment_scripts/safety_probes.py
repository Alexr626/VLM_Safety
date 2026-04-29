#!/usr/bin/env python3
"""
Cross-Evaluation Safety Probes (multi-source × multi-representation)
=====================================================================
Train and cross-evaluate logistic-regression safety probes:

  semantic_safety_probe                    — CatQA train (TT-only)
  compositional_safety_probe_holisafe_tt   — HoliSafe SSS_train+SSU_train (TT)
  compositional_safety_probe_holisafe_vl   — HoliSafe SSS_train+SSU_train (VL)
  compositional_safety_probe_mssbench_tt   — MSSBench train pairs (TT)
  compositional_safety_probe_mssbench_vl   — MSSBench train pairs (VL)

Each is evaluated on every test set that produces a well-defined binary
classification:
  - holisafe_eval_{tt,vl}                  (SSS_eval vs SSU_eval)
  - holisafe_eval_sss_vs_{usu,suu,uuu}_{tt,vl}  (compositional pairs)
  - catqa_eval                             (text-only)
  - ssu_behavioral                         (HoliSafe SSU TT activations,
                                            label = refusal)
  - mssbench_eval_{tt,vl}                  (MSSBench held-out SSS vs SSU)

Output
------
  diagnostic_experiments/{model}/compositional_safety/outputs/results/probe_results.json
  Wide-format keys: {probe_name}__{test_name}__{accuracy|f1|auc_roc|n}
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
    DATASET_DATA_DIRS,
)
from src.model import _normalize_model_name

_EXPERIMENT_NAME = "compositional_safety"


# ── Probe registry: probe_id -> (display name, source, representation) ─────
PROBE_DEFS = [
    ("semantic_safety_probe",                   "catqa",    "tt"),
    ("compositional_safety_probe_holisafe_tt",  "holisafe", "tt"),
    ("compositional_safety_probe_holisafe_vl",  "holisafe", "vl"),
    ("compositional_safety_probe_mssbench_tt",  "mssbench", "tt"),
    ("compositional_safety_probe_mssbench_vl",  "mssbench", "vl"),
]


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
    if not label_entry:
        return None
    key = f"refused_{suffix}"
    if key in label_entry:
        return bool(label_entry[key])
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
    experiment_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME
    results_dir = experiment_dir / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── HoliSafe split ──────────────────────────────────────────────────────
    split_path = _DATA / "holisafe-bench" / "train_eval_split.json"
    split: dict = {}
    if split_path.exists():
        split = load_json(str(split_path))
        sss_train_ids = split["sss_train_ids"]
        sss_eval_ids = split["sss_eval_ids"]
        ssu_train_ids = split["ssu_train_ids"]
        ssu_eval_ids = split["ssu_eval_ids"]
    else:
        print("No saved HoliSafe split; creating one...")
        entries, images_base = load_holisafe()
        sss, ssu = filter_subsets(entries, images_base)
        sss_train, sss_eval, ssu_train, ssu_eval = split_holisafe_train_eval(sss, ssu)
        sss_train_ids = [s["id"] for s in sss_train]
        sss_eval_ids = [s["id"] for s in sss_eval]
        ssu_train_ids = [s["id"] for s in ssu_train]
        ssu_eval_ids = [s["id"] for s in ssu_eval]
    print(f"HoliSafe split: SSS {len(sss_train_ids)}/{len(sss_eval_ids)} "
          f"SSU {len(ssu_train_ids)}/{len(ssu_eval_ids)}")

    new_eval_ids = {
        "usu": split.get("usu_eval_ids", []),
        "suu": split.get("suu_eval_ids", []),
        "uuu": split.get("uuu_eval_ids", []),
    }
    print(f"  Compositional eval pools: "
          f"{ {k: len(v) for k, v in new_eval_ids.items()} }")

    # ── MSSBench split (optional; probes/tests fall back gracefully) ────────
    mssbench_split_path = _DATA / "mssbench" / "train_eval_split.json"
    mssbench_train_ids: list[str] = []
    mssbench_eval_ids: list[str] = []
    if mssbench_split_path.exists():
        mss = load_json(str(mssbench_split_path))
        mssbench_train_ids = mss.get("train_sample_ids", [])
        mssbench_eval_ids = mss.get("eval_sample_ids", [])
        print(f"MSSBench split: {len(mssbench_train_ids)} train / "
              f"{len(mssbench_eval_ids)} eval")
    else:
        print("MSSBench split not found; MSSBench probes/tests will be skipped.")

    def _mssbench_split(ids: list[str]):
        sss = [i for i in ids if "_SSS_" in i]
        ssu = [i for i in ids if "_SSU_" in i]
        return sss, ssu
    mss_sss_train, mss_ssu_train = _mssbench_split(mssbench_train_ids)
    mss_sss_eval,  mss_ssu_eval  = _mssbench_split(mssbench_eval_ids)

    # ── CatQA reference activations + train/eval indices ────────────────────
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

    if (safe_meta["n_samples"] != unsafe_meta["n_samples"]
            or safe_meta["seed"] != unsafe_meta["seed"]):
        raise ValueError(
            "Mismatched n_samples/seed between safe and unsafe CatQA extractions."
        )
    catqa_train_idx, catqa_eval_idx = split_catqa_train_eval(
        n_samples=safe_meta["n_samples"], seed=safe_meta["seed"]
    )
    print(f"CatQA split: {len(catqa_train_idx)} train / {len(catqa_eval_idx)} eval")

    # ── Behavioral labels (optional) ────────────────────────────────────────
    refusal_path = (_DIAGNOSTIC_ROOT / model_name / "behavioral_ground_truth" /
                    "outputs" / "results" / "holisafe_refusal_labels.json")
    refusal_map: dict = {}
    if refusal_path.exists():
        refusal_data = load_json(str(refusal_path))
        refusal_map = {r["id"]: r for r in refusal_data}
        print(f"Loaded behavioral labels for {len(refusal_map)} samples")

    # ── Caches per dataset ──────────────────────────────────────────────────
    holisafe_cache = ActivationCache(str(
        _DATA / DATASET_DATA_DIRS["holisafe"] / "activations" / model_name))
    mssbench_cache_dir = _DATA / DATASET_DATA_DIRS["mssbench"] / "activations" / model_name
    mssbench_cache = ActivationCache(str(mssbench_cache_dir)) \
        if mssbench_cache_dir.exists() else None

    # ── Layer set (driven by CatQA reference activations) ──────────────────
    layers = sorted(int(k.replace(f"{safe_prefix}_layer_", ""))
                    for k in ref_safe_npz.files
                    if k.startswith(f"{safe_prefix}_layer_"))
    print(f"Processing {len(layers)} layers")

    # ── Helpers to load (X_train, y_train) per probe ───────────────────────
    def _train_set(probe_id: str, layer: int):
        """Return (X_train, y_train) or None if data is missing."""
        if probe_id == "semantic_safety_probe":
            safe = ref_safe_npz[f"{safe_prefix}_layer_{layer}"].astype(np.float32)
            unsafe = ref_unsafe_npz[f"{unsafe_prefix}_layer_{layer}"].astype(np.float32)
            X = np.vstack([safe[catqa_train_idx], unsafe[catqa_train_idx]])
            y = np.array([0] * len(catqa_train_idx) + [1] * len(catqa_train_idx))
            return X, y
        if probe_id.startswith("compositional_safety_probe_holisafe_"):
            suffix = probe_id.rsplit("_", 1)[-1]   # tt | vl
            try:
                X_sss = load_activation_matrix(holisafe_cache, sss_train_ids, layer, suffix=suffix)
                X_ssu = load_activation_matrix(holisafe_cache, ssu_train_ids, layer, suffix=suffix)
            except FileNotFoundError:
                return None
            X = np.vstack([X_sss, X_ssu])
            y = np.array([0] * len(X_sss) + [1] * len(X_ssu))
            return X, y
        if probe_id.startswith("compositional_safety_probe_mssbench_"):
            if mssbench_cache is None or not mss_sss_train or not mss_ssu_train:
                return None
            suffix = probe_id.rsplit("_", 1)[-1]
            try:
                X_sss = load_activation_matrix(mssbench_cache, mss_sss_train, layer, suffix=suffix)
                X_ssu = load_activation_matrix(mssbench_cache, mss_ssu_train, layer, suffix=suffix)
            except FileNotFoundError:
                return None
            X = np.vstack([X_sss, X_ssu])
            y = np.array([0] * len(X_sss) + [1] * len(X_ssu))
            return X, y
        raise ValueError(f"Unknown probe_id: {probe_id}")

    def _build_test_sets(layer: int, hidden_dim: int):
        """Build all available test sets at this layer. Each entry:
        (test_name, X, y) — empty arrays when unavailable.
        """
        empty = (np.empty((0, hidden_dim)), np.array([]))
        tests: list[tuple] = []

        # HoliSafe SSS_eval vs SSU_eval (TT/VL)
        for suffix in ("tt", "vl"):
            try:
                X_sss = load_activation_matrix(holisafe_cache, sss_eval_ids, layer, suffix=suffix)
                X_ssu = load_activation_matrix(holisafe_cache, ssu_eval_ids, layer, suffix=suffix)
                X = np.vstack([X_sss, X_ssu])
                y = np.array([0] * len(X_sss) + [1] * len(X_ssu))
                tests.append((f"holisafe_eval_{suffix}", X, y))
            except FileNotFoundError:
                tests.append((f"holisafe_eval_{suffix}", *empty))

        # CatQA eval (TT only)
        catqa_safe = ref_safe_npz[f"{safe_prefix}_layer_{layer}"].astype(np.float32)
        catqa_unsafe = ref_unsafe_npz[f"{unsafe_prefix}_layer_{layer}"].astype(np.float32)
        X_catqa_eval = np.vstack([catqa_safe[catqa_eval_idx], catqa_unsafe[catqa_eval_idx]])
        y_catqa_eval = np.array(
            [0] * len(catqa_eval_idx) + [1] * len(catqa_eval_idx)
        )
        tests.append(("catqa_eval", X_catqa_eval, y_catqa_eval))

        # SSU behavioral (HoliSafe TT activations, label = refusal)
        if refusal_map:
            X_b, y_b = [], []
            for sid in ssu_eval_ids:
                if sid not in refusal_map:
                    continue
                acts = holisafe_cache.load_or_none(sid, "tt")
                if acts is None or layer not in acts:
                    continue
                refused = _read_refusal_key(refusal_map[sid], "vl")
                if refused is None:
                    continue
                X_b.append(acts[layer])
                y_b.append(0 if refused else 1)
            if X_b:
                tests.append(("ssu_behavioral", np.array(X_b), np.array(y_b)))
            else:
                tests.append(("ssu_behavioral", *empty))
        else:
            tests.append(("ssu_behavioral", *empty))

        # Compositional pairs (USU/SUU/UUU vs SSS_eval) at TT and VL
        for subset_key, ids in new_eval_ids.items():
            if not ids:
                continue
            for suffix in ("tt", "vl"):
                try:
                    X_sss = load_activation_matrix(holisafe_cache, sss_eval_ids, layer, suffix=suffix)
                    X_neg = load_activation_matrix(holisafe_cache, ids, layer, suffix=suffix)
                except FileNotFoundError:
                    continue
                X = np.vstack([X_sss, X_neg])
                y = np.array([0] * len(X_sss) + [1] * len(X_neg))
                tests.append((f"holisafe_eval_sss_vs_{subset_key}_{suffix}", X, y))

        # MSSBench eval (TT/VL)
        if mssbench_cache is not None and mss_sss_eval and mss_ssu_eval:
            for suffix in ("tt", "vl"):
                try:
                    X_sss = load_activation_matrix(mssbench_cache, mss_sss_eval, layer, suffix=suffix)
                    X_ssu = load_activation_matrix(mssbench_cache, mss_ssu_eval, layer, suffix=suffix)
                    X = np.vstack([X_sss, X_ssu])
                    y = np.array([0] * len(X_sss) + [1] * len(X_ssu))
                    tests.append((f"mssbench_eval_{suffix}", X, y))
                except FileNotFoundError:
                    tests.append((f"mssbench_eval_{suffix}", *empty))
        return tests

    # ── Layer loop ─────────────────────────────────────────────────────────
    all_results = []
    for layer in layers:
        row = {"layer": layer}

        # Hidden dim from CatQA matrix (always available).
        hidden_dim = ref_safe_npz[f"{safe_prefix}_layer_{layer}"].shape[1]

        # Train every probe.
        probes: dict[str, LogisticRegression] = {}
        for probe_id, _, _ in PROBE_DEFS:
            train = _train_set(probe_id, layer)
            if train is None:
                continue
            X_train, y_train = train
            if len(np.unique(y_train)) < 2:
                continue
            probes[probe_id] = (
                LogisticRegression(max_iter=1000, solver="lbfgs")
                .fit(X_train, y_train)
            )

        if not probes:
            print(f"  layer {layer}: no probes trained; skipping")
            continue

        tests = _build_test_sets(layer, hidden_dim)

        for probe_id, probe in probes.items():
            for test_name, X_test, y_test in tests:
                metrics = _evaluate(probe, X_test, y_test)
                for k, v in metrics.items():
                    row[f"{probe_id}__{test_name}__{k}"] = v

        all_results.append(row)

    save_json(all_results, str(results_dir / "probe_results.json"))

    # ── Best-layer summary ─────────────────────────────────────────────────
    if all_results:
        all_test_names = sorted({
            key.split("__")[1]
            for row in all_results
            for key in row
            if "__" in key and key != "layer"
        })
        all_probe_ids = [pid for pid, _, _ in PROBE_DEFS]
        print(f"\nProbe Results Summary (best layer per probe×test):")
        for probe in all_probe_ids:
            for test in all_test_names:
                key = f"{probe}__{test}__accuracy"
                accs = [(r["layer"], r.get(key)) for r in all_results
                        if r.get(key) is not None]
                if accs:
                    best = max(accs, key=lambda x: x[1])
                    print(f"  {probe} on {test}: best acc={best[1]:.3f} "
                          f"at layer {best[0]}")


if __name__ == "__main__":
    main()
