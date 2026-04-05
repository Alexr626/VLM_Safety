#!/usr/bin/env python3
"""
Experiment 1: Augmented Baseline Projections
=============================================
Extend the existing TT baseline analysis to include CT (cohesive text)
representations, and optionally incorporate behavioral labels from
Experiment 3.

Key question: Does fusing image semantics into a natural text query (CT)
reveal a safety direction gap that caption+text (TT) misses?

Prerequisites
-------------
  1. Safety direction vectors computed (vl_activation_shift.py)
  2. TT + CT activations extracted (extract_tt.py, extract_ct.py)
  3. Optional: behavioral labels from classify_responses.py

Outputs
-------
  outputs/results/augmented_baseline_stats.json
  outputs/artifacts/augmented_baseline_per_sample.npz

Usage
-----
  python .../augmented_baseline_projections.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent
_MODEL_NAME = _EXPERIMENT_DIR.parent.name
_PROJECT_ROOT = _EXPERIMENT_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_DATA = _PROJECT_ROOT / "data"
_ARTIFACTS = _PROJECT_ROOT / "experiment_artifacts" / _MODEL_NAME
_RESULTS = _EXPERIMENT_DIR / "outputs" / "results"
_LOCAL_ARTIFACTS = _EXPERIMENT_DIR / "outputs" / "artifacts"
_PLOTS = _RESULTS / "plots"

for d in [_RESULTS, _LOCAL_ARTIFACTS, _PLOTS]:
    d.mkdir(parents=True, exist_ok=True)

from src.extraction import ActivationCache, load_json, save_json, save_npz


def _ttest(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b).pvalue)


def _mannwhitney(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def main():
    # ── Load safety directions ───────────────────────────────────────────────
    sd_path = _ARTIFACTS / "vl_activation_shift" / "safety_direction_vectors.npz"
    if not sd_path.exists():
        raise FileNotFoundError(
            f"Safety direction vectors not found: {sd_path}\n"
            "Run vl_activation_shift.py first."
        )
    safety_dir = dict(np.load(sd_path))
    layers = sorted(int(k.replace("layer_", "")) for k in safety_dir)
    print(f"Loaded safety directions for {len(layers)} layers")

    # ── Load sample metadata ─────────────────────────────────────────────────
    meta_path = _DATA / "holisafe-bench" / "activations" / _MODEL_NAME / "sample_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Sample metadata not found: {meta_path}")
    metadata = load_json(str(meta_path))

    sss_ids = [m["id"] for m in metadata if m["label"] == "SSS"]
    ssu_ids = [m["id"] for m in metadata if m["label"] == "SSU"]
    all_ids = sss_ids + ssu_ids
    id_to_label = {m["id"]: m["label"] for m in metadata}
    print(f"Samples: {len(sss_ids)} SSS + {len(ssu_ids)} SSU")

    # ── Load behavioral labels (optional) ────────────────────────────────────
    refusal_path = (_EXPERIMENT_DIR.parent / "behavioral_ground_truth" /
                    "outputs" / "results" / "holisafe_refusal_labels.json")
    refusal_map = {}
    if refusal_path.exists():
        refusal_data = load_json(str(refusal_path))
        refusal_map = {r["id"]: r for r in refusal_data}
        print(f"Loaded behavioral labels for {len(refusal_map)} samples")
    else:
        print("No behavioral labels found; skipping conditional analysis.")

    # ── Load activations and compute projections ─────────────────────────────
    cache = ActivationCache(str(_DATA / "holisafe-bench" / "activations" / _MODEL_NAME))

    # Check which suffixes are available
    has_ct = cache.exists(all_ids[0], "ct") if all_ids else False
    suffixes = ["tt"]
    if has_ct:
        suffixes.append("ct")
        print("CT activations found.")
    else:
        print("CT activations not found; TT-only analysis.")

    per_sample_data = {}  # key: "{suffix}_layer_{l}" -> array of projections

    layer_stats = []
    for l in layers:
        s_l = safety_dir[f"layer_{l}"].astype(np.float64)
        s_norm_sq = float(np.dot(s_l, s_l))
        if s_norm_sq < 1e-12:
            continue

        row = {"layer": l}

        for suffix in suffixes:
            # Collect projections per sample
            sss_projs, ssu_projs = [], []
            all_projs = []

            for sid in all_ids:
                acts = cache.load_or_none(sid, suffix)
                if acts is None or l not in acts:
                    all_projs.append(float("nan"))
                    continue
                x = acts[l].astype(np.float64)
                proj = float(np.dot(x, s_l) / s_norm_sq)
                all_projs.append(proj)
                if id_to_label[sid] == "SSS":
                    sss_projs.append(proj)
                else:
                    ssu_projs.append(proj)

            per_sample_data[f"{suffix}_layer_{l}"] = np.array(all_projs, dtype=np.float32)

            sss_arr = np.array(sss_projs)
            ssu_arr = np.array(ssu_projs)

            row[f"{suffix}_SSS_mean"] = float(sss_arr.mean()) if len(sss_arr) else None
            row[f"{suffix}_SSU_mean"] = float(ssu_arr.mean()) if len(ssu_arr) else None
            row[f"{suffix}_gap"] = (
                (float(sss_arr.mean()) - float(ssu_arr.mean()))
                if len(sss_arr) and len(ssu_arr) else None
            )
            row[f"{suffix}_p_ttest"] = _ttest(sss_arr, ssu_arr)
            row[f"{suffix}_p_mannwhitney"] = _mannwhitney(sss_arr, ssu_arr)

            # Behavioral conditional analysis (within SSU)
            if refusal_map and suffix in ["vl", "tt", "ct"]:
                refused_projs = [p for sid, p in zip(ssu_ids, ssu_projs)
                                 if refusal_map.get(sid, {}).get(f"refused_{suffix}", None) is True
                                 and not np.isnan(p)]
                complied_projs = [p for sid, p in zip(ssu_ids, ssu_projs)
                                  if refusal_map.get(sid, {}).get(f"refused_{suffix}", None) is False
                                  and not np.isnan(p)]
                if refused_projs and complied_projs:
                    row[f"{suffix}_ssu_refused_mean"] = float(np.mean(refused_projs))
                    row[f"{suffix}_ssu_complied_mean"] = float(np.mean(complied_projs))
                    row[f"{suffix}_ssu_behavioral_p"] = _mannwhitney(
                        np.array(refused_projs), np.array(complied_projs))

        layer_stats.append(row)

    # ── Save ─────────────────────────────────────────────────────────────────
    save_json(layer_stats, str(_RESULTS / "augmented_baseline_stats.json"))
    save_npz(per_sample_data, str(_LOCAL_ARTIFACTS / "augmented_baseline_per_sample.npz"))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\nResults saved to {_RESULTS}/")
    for suffix in suffixes:
        gaps = [r.get(f"{suffix}_gap") for r in layer_stats if r.get(f"{suffix}_gap") is not None]
        if gaps:
            best_idx = np.argmax(np.abs(gaps))
            best_layer = layer_stats[best_idx]["layer"]
            print(f"  {suffix.upper()}: largest gap at layer {best_layer} = {gaps[best_idx]:.4f}")


if __name__ == "__main__":
    main()
