#!/usr/bin/env python3
"""
Augmented Baseline Projections (TT vs CT)
==========================================
Extend the existing TT baseline analysis to include CT (cohesive text)
representations, and optionally incorporate behavioral labels.

Outputs
-------
  diagnostic_experiments/{model}/augmented_baseline/outputs/results/augmented_baseline_stats.json
  diagnostic_experiments/{model}/augmented_baseline/outputs/artifacts/augmented_baseline_per_sample.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
_DIAGNOSTIC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DIAGNOSTIC_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction import ActivationCache, load_json, save_json, save_npz
from src.model import _normalize_model_name

_EXPERIMENT_NAME = "augmented_baseline"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    return p.parse_args()


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


def _refused(label_entry, suffix):
    """Return True/False/None from legacy or twoaxis schema."""
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
    shared_artifacts = _PROJECT_ROOT / "experiment_artifacts" / model_name
    experiment_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME
    results_dir = experiment_dir / "outputs" / "results"
    local_artifacts = experiment_dir / "outputs" / "artifacts"
    for d in [results_dir, local_artifacts]:
        d.mkdir(parents=True, exist_ok=True)

    sd_path = shared_artifacts / "vl_activation_shift" / "safety_direction_vectors.npz"
    if not sd_path.exists():
        raise FileNotFoundError(f"Safety direction vectors not found: {sd_path}")
    safety_dir = dict(np.load(sd_path))
    layers = sorted(int(k.replace("layer_", "")) for k in safety_dir)
    print(f"Loaded safety directions for {len(layers)} layers")

    meta_path = _DATA / "holisafe-bench" / "activations" / model_name / "sample_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Sample metadata not found: {meta_path}")
    metadata = load_json(str(meta_path))

    sss_ids = [m["id"] for m in metadata if m["label"] == "SSS"]
    ssu_ids = [m["id"] for m in metadata if m["label"] == "SSU"]
    all_ids = sss_ids + ssu_ids
    id_to_label = {m["id"]: m["label"] for m in metadata}
    print(f"Samples: {len(sss_ids)} SSS + {len(ssu_ids)} SSU")

    refusal_path = (_DIAGNOSTIC_ROOT / model_name / "behavioral_ground_truth" /
                    "outputs" / "results" / "holisafe_refusal_labels.json")
    refusal_map = {}
    if refusal_path.exists():
        refusal_data = load_json(str(refusal_path))
        refusal_map = {r["id"]: r for r in refusal_data}
        print(f"Loaded behavioral labels for {len(refusal_map)} samples")

    cache = ActivationCache(str(_DATA / "holisafe-bench" / "activations" / model_name))

    has_ct = cache.exists(all_ids[0], "ct") if all_ids else False
    suffixes = ["tt"]
    if has_ct:
        suffixes.append("ct")
        print("CT activations found.")
    else:
        print("CT activations not found; TT-only analysis.")

    per_sample_data = {}
    layer_stats = []
    for l in layers:
        s_l = safety_dir[f"layer_{l}"].astype(np.float64)
        s_norm_sq = float(np.dot(s_l, s_l))
        if s_norm_sq < 1e-12:
            continue

        row = {"layer": l}

        for suffix in suffixes:
            sss_projs, ssu_projs, all_projs = [], [], []
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
            sss_arr, ssu_arr = np.array(sss_projs), np.array(ssu_projs)

            row[f"{suffix}_SSS_mean"] = float(sss_arr.mean()) if len(sss_arr) else None
            row[f"{suffix}_SSU_mean"] = float(ssu_arr.mean()) if len(ssu_arr) else None
            row[f"{suffix}_gap"] = (
                (float(sss_arr.mean()) - float(ssu_arr.mean()))
                if len(sss_arr) and len(ssu_arr) else None
            )
            row[f"{suffix}_p_ttest"] = _ttest(sss_arr, ssu_arr)
            row[f"{suffix}_p_mannwhitney"] = _mannwhitney(sss_arr, ssu_arr)

            if refusal_map and suffix in ["vl", "tt", "ct"]:
                refused_projs = [p for sid, p in zip(ssu_ids, ssu_projs)
                                 if _refused(refusal_map.get(sid), suffix) is True
                                 and not np.isnan(p)]
                complied_projs = [p for sid, p in zip(ssu_ids, ssu_projs)
                                  if _refused(refusal_map.get(sid), suffix) is False
                                  and not np.isnan(p)]
                if refused_projs and complied_projs:
                    row[f"{suffix}_ssu_refused_mean"] = float(np.mean(refused_projs))
                    row[f"{suffix}_ssu_complied_mean"] = float(np.mean(complied_projs))
                    row[f"{suffix}_ssu_behavioral_p"] = _mannwhitney(
                        np.array(refused_projs), np.array(complied_projs))

        layer_stats.append(row)

    save_json(layer_stats, str(results_dir / "augmented_baseline_stats.json"))
    save_npz(per_sample_data, str(local_artifacts / "augmented_baseline_per_sample.npz"))

    print(f"\nResults saved to {results_dir}/")
    for suffix in suffixes:
        gaps = [r.get(f"{suffix}_gap") for r in layer_stats if r.get(f"{suffix}_gap") is not None]
        if gaps:
            best_idx = int(np.argmax(np.abs(gaps)))
            best_layer = layer_stats[best_idx]["layer"]
            print(f"  {suffix.upper()}: largest gap at layer {best_layer} = {gaps[best_idx]:.4f}")


if __name__ == "__main__":
    main()
