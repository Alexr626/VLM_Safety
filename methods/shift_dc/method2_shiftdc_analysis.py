#!/usr/bin/env python3
"""
Method 2: ShiftDC Analysis
===========================
Implements the ShiftDC diagnostic from:
  "Understanding and Rectifying Safety Perception Distortion in VLMs"
  (Zou et al., 2025)

Step 1 — Safety direction:
  s^l = mean(LLaVA-Instruct activations^l) - mean(MM-SafetyBench activations^l)

Step 2 — Per-sample modality shift:
  m^l          = x_vl^l - x_tt^l
  cosine_sim^l = cosine(m^l, s^l)
  proj_mag^l   = dot(m^l, s^l) / ||s^l||^2

Step 3 — Aggregate SSS vs SSU per layer, run t-test.

Prerequisites
-------------
  1. extract_vl.py             — VL activations cached
  2. extract_tt.py             — TT activations cached
  3. extract_ref_activations.py — reference matrices exist

Outputs (under outputs/{model_name}/method2_shiftdc/)
-------
  safety_direction_vectors.npz  — s^l per layer
  per_sample_shifts.json        — cosine_sim / proj_mag per sample per layer
  aggregate_stats.json          — SSS/SSU means + t-test p-values per layer
  sample_metadata.json          — id / label / category (used by followup scripts)

Usage
-----
  python methods/shift_dc/method2_shiftdc_analysis.py
  python methods/shift_dc/method2_shiftdc_analysis.py --limit 50
  python methods/shift_dc/method2_shiftdc_analysis.py --skip_safety_dir
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import load_holisafe, filter_subsets
from src.extraction import ActivationCache, save_json, save_npz


# ── Safety direction ──────────────────────────────────────────────────────────

def compute_safety_direction(ref_base):
    """
    PCA-based steering vector extraction (Section 3.3):
      1. Mean-center H+ and H- using joint mean μ = (mean(H+) + mean(H-))/2
      2. Concatenate [H+_centered; H-_centered]
      3. vector_l = first principal component (first right singular vector of SVD)
      4. Flip sign so vector points from unsafe → safe (aligns with mean difference)
    """
    safe_npz   = np.load(ref_base / "llava-instruct" / "activation_matrices.npz")
    unsafe_npz = np.load(ref_base / "mm-safetybench" / "activation_matrices.npz")
    layers = sorted(int(k.replace("safe_layer_", "")) for k in safe_npz.files)
    print(f"  Running PCA over {len(layers)} layers ...")
    safety_dir = {}
    for l in layers:
        H_pos = safe_npz[f"safe_layer_{l}"].astype(np.float64)    # (N+, d)
        H_neg = unsafe_npz[f"unsafe_layer_{l}"].astype(np.float64) # (N-, d)

        # Joint mean-centering
        mu = (H_pos.mean(axis=0) + H_neg.mean(axis=0)) / 2
        M  = np.concatenate([H_pos - mu, H_neg - mu], axis=0)     # (N+ + N-, d)

        # First principal component via SVD
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        vector = Vt[0]  # (d,)

        # Canonical sign: align with safe - unsafe mean direction
        if np.dot(vector, H_pos.mean(axis=0) - H_neg.mean(axis=0)) < 0:
            vector = -vector

        safety_dir[f"layer_{l}"] = vector.astype(np.float32)
    return safety_dir, layers


# ── Shift computation ─────────────────────────────────────────────────────────

def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _projection(m, s):
    s2 = float(np.dot(s, s))
    if s2 < 1e-12:
        return float("nan")
    return float(np.dot(m, s) / s2)


def compute_shifts(samples, safety_dir, cache, layers):
    per_sample = []
    layer_data = {l: {"sss_cos": [], "sss_proj": [], "ssu_cos": [], "ssu_proj": []}
                  for l in layers}

    for sample in tqdm(samples, desc="Computing shifts"):
        sid, lbl = sample["id"], sample["label"]
        vl = cache.load_or_none(sid, "vl")
        tt = cache.load_or_none(sid, "tt")
        if vl is None or tt is None:
            continue

        record = {"sample_id": sid, "label": lbl, "category": sample["category"],
                  "per_layer": {}}

        for l in layers:
            if l not in vl or l not in tt or f"layer_{l}" not in safety_dir:
                continue
            m   = vl[l].astype(np.float64) - tt[l].astype(np.float64)
            s   = safety_dir[f"layer_{l}"].astype(np.float64)
            cos = _cosine(m, s)
            proj = _projection(m, s)
            record["per_layer"][str(l)] = {"cosine_sim": cos, "proj_mag": proj}
            key = lbl.lower()
            if not np.isnan(cos):
                layer_data[l][f"{key}_cos"].append(cos)
            if not np.isnan(proj):
                layer_data[l][f"{key}_proj"].append(proj)

        per_sample.append(record)
    return per_sample, layer_data


# ── Aggregate + t-test ────────────────────────────────────────────────────────

def aggregate(layer_data, layers):
    def _ttest(a, b):
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        return float(stats.ttest_ind(a, b).pvalue)

    results = []
    for l in sorted(layers):
        d = layer_data[l]
        sss_cos  = np.array(d["sss_cos"])
        ssu_cos  = np.array(d["ssu_cos"])
        sss_proj = np.array(d["sss_proj"])
        ssu_proj = np.array(d["ssu_proj"])
        results.append({
            "layer": l,
            "SSS_mean_cosine": float(sss_cos.mean())  if len(sss_cos)  else None,
            "SSU_mean_cosine": float(ssu_cos.mean())  if len(ssu_cos)  else None,
            "SSS_mean_proj":   float(sss_proj.mean()) if len(sss_proj) else None,
            "SSU_mean_proj":   float(ssu_proj.mean()) if len(ssu_proj) else None,
            "p_cosine":        _ttest(sss_cos, ssu_cos),
            "p_proj":          _ttest(sss_proj, ssu_proj),
            "n_sss": len(sss_cos), "n_ssu": len(ssu_cos),
        })
    return results


# ── CLI + main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--dataset", default="holisafe")
    p.add_argument("--output_dir", default=str(_PROJECT_ROOT / "outputs"))
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip_safety_dir", action="store_true",
                   help="Skip recomputing s^l if safety_direction_vectors.npz already exists")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    out_base = Path(args.output_dir) / model_name
    out_dir  = out_base / "method2_shiftdc"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Safety direction ──────────────────────────────────────────────
    sd_path = out_dir / "safety_direction_vectors.npz"
    if args.skip_safety_dir and sd_path.exists():
        print("Loading existing safety direction vectors ...")
        safety_dir = dict(np.load(sd_path))
        layers = sorted(int(k.replace("layer_", "")) for k in safety_dir)
    else:
        print("Computing safety direction vectors ...")
        safety_dir, layers = compute_safety_direction(out_base / "reference")
        save_npz(safety_dir, str(sd_path))
        print(f"  → {len(layers)} layers saved to {sd_path}")

    # ── Step 2: Load samples ──────────────────────────────────────────────────
    entries, images_base = load_holisafe(cache_dir=args.cache_dir)
    sss, ssu = filter_subsets(entries, images_base)
    if args.limit:
        sss, ssu = sss[:args.limit], ssu[:args.limit]
    samples = sss + ssu
    print(f"Samples: {len(sss)} SSS + {len(ssu)} SSU = {len(samples)}")

    # ── Step 3: Compute shifts ────────────────────────────────────────────────
    cache = ActivationCache(str(out_base / "activations" / args.dataset))
    per_sample, layer_data = compute_shifts(samples, safety_dir, cache, layers)
    agg = aggregate(layer_data, layers)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_json(per_sample, str(out_dir / "per_sample_shifts.json"))
    save_json(agg,        str(out_dir / "aggregate_stats.json"))
    save_json(
        [{"id": s["id"], "label": s["label"], "category": s["category"],
          "text_snippet": s["text"][:120]} for s in samples],
        str(out_dir / "sample_metadata.json"),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    def _diff(r):
        a, b = r.get("SSS_mean_cosine"), r.get("SSU_mean_cosine")
        return abs(a - b) if (a is not None and b is not None) else 0.0

    best = max(agg, key=_diff)
    print(f"\nLargest SSS/SSU cosine split at layer {best['layer']}:")
    print(f"  SSS mean cosine = {best['SSS_mean_cosine']:.4f}")
    print(f"  SSU mean cosine = {best['SSU_mean_cosine']:.4f}")
    print(f"  p-value         = {best['p_cosine']:.4e}")
    print(f"\nDone → {out_dir}/")


if __name__ == "__main__":
    main()
