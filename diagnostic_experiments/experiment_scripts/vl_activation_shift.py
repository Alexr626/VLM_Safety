#!/usr/bin/env python3
"""
ShiftDC Analysis: VL Activation Shift
======================================
Implements the ShiftDC diagnostic.

Step 1 — Safety direction (content):
  s^l computed via CAST-style PCA on safe vs unsafe reference activations.

Step 2 — Per-sample modality shift:
  m^l          = x_vl^l - x_tt^l
  cosine_sim^l = cosine(m^l, s^l)
  proj_mag^l   = dot(m^l, s^l) / ||s^l||^2

When --combinatorial_dir is passed, also project onto the combinatorial
direction c^l and record comb_cosine_sim / comb_proj_mag.

Outputs (under diagnostic_experiments/{model}/shift_dc/outputs/)
-------
  (experiment_artifacts)/safety_direction_vectors.npz  — s^l per layer
  results/vl_activation_shift/per_sample_shifts.json
  results/vl_activation_shift/aggregate_stats.json
  results/vl_activation_shift/sample_metadata.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_DIAGNOSTIC_ROOT = _SCRIPT_DIR.parent                      # diagnostic_experiments/
_PROJECT_ROOT = _DIAGNOSTIC_ROOT.parent                    # VLM_Safety/
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import load_holisafe, filter_subsets
from src.extraction import ActivationCache, save_json, save_npz
from src.model import _normalize_model_name

_EXPERIMENT_NAME = "shift_dc"


# ── Safety direction ──────────────────────────────────────────────────────────

def _load_ref_npz(ref_dir: Path):
    meta_path = ref_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {ref_dir}. "
            "Re-run extract_ref_activations.py to regenerate."
        )
    role = json.loads(meta_path.read_text())["role"]
    npz = np.load(ref_dir / "activation_matrices.npz")
    return npz, role


def compute_safety_direction(ref_base, safe_ref, unsafe_ref):
    pos_npz, pos_role = _load_ref_npz(ref_base / safe_ref)
    neg_npz, neg_role = _load_ref_npz(ref_base / unsafe_ref)
    print(f"  safe_ref='{safe_ref}' (role={pos_role}), "
          f"unsafe_ref='{unsafe_ref}' (role={neg_role})")
    layers = sorted(int(k.replace(f"{pos_role}_layer_", ""))
                    for k in pos_npz.files if k.startswith(f"{pos_role}_layer_"))
    print(f"  Running PCA over {len(layers)} layers ...")
    safety_dir = {}
    for l in layers:
        H_pos = pos_npz[f"{pos_role}_layer_{l}"].astype(np.float64)
        H_neg = neg_npz[f"{neg_role}_layer_{l}"].astype(np.float64)
        mu = (H_pos.mean(axis=0) + H_neg.mean(axis=0)) / 2
        M = np.concatenate([H_pos - mu, H_neg - mu], axis=0)
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        vector = Vt[0]
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


def compute_shifts(samples, safety_dir, cache, layers, comb_dir=None):
    per_sample = []
    layer_data = {l: {"sss_cos": [], "sss_proj": [], "ssu_cos": [], "ssu_proj": [],
                      "sss_comb_cos": [], "sss_comb_proj": [],
                      "ssu_comb_cos": [], "ssu_comb_proj": []}
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
            m = vl[l].astype(np.float64) - tt[l].astype(np.float64)
            s = safety_dir[f"layer_{l}"].astype(np.float64)
            cos = _cosine(m, s)
            proj = _projection(m, s)
            entry = {"cosine_sim": cos, "proj_mag": proj}

            key = lbl.lower()
            if not np.isnan(cos):
                layer_data[l][f"{key}_cos"].append(cos)
            if not np.isnan(proj):
                layer_data[l][f"{key}_proj"].append(proj)

            # Combinatorial projections
            if comb_dir is not None and f"layer_{l}" in comb_dir:
                c = comb_dir[f"layer_{l}"].astype(np.float64)
                comb_cos = _cosine(m, c)
                comb_proj = _projection(m, c)
                entry["comb_cosine_sim"] = comb_cos
                entry["comb_proj_mag"] = comb_proj
                if not np.isnan(comb_cos):
                    layer_data[l][f"{key}_comb_cos"].append(comb_cos)
                if not np.isnan(comb_proj):
                    layer_data[l][f"{key}_comb_proj"].append(comb_proj)

            record["per_layer"][str(l)] = entry

        per_sample.append(record)
    return per_sample, layer_data


# ── Aggregate + t-test ────────────────────────────────────────────────────────

def aggregate(layer_data, layers, has_comb=False):
    def _ttest(a, b):
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        return float(stats.ttest_ind(a, b).pvalue)

    results = []
    for l in sorted(layers):
        d = layer_data[l]
        sss_cos = np.array(d["sss_cos"])
        ssu_cos = np.array(d["ssu_cos"])
        sss_proj = np.array(d["sss_proj"])
        ssu_proj = np.array(d["ssu_proj"])
        row = {
            "layer": l,
            "SSS_mean_cosine": float(sss_cos.mean()) if len(sss_cos) else None,
            "SSU_mean_cosine": float(ssu_cos.mean()) if len(ssu_cos) else None,
            "SSS_mean_proj": float(sss_proj.mean()) if len(sss_proj) else None,
            "SSU_mean_proj": float(ssu_proj.mean()) if len(ssu_proj) else None,
            "p_cosine": _ttest(sss_cos, ssu_cos),
            "p_proj": _ttest(sss_proj, ssu_proj),
            "n_sss": len(sss_cos), "n_ssu": len(ssu_cos),
        }
        if has_comb:
            sss_cc = np.array(d["sss_comb_cos"])
            ssu_cc = np.array(d["ssu_comb_cos"])
            sss_cp = np.array(d["sss_comb_proj"])
            ssu_cp = np.array(d["ssu_comb_proj"])
            row.update({
                "SSS_mean_comb_cosine": float(sss_cc.mean()) if len(sss_cc) else None,
                "SSU_mean_comb_cosine": float(ssu_cc.mean()) if len(ssu_cc) else None,
                "SSS_mean_comb_proj": float(sss_cp.mean()) if len(sss_cp) else None,
                "SSU_mean_comb_proj": float(ssu_cp.mean()) if len(ssu_cp) else None,
                "p_comb_cosine": _ttest(sss_cc, ssu_cc),
                "p_comb_proj": _ttest(sss_cp, ssu_cp),
            })
        results.append(row)
    return results


# ── CLI + main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--dataset", default="holisafe")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip_safety_dir", action="store_true",
                   help="Skip recomputing s^l if safety_direction_vectors.npz already exists")
    p.add_argument("--safe_ref", default="catqa-harmless",
                   help="Subdirectory name under data/catqa-contrastive/activations/{model}/")
    p.add_argument("--unsafe_ref", default="catqa-harmful")
    p.add_argument("--combinatorial_dir", action="store_true",
                   help="Also project shifts onto the combinatorial direction c^l")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)

    # ── Path resolution ──────────────────────────────────────────────────────
    experiment_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME
    out_base = experiment_dir / "outputs"
    out_dir = out_base / "results" / "vl_activation_shift"
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_artifacts = _PROJECT_ROOT / "experiment_artifacts" / model_name / "vl_activation_shift"
    experiment_artifacts.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Safety direction ─────────────────────────────────────────────
    sd_path = experiment_artifacts / "safety_direction_vectors.npz"
    if args.skip_safety_dir and sd_path.exists():
        print("Loading existing safety direction vectors ...")
        safety_dir = dict(np.load(sd_path))
        layers = sorted(int(k.replace("layer_", "")) for k in safety_dir)
    else:
        print("Computing safety direction vectors ...")
        safety_dir, layers = compute_safety_direction(
            _PROJECT_ROOT / "data" / "catqa-contrastive" / "activations" / model_name,
            args.safe_ref, args.unsafe_ref)
        save_npz(safety_dir, str(sd_path))
        print(f"  → {len(layers)} layers saved to {sd_path}")

    # ── Load combinatorial direction (optional) ─────────────────────────────
    comb_dir = None
    if args.combinatorial_dir:
        comb_path = (_PROJECT_ROOT / "experiment_artifacts" / model_name /
                     "combinatorial_safety" / "combinatorial_direction_vectors.npz")
        if not comb_path.exists():
            print(f"ERROR: --combinatorial_dir requested but {comb_path} does not exist.")
            print("Run diagnostic_experiments/experiment_scripts/combinatorial_direction.py first.")
            sys.exit(1)
        comb_dir = dict(np.load(comb_path))
        print(f"Loaded combinatorial direction for {len(comb_dir)} layers")

    # ── Step 2: Load samples ─────────────────────────────────────────────────
    entries, images_base = load_holisafe(cache_dir=args.cache_dir)
    sss, ssu = filter_subsets(entries, images_base)
    if args.limit:
        sss, ssu = sss[:args.limit], ssu[:args.limit]
    samples = sss + ssu
    print(f"Samples: {len(sss)} SSS + {len(ssu)} SSU = {len(samples)}")

    # ── Step 3: Compute shifts ───────────────────────────────────────────────
    cache = ActivationCache(str(
        _PROJECT_ROOT / "data" / "holisafe-bench" / "activations" / model_name))
    per_sample, layer_data = compute_shifts(samples, safety_dir, cache, layers, comb_dir)
    agg = aggregate(layer_data, layers, has_comb=comb_dir is not None)

    # ── Save ─────────────────────────────────────────────────────────────────
    save_json(per_sample, str(out_dir / "per_sample_shifts.json"))
    save_json(agg, str(out_dir / "aggregate_stats.json"))
    save_json(
        [{"id": s["id"], "label": s["label"], "category": s["category"],
          "text_snippet": s["text"][:120]} for s in samples],
        str(out_dir / "sample_metadata.json"),
    )

    # ── Summary ──────────────────────────────────────────────────────────────
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
