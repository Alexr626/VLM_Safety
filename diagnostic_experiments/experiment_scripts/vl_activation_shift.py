#!/usr/bin/env python3
"""
ShiftDC Analysis: VL Activation Shift
======================================
Implements the ShiftDC diagnostic.

Step 1 — Semantic safety direction:
  s^l is the top-1 PC of the per-pair *difference* matrix
      D = H_safe' - H_unsafe'
  where H_safe' and H_unsafe' are row-aligned by parsed sample-id (e.g.
  catqa_harmless_N ↔ catqa_harmful_N). Pairwise PCA isolates the safety
  axis from per-pair topic/style residual that joint PCA absorbs.

  When the configured refs lack pair structure (e.g. mm-safetybench +
  llava-instruct), s^l falls back to the original CAST-style joint PCA
  on the centered, vertically-stacked activation matrix.

Step 2 — Per-sample modality shift:
  m^l          = x_vl^l - x_tt^l
  cosine_sim^l = cosine(m^l, s^l)
  proj_mag^l   = dot(m^l, s^l) / ||s^l||^2

When --compositional_safety_dir is passed, also project onto the
compositional safety direction c^l and record comp_cosine_sim / comp_proj_mag.

Outputs (under diagnostic_experiments/{model}/shift_dc/outputs/)
-------
  (experiment_artifacts)/safety_direction_vectors.npz       — s^l per layer (pairwise)
  (experiment_artifacts)/safety_direction_vectors_joint.npz — joint-PCA s^l (sanity)
  results/vl_activation_shift/per_sample_shifts.json
  results/vl_activation_shift/aggregate_stats.json
  results/vl_activation_shift/sample_metadata.json
  results/vl_activation_shift/recipe_sanity.json            — per-layer cos(pair, joint)

Note on `--skip_safety_dir`: pre-existing `safety_direction_vectors.npz` files
written by earlier joint-only versions of this script will be silently reused
unless that flag is omitted. To regenerate cleanly, delete
`experiment_artifacts/{model}/vl_activation_shift/` once before the first run
after this change.
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
from src.extraction import (
    ActivationCache, save_json, save_npz, pairwise_difference_matrix,
)
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
    meta = json.loads(meta_path.read_text())
    npz = np.load(ref_dir / "activation_matrices.npz")
    return npz, meta["role"], meta.get("sample_ids", [])


def _joint_pca_direction(H_pos: np.ndarray, H_neg: np.ndarray) -> np.ndarray:
    """CAST-style joint PCA: top-1 PC of the centered, vertically-stacked
    safe/unsafe activation matrix, sign-oriented toward `safe`.
    """
    mu = (H_pos.mean(axis=0) + H_neg.mean(axis=0)) / 2
    M = np.concatenate([H_pos - mu, H_neg - mu], axis=0)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    v = Vt[0]
    if np.dot(v, H_pos.mean(axis=0) - H_neg.mean(axis=0)) < 0:
        v = -v
    return v


def _pairwise_pca_direction(D: np.ndarray, H_pos: np.ndarray,
                            H_neg: np.ndarray) -> np.ndarray:
    """Top-1 PC of the centered per-pair difference matrix `D = H_safe' -
    H_unsafe'`, sign-oriented toward `safe` using class-mean midpoints.
    """
    Dc = D - D.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Dc, full_matrices=False)
    v = Vt[0]
    if np.dot(v, H_pos.mean(axis=0) - H_neg.mean(axis=0)) < 0:
        v = -v
    return v


def compute_safety_direction(ref_base, safe_ref, unsafe_ref,
                             return_joint: bool = False):
    """Compute per-layer semantic safety direction s^l.

    Default recipe: pairwise PCA on row-aligned per-pair difference vectors
    (when both refs share an integer-indexed prefix structure such as
    `catqa_harmless_N` / `catqa_harmful_N`). Falls back to joint PCA when
    pair structure is not detected.

    Args:
        ref_base: Path to data/{...}/activations/{model}/.
        safe_ref / unsafe_ref: subdirectory names (e.g. catqa-harmless,
            catqa-harmful, mm-safetybench, llava-instruct).
        return_joint: when True, also compute the joint-PCA direction in
            parallel and return it as a third value (used for the sanity
            artifact written by main()).

    Returns:
        (safety_dir, layers) by default, or
        (safety_dir, safety_dir_joint, layers, n_pairs) when return_joint=True.
        n_pairs is the count of aligned pairs used (0 when pairwise was
        not applicable and joint fallback was used for s^l).
    """
    pos_npz, pos_role, pos_ids = _load_ref_npz(ref_base / safe_ref)
    neg_npz, neg_role, neg_ids = _load_ref_npz(ref_base / unsafe_ref)
    print(f"  safe_ref='{safe_ref}' (role={pos_role}, n={len(pos_ids)}), "
          f"unsafe_ref='{unsafe_ref}' (role={neg_role}, n={len(neg_ids)})")
    layers = sorted(int(k.replace(f"{pos_role}_layer_", ""))
                    for k in pos_npz.files if k.startswith(f"{pos_role}_layer_"))

    # Probe pair structure once on layer 0 to decide the recipe.
    safe_prefix = f"{safe_ref.replace('-', '_')}_"
    unsafe_prefix = f"{unsafe_ref.replace('-', '_')}_"
    H_pos0 = pos_npz[f"{pos_role}_layer_{layers[0]}"].astype(np.float64)
    H_neg0 = neg_npz[f"{neg_role}_layer_{layers[0]}"].astype(np.float64)
    D0, n_pairs = pairwise_difference_matrix(
        H_pos0, pos_ids, H_neg0, neg_ids,
        safe_prefix=safe_prefix, unsafe_prefix=unsafe_prefix,
    )
    use_pairwise = D0 is not None
    if use_pairwise:
        smaller = min(len(pos_ids), len(neg_ids))
        if n_pairs < smaller:
            print(f"  WARN: pairwise alignment used {n_pairs} pairs out of "
                  f"min({len(pos_ids)}, {len(neg_ids)})={smaller} "
                  f"(some samples were dropped during extraction)")
        print(f"  Recipe: PAIRWISE PCA on per-pair differences "
              f"(n_pairs={n_pairs}) over {len(layers)} layers ...")
    else:
        print(f"  WARN: pair structure not detected for "
              f"'{safe_ref}'/'{unsafe_ref}' (prefixes='{safe_prefix}', "
              f"'{unsafe_prefix}'). Falling back to JOINT PCA.")
        print(f"  Recipe: JOINT PCA over {len(layers)} layers ...")

    safety_dir, safety_dir_joint = {}, {}
    for l in layers:
        H_pos = pos_npz[f"{pos_role}_layer_{l}"].astype(np.float64)
        H_neg = neg_npz[f"{neg_role}_layer_{l}"].astype(np.float64)
        v_joint = _joint_pca_direction(H_pos, H_neg)
        if use_pairwise:
            D, _ = pairwise_difference_matrix(
                H_pos, pos_ids, H_neg, neg_ids,
                safe_prefix=safe_prefix, unsafe_prefix=unsafe_prefix,
            )
            v_pair = _pairwise_pca_direction(D, H_pos, H_neg)
            safety_dir[f"layer_{l}"] = v_pair.astype(np.float32)
        else:
            safety_dir[f"layer_{l}"] = v_joint.astype(np.float32)
        if return_joint:
            safety_dir_joint[f"layer_{l}"] = v_joint.astype(np.float32)

    if return_joint:
        return safety_dir, safety_dir_joint, layers, (n_pairs if use_pairwise else 0)
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


def compute_shifts(samples, safety_dir, cache, layers, comp_dir=None):
    per_sample = []
    layer_data = {l: {"sss_cos": [], "sss_proj": [], "ssu_cos": [], "ssu_proj": [],
                      "sss_comp_cos": [], "sss_comp_proj": [],
                      "ssu_comp_cos": [], "ssu_comp_proj": []}
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

            # Compositional safety projections
            if comp_dir is not None and f"layer_{l}" in comp_dir:
                c = comp_dir[f"layer_{l}"].astype(np.float64)
                comp_cos = _cosine(m, c)
                comp_proj = _projection(m, c)
                entry["comp_cosine_sim"] = comp_cos
                entry["comp_proj_mag"] = comp_proj
                if not np.isnan(comp_cos):
                    layer_data[l][f"{key}_comp_cos"].append(comp_cos)
                if not np.isnan(comp_proj):
                    layer_data[l][f"{key}_comp_proj"].append(comp_proj)

            record["per_layer"][str(l)] = entry

        per_sample.append(record)
    return per_sample, layer_data


# ── Aggregate + t-test ────────────────────────────────────────────────────────

def aggregate(layer_data, layers, has_comp=False):
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
        if has_comp:
            sss_cc = np.array(d["sss_comp_cos"])
            ssu_cc = np.array(d["ssu_comp_cos"])
            sss_cp = np.array(d["sss_comp_proj"])
            ssu_cp = np.array(d["ssu_comp_proj"])
            row.update({
                "SSS_mean_comp_cosine": float(sss_cc.mean()) if len(sss_cc) else None,
                "SSU_mean_comp_cosine": float(ssu_cc.mean()) if len(ssu_cc) else None,
                "SSS_mean_comp_proj": float(sss_cp.mean()) if len(sss_cp) else None,
                "SSU_mean_comp_proj": float(ssu_cp.mean()) if len(ssu_cp) else None,
                "p_comp_cosine": _ttest(sss_cc, ssu_cc),
                "p_comp_proj": _ttest(sss_cp, ssu_cp),
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
    p.add_argument("--comp_source",
                   choices=["none", "holisafe_tt", "holisafe_vl",
                            "mssbench_tt", "mssbench_vl"],
                   default="none",
                   help="Optional compositional direction to also project shifts "
                        "onto. 'none' (default) skips this projection. Otherwise "
                        "loads experiment_artifacts/{model}/compositional_safety/"
                        "{comp_source}/compositional_safety_direction_vectors.npz "
                        "and records comp_cosine_sim/comp_proj_mag per layer.")
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
    sd_joint_path = experiment_artifacts / "safety_direction_vectors_joint.npz"
    if args.skip_safety_dir and sd_path.exists():
        print("Loading existing safety direction vectors ...")
        safety_dir = dict(np.load(sd_path))
        layers = sorted(int(k.replace("layer_", "")) for k in safety_dir)
    else:
        print("Computing safety direction vectors ...")
        safety_dir, safety_dir_joint, layers, n_pairs = compute_safety_direction(
            _PROJECT_ROOT / "data" / "catqa-contrastive" / "activations" / model_name,
            args.safe_ref, args.unsafe_ref, return_joint=True)
        save_npz(safety_dir, str(sd_path))
        save_npz(safety_dir_joint, str(sd_joint_path))
        print(f"  → {len(layers)} layers saved to {sd_path}")
        print(f"  → joint sanity copy saved to {sd_joint_path}")

        # Per-layer cosine sanity: how much do pair vs joint disagree?
        recipe_rows = []
        for l in layers:
            v_p = safety_dir[f"layer_{l}"].astype(np.float64)
            v_j = safety_dir_joint[f"layer_{l}"].astype(np.float64)
            np_, nj = np.linalg.norm(v_p), np.linalg.norm(v_j)
            cos = float(np.dot(v_p, v_j) / (np_ * nj)) if (np_ > 1e-12 and nj > 1e-12) else float("nan")
            recipe_rows.append({"layer": l, "cos_pair_vs_joint": cos,
                                "n_pairs": int(n_pairs)})
        save_json(recipe_rows, str(out_dir / "recipe_sanity.json"))
        cos_vals = [r["cos_pair_vs_joint"] for r in recipe_rows
                    if not np.isnan(r["cos_pair_vs_joint"])]
        if cos_vals:
            print(f"  cos(s_pair, s_joint): mean={np.mean(cos_vals):.4f} "
                  f"min={np.min(cos_vals):.4f} max={np.max(cos_vals):.4f}")

    # ── Load compositional safety direction (optional) ──────────────────────
    comp_dir = None
    if args.comp_source != "none":
        comp_path = (_PROJECT_ROOT / "experiment_artifacts" / model_name /
                     "compositional_safety" / args.comp_source /
                     "compositional_safety_direction_vectors.npz")
        if not comp_path.exists():
            print(f"ERROR: --comp_source={args.comp_source} requested but "
                  f"{comp_path} does not exist.")
            print(f"Run: python diagnostic_experiments/experiment_scripts/"
                  f"compositional_safety_direction.py "
                  f"--source {args.comp_source.split('_')[0]} "
                  f"--representation {args.comp_source.split('_')[1]}")
            sys.exit(1)
        comp_dir = dict(np.load(comp_path))
        print(f"Loaded compositional safety direction ({args.comp_source}) "
              f"for {len(comp_dir)} layers")

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
    per_sample, layer_data = compute_shifts(samples, safety_dir, cache, layers, comp_dir)
    agg = aggregate(layer_data, layers, has_comp=comp_dir is not None)

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
