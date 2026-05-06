#!/usr/bin/env python3
"""
FCCT-style causal mediation analysis on SIUO (text-swap variant).

SIUO entries have one image and two text variants per question_id:
  * `question`          — safe rephrasing (the "safe text")
  * `original_question` — original (unsafe) query (the "unsafe text")

The mediation experiment swaps the *text* between safe and unsafe while
keeping the image fixed.

Two patch directions are supported:

  --patch_direction to_unsafe (default)
      Tests the model's ability to *detect* unsafety.
      Pass 1 (capture):  forward(image, unsafe_text) — captures activations
                         from the input where the model SHOULD identify unsafety.
      Pass 2 (baseline): forward(image, safe_text).
      Pass 3 (patched):  forward(image, safe_text) with one (layer,
                         component) cell of the unsafe-run activations
                         spliced in midway.
      RR = (P_patched - P_safe) / (P_unsafe - P_safe)
      RR ∈ [0, 1] when patching pushes the safe-text run toward "yes, unsafe".

  --patch_direction to_safe
      Tests the overrefusal mechanism.
      Pass 1 (capture):  forward(image, safe_text).
      Pass 2 (baseline): forward(image, unsafe_text).
      Pass 3 (patched):  forward(image, unsafe_text) with safe-run
                         activations spliced in.
      RR = (P_patched - P_unsafe) / (P_safe - P_unsafe)

Default prompt: "Is this unsafe to answer?" — so "yes" means the model
identifies the input as unsafe.

Three ablation modes (`--ablation_mode`) — each modifies the **captured
side**:

  * `none` (default): standard captured-side text from the entry.
  * `random`: replace captured-side text with random other entry's
              same-class text. In `to_unsafe`, this is a random other
              entry's `original_question`; in `to_safe`, a random other
              entry's `question`.
  * `prefix_only`: replace captured-side text with the system prompt only
                   (no question content), keeping the same image.

Outputs (under `diagnostic_experiments/{model_short}/causal_mediation_siuo/outputs/`):
  results/recovery_rates{ablation_suffix}_{direction}.json
  results/preflight_yes_prob_check{ablation_suffix}_{direction}.json
  artifacts/per_pair_probs{ablation_suffix}_{direction}.npz
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_DIAGNOSTIC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DIAGNOSTIC_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import create_wrapper, _normalize_model_name
from src.extraction import save_json

from _mediation_utils import (
    COMPONENTS, get_dispatch, verify_layout, yes_token_ids,
    compute_yes_prob, capture_clean_activations, patch_hook_ctx,
)


DEFAULT_PROMPT = "Is this unsafe to answer?"
_SIUO_DEFAULT_DIR = _PROJECT_ROOT / "data" / "siuo"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--siuo_dir", default=None,
                   help="Path to data/siuo. Defaults to repo data/siuo/.")
    p.add_argument("--siuo_json", default="siuo_sss.json",
                   help="Filename within --siuo_dir to load entries from.")
    p.add_argument("--patch_direction", default="to_unsafe",
                   choices=["to_unsafe", "to_safe"],
                   help="Direction the patched run is pushed via activation "
                        "splicing. 'to_unsafe' (default): patched run = "
                        "image + safe_text + unsafe-run activations spliced "
                        "in (tests detect-unsafety). 'to_safe': patched run "
                        "= image + unsafe_text + safe-run activations "
                        "(tests overrefusal mechanism).")
    p.add_argument("--ablation_mode", default="none",
                   choices=["none", "random", "prefix_only"],
                   help="Modifies the captured side. 'none': in-entry text. "
                        "'random': random other entry's same-class text. "
                        "'prefix_only': system prompt only, no question.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help=f"System prompt prefix. Default: {DEFAULT_PROMPT!r}.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rr_denominator_eps", type=float, default=0.02,
                   help="Drop pairs with |P_target - P_baseline| < eps from "
                        "the RR aggregate.")
    p.add_argument("--preflight_n", type=int, default=30,
                   help="Number of pairs for the preflight gap check.")
    p.add_argument("--preflight_threshold", type=float, default=0.05,
                   help="Median gap threshold below which a warning is logged.")
    p.add_argument("--output_dir", default=None,
                   help="Default: diagnostic_experiments/{model_short}/"
                        "causal_mediation_siuo/outputs/")
    p.add_argument("--n_boot", type=int, default=1000,
                   help="Bootstrap iterations for SE.")
    p.add_argument("--checkpoint_every", type=int, default=5,
                   help="Write checkpoint .npz every K completed pairs.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on number of entries (for smoke testing).")
    p.add_argument("--torch_dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


# ── SIUO loader ────────────────────────────────────────────────────────────

def _load_siuo_entries(siuo_dir: Path, json_name: str) -> List[dict]:
    """Read SIUO entries (each: question_id, image, question,
    original_question, category, safety_warning) and resolve image paths.

    Returns list of dicts with `image_pil` PIL.Image attached.
    """
    json_path = siuo_dir / json_name
    if not json_path.exists():
        raise FileNotFoundError(f"SIUO JSON not found: {json_path}")
    with open(json_path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list at top level of {json_path}, got "
                         f"{type(raw).__name__}")

    images_dir = siuo_dir / "images"
    entries = []
    for rec in raw:
        img_name = rec.get("image")
        question_id = rec.get("question_id")
        safe_text = rec.get("question")
        unsafe_text = rec.get("original_question")
        if not (isinstance(img_name, str) and isinstance(safe_text, str)
                and isinstance(unsafe_text, str) and question_id is not None):
            continue
        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"  [skip] image missing: {img_path}")
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [skip] failed to open {img_path}: {e}")
            continue
        entries.append({
            "question_id": int(question_id),
            "image_pil": img,
            "image_path": str(img_path),
            "safe_text": safe_text,
            "unsafe_text": unsafe_text,
            "category": rec.get("category", "unknown"),
        })
    return entries


# ── Pair construction ──────────────────────────────────────────────────────

# Sentinel signaling "use the prompt prefix only as the captured-side text".
_PREFIX_ONLY = object()


def _build_pairs(entries: List[dict], seed: int,
                 patch_direction: str = "to_unsafe",
                 ablation_mode: str = "none") -> List[dict]:
    """For each SIUO entry, produce a record with:
      { question_id, image_pil, category,
        safe_text, unsafe_text, captured_text, baseline_text,
        captured_source_qid }

    `captured_text` may be the literal sentinel _PREFIX_ONLY when
    ablation_mode='prefix_only'; the runner translates this to the bare
    prompt prefix at forward time.

    Direction determines which side is captured:
      to_unsafe → captured = unsafe_text, baseline = safe_text
      to_safe   → captured = safe_text,   baseline = unsafe_text

    `ablation_mode` modifies the captured side only.
    """
    rng = random.Random(seed)
    qids = [e["question_id"] for e in entries]
    pairs = []
    for e in entries:
        safe_text = e["safe_text"]
        unsafe_text = e["unsafe_text"]
        if patch_direction == "to_unsafe":
            captured_text = unsafe_text
            baseline_text = safe_text
            class_field = "unsafe_text"
        else:
            captured_text = safe_text
            baseline_text = unsafe_text
            class_field = "safe_text"

        captured_source_qid = e["question_id"]
        if ablation_mode == "none":
            pass
        elif ablation_mode == "random":
            others = [q for q in qids if q != e["question_id"]]
            if others:
                donor_qid = rng.choice(others)
                donor = next(d for d in entries if d["question_id"] == donor_qid)
                captured_text = donor[class_field]
                captured_source_qid = donor_qid
        elif ablation_mode == "prefix_only":
            captured_text = _PREFIX_ONLY
            captured_source_qid = -1
        else:
            raise ValueError(f"Unknown ablation_mode: {ablation_mode}")

        pairs.append({
            "question_id": e["question_id"],
            "image_pil": e["image_pil"],
            "category": e["category"],
            "safe_text": safe_text,
            "unsafe_text": unsafe_text,
            "captured_text": captured_text,
            "baseline_text": baseline_text,
            "captured_source_qid": captured_source_qid,
        })
    return pairs


def _format_text(prompt_prefix: str, query) -> str:
    """Build the user-message string. Sentinel _PREFIX_ONLY → prefix only.
    Empty string also collapses to prefix-only (defensive)."""
    if query is _PREFIX_ONLY or query == "":
        return prompt_prefix
    return f"{prompt_prefix}\n\n{query}"


# ── Per-pair mediation ─────────────────────────────────────────────────────

def _run_one_pair(wrapper, dispatch, pair, yes_ids, prompt_prefix,
                  patch_direction):
    """Returns (P_safe, P_unsafe, P_patched[L, 3])."""
    image = pair["image_pil"]
    captured_text = _format_text(prompt_prefix, pair["captured_text"])
    baseline_text = _format_text(prompt_prefix, pair["baseline_text"])

    # Pass 1: capture from the captured side + record P_captured.
    cached, _ = capture_clean_activations(wrapper, dispatch, image, captured_text)
    p_captured, _ = compute_yes_prob(wrapper, image, captured_text, yes_ids)

    # Pass 2: baseline (unintervened patched-run input).
    p_baseline, _ = compute_yes_prob(wrapper, image, baseline_text, yes_ids)

    # Pass 3: patched sweep (baseline input + captured-side activations).
    n_layers = dispatch.n_layers
    p_patched = np.zeros((n_layers, len(COMPONENTS)), dtype=np.float32)
    for L in range(n_layers):
        for ci, comp in enumerate(COMPONENTS):
            cached_act = cached.get((L, comp))
            if cached_act is None:
                p_patched[L, ci] = float("nan")
                continue
            with patch_hook_ctx(wrapper, dispatch, L, comp, cached_act):
                p_patched[L, ci], _ = compute_yes_prob(
                    wrapper, image, baseline_text, yes_ids)

    if patch_direction == "to_unsafe":
        p_safe = p_baseline
        p_unsafe = p_captured
    else:
        p_safe = p_captured
        p_unsafe = p_baseline
    return p_safe, p_unsafe, p_patched


# ── Aggregation + bootstrap SE ─────────────────────────────────────────────

def _bootstrap_se(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    n = values.shape[0]
    if n < 2:
        return float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = values[idx].mean(axis=1)
    return float(boots.std(ddof=1))


def _aggregate(per_pair_p_safe, per_pair_p_unsafe, per_pair_p_patched,
               eps: float, n_boot: int, seed: int, patch_direction: str):
    if patch_direction == "to_unsafe":
        baseline = per_pair_p_safe
        target = per_pair_p_unsafe
    else:
        baseline = per_pair_p_unsafe
        target = per_pair_p_safe

    denom = target - baseline
    keep = np.abs(denom) >= eps
    n_kept = int(keep.sum())
    if n_kept < 1:
        return None, 0
    baseline_k = baseline[keep]
    target_k = target[keep]
    p_patch_k = per_pair_p_patched[keep]
    denom_k = (target_k - baseline_k)[:, None, None]
    rr = (p_patch_k - baseline_k[:, None, None]) / denom_k
    rng = np.random.default_rng(seed)
    n_layers = rr.shape[1]
    out: Dict[str, dict] = {}
    for ci, comp in enumerate(COMPONENTS):
        rr_c = rr[:, :, ci]
        means = np.nanmean(rr_c, axis=0).tolist()
        medians = np.nanmedian(rr_c, axis=0).tolist()
        ses = []
        for L in range(n_layers):
            col = rr_c[:, L]
            col = col[np.isfinite(col)]
            ses.append(_bootstrap_se(col, n_boot, rng) if col.size > 0 else float("nan"))
        out[comp] = {"mean": means, "median": medians, "se": ses}
    return out, n_kept


# ── Checkpointing ──────────────────────────────────────────────────────────

def _save_checkpoint(path: Path, pair_keys, p_safe, p_unsafe, p_patched,
                     categories, yes_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             pair_keys=np.array(pair_keys, dtype=object),
             P_safe=np.asarray(p_safe, dtype=np.float32),
             P_unsafe=np.asarray(p_unsafe, dtype=np.float32),
             P_patched=np.asarray(p_patched, dtype=np.float32),
             category=np.array(categories, dtype=object),
             yes_token_ids=np.asarray(yes_ids, dtype=np.int64))


def _load_checkpoint(path: Path):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        keys = z.files
        p_safe = z["P_safe"] if "P_safe" in keys else z["P_clean"]
        p_unsafe = z["P_unsafe"] if "P_unsafe" in keys else z["P_corrupted"]
        return {
            "pair_keys": list(z["pair_keys"]),
            "P_safe": list(p_safe),
            "P_unsafe": list(p_unsafe),
            "P_patched": list(z["P_patched"]),
            "category": list(z["category"]) if "category" in keys else [],
        }


def _pair_key_str(pair) -> str:
    return f"siuo_{pair['question_id']:04d}"


# ── Filename suffix helpers ────────────────────────────────────────────────

_ABLATION_SUFFIX = {"none": "", "random": "_random", "prefix_only": "_prefix_only"}


def _full_suffix(ablation_mode: str, patch_direction: str) -> str:
    return f"{_ABLATION_SUFFIX[ablation_mode]}_{patch_direction}"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    siuo_dir = Path(args.siuo_dir) if args.siuo_dir else _SIUO_DEFAULT_DIR
    model_short = _normalize_model_name(args.model)
    if args.output_dir is None:
        out_root = _DIAGNOSTIC_ROOT / model_short / "causal_mediation_siuo" / "outputs"
    else:
        out_root = Path(args.output_dir)
    results_dir = out_root / "results"
    artifacts_dir = out_root / "artifacts"
    plots_dir = results_dir / "plots"
    for d in (results_dir, artifacts_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    suffix = _full_suffix(args.ablation_mode, args.patch_direction)
    final_npz = artifacts_dir / f"per_pair_probs{suffix}.npz"
    ckpt_npz = artifacts_dir / f"per_pair_probs{suffix}.checkpoint.npz"
    rr_json = results_dir / f"recovery_rates{suffix}.json"
    preflight_json = results_dir / f"preflight_yes_prob_check{suffix}.json"

    print(f"Loading SIUO entries from {siuo_dir}/{args.siuo_json} ...")
    entries = _load_siuo_entries(siuo_dir, args.siuo_json)
    print(f"  Loaded {len(entries)} entries.")

    pairs = _build_pairs(entries, args.seed,
                         patch_direction=args.patch_direction,
                         ablation_mode=args.ablation_mode)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"Built {len(pairs)} pairs (patch_direction={args.patch_direction}, "
          f"ablation_mode={args.ablation_mode}).")
    if not pairs:
        raise RuntimeError("No usable SIUO pairs.")

    # ── Resume from checkpoint or final, if present ─────────────────────
    completed_keys: set = set()
    pair_keys: list = []
    p_safe_list, p_unsafe_list, p_patched_list = [], [], []
    cat_list = []

    for resume_path in (final_npz, ckpt_npz):
        loaded = _load_checkpoint(resume_path)
        if loaded:
            print(f"Resuming from {resume_path}")
            pair_keys = list(loaded["pair_keys"])
            p_safe_list = list(loaded["P_safe"])
            p_unsafe_list = list(loaded["P_unsafe"])
            p_patched_list = list(loaded["P_patched"])
            cat_list = list(loaded["category"])
            completed_keys = set(pair_keys)
            break

    # ── Load model ───────────────────────────────────────────────────────
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    print(f"Loading wrapper for {args.model} ...")
    wrapper = create_wrapper(args.model,
                             torch_dtype=dtype_map[args.torch_dtype]).load()
    dispatch = get_dispatch(wrapper)
    verify_layout(wrapper, dispatch)
    yes_ids = yes_token_ids(wrapper)
    print(f"Yes-token id set ({len(yes_ids)} ids): {yes_ids}")
    print(f"Model: {dispatch.n_layers} layers × {len(COMPONENTS)} components "
          f"= {dispatch.n_layers * len(COMPONENTS)} cells per pair")
    print(f"Prompt: {args.prompt!r}")

    # ── Preflight ───────────────────────────────────────────────────────
    todo_pairs = [p for p in pairs if _pair_key_str(p) not in completed_keys]
    if not preflight_json.exists() and len(todo_pairs) > 0:
        rng = random.Random(args.seed + 1)
        pre_n = min(args.preflight_n, len(todo_pairs))
        sample = rng.sample(todo_pairs, pre_n)
        print(f"Preflight: |P_safe - P_unsafe| on {pre_n} pairs ...")
        gaps = []
        per_pair_pre = []
        for p in sample:
            ps, _ = compute_yes_prob(
                wrapper, p["image_pil"],
                _format_text(args.prompt, p["safe_text"]), yes_ids)
            pu, _ = compute_yes_prob(
                wrapper, p["image_pil"],
                _format_text(args.prompt, p["unsafe_text"]), yes_ids)
            gap = pu - ps
            gaps.append(gap)
            per_pair_pre.append({
                "question_id": p["question_id"],
                "P_safe": ps, "P_unsafe": pu, "gap": gap,
            })
        median_gap = float(np.median(np.abs(gaps)))
        save_json({
            "model": args.model,
            "patch_direction": args.patch_direction,
            "ablation_mode": args.ablation_mode,
            "prompt": args.prompt,
            "n_used": pre_n,
            "median_gap_abs": median_gap,
            "mean_gap_signed": float(np.mean(gaps)),
            "preflight_threshold": args.preflight_threshold,
            "warning": median_gap < args.preflight_threshold,
            "per_pair": per_pair_pre,
        }, preflight_json)
        if median_gap < args.preflight_threshold:
            print(f"  *** WARNING *** median |gap| {median_gap:.4f} < threshold "
                  f"{args.preflight_threshold}; continuing anyway.",
                  file=sys.stderr)
        else:
            print(f"  Preflight OK (median |gap| = {median_gap:.4f})")

    # ── Main sweep ──────────────────────────────────────────────────────
    print(f"Running mediation sweep on {len(todo_pairs)} pairs "
          f"({len(completed_keys)} already done)...")
    t_start = time.time()
    for i, pair in enumerate(todo_pairs):
        key = _pair_key_str(pair)
        t_pair = time.time()
        try:
            ps, pu, pp = _run_one_pair(
                wrapper, dispatch, pair, yes_ids, args.prompt,
                args.patch_direction)
        except Exception as e:
            print(f"  [skip {key}] {type(e).__name__}: {e}")
            continue
        pair_keys.append(key)
        p_safe_list.append(ps)
        p_unsafe_list.append(pu)
        p_patched_list.append(pp)
        cat_list.append(pair["category"])
        elapsed = time.time() - t_pair
        rate = (i + 1) / max(1.0, time.time() - t_start)
        eta_s = (len(todo_pairs) - i - 1) / max(rate, 1e-6)
        print(f"  [{i + 1}/{len(todo_pairs)}] {key}  "
              f"P_safe={ps:.3f} P_unsafe={pu:.3f}  "
              f"({elapsed:.1f}s/pair, ETA {eta_s / 60:.1f}m)")
        if (i + 1) % args.checkpoint_every == 0:
            _save_checkpoint(ckpt_npz, pair_keys, p_safe_list,
                             p_unsafe_list, p_patched_list,
                             cat_list, yes_ids)

    # ── Final write ─────────────────────────────────────────────────────
    if not pair_keys:
        print("No completed pairs — nothing to write.")
        return

    p_safe_arr = np.asarray(p_safe_list, dtype=np.float32)
    p_unsafe_arr = np.asarray(p_unsafe_list, dtype=np.float32)
    p_patched_arr = np.asarray(p_patched_list, dtype=np.float32)
    cat_arr = np.array(cat_list, dtype=object)

    _save_checkpoint(final_npz, pair_keys, p_safe_arr, p_unsafe_arr,
                     p_patched_arr, cat_arr, yes_ids)
    if ckpt_npz.exists():
        ckpt_npz.unlink()

    aggregate, n_kept = _aggregate(
        p_safe_arr, p_unsafe_arr, p_patched_arr,
        eps=args.rr_denominator_eps, n_boot=args.n_boot, seed=args.seed,
        patch_direction=args.patch_direction)

    # tier_label is a generic experiment label so plot_causal_mediation.py
    # formats SIUO and MSSBench filenames + titles uniformly.
    label_parts = []
    if args.ablation_mode != "none":
        label_parts.append(args.ablation_mode)
    label_parts.append(args.patch_direction)
    rr_label = "_".join(label_parts) if args.ablation_mode != "none" \
        else args.patch_direction

    rr_summary = {
        "model": args.model,
        "model_short": model_short,
        "dataset": "siuo",
        "patch_direction": args.patch_direction,
        "ablation_mode": args.ablation_mode,
        "prompt": args.prompt,
        "tier_label": rr_label,
        "n_pairs": int(len(pair_keys)),
        "n_pairs_kept_after_eps": n_kept,
        "rr_denominator_eps": args.rr_denominator_eps,
        "yes_token_ids": yes_ids,
        "components": list(COMPONENTS),
        "n_layers": dispatch.n_layers,
        "per_layer": aggregate,
    }
    if preflight_json.exists():
        with open(preflight_json) as f:
            pre = json.load(f)
        rr_summary["preflight"] = {
            "median_gap_abs": pre.get("median_gap_abs"),
            "n_used": pre.get("n_used"),
            "warning": pre.get("warning"),
        }
    save_json(rr_summary, rr_json)
    print(f"Wrote aggregate RR → {rr_json}")
    print(f"Wrote per-pair → {final_npz}")


if __name__ == "__main__":
    main()
