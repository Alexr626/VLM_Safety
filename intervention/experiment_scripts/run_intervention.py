#!/usr/bin/env python3
"""
Intervention Inference: ShiftDC vs Spherical ShiftDC
======================================================
For each SSU (and optionally SSS) sample in the eval split, run VL inference
with one of three intervention modes applied as a forward hook at a chosen layer:

  --method none        : standard VL generation (baseline)
  --method original    : subtract safety-relevant projection of modality shift
                         (original ShiftDC calibration, Zou et al. 2025)
  --method spherical   : rotate toward ShiftDC target via Slerp, preserving norm
                         (Spherical ShiftDC, proposed)

Supports all four models in the project via --model:
  llava-hf/llava-1.5-7b-hf
  Qwen/Qwen2.5-VL-7B-Instruct
  OpenGVLab/InternVL2-8B
  OpenGVLab/InternVL2_5-8B-MPO

Prerequisites
-------------
  1. VL and TT activations cached at:
       data/holisafe-bench/activations/{model}/sample_{id}_{vl|tt}.npz
  2. Safety direction vectors at:
       experiment_artifacts/{model}/vl_activation_shift/safety_direction_vectors.npz
  3. HoliSafe-Bench dataset loaded via src.dataset

Outputs (under intervention/{model}/outputs/results/)
-------
  {method}_results.json          — per-sample responses + refusal labels
  {method}_summary.json          — aggregate ASR / refusal rates + norm stats

Usage
-----
  # Baseline (no intervention) on LLaVA
  python run_intervention.py --method none --model llava-hf/llava-1.5-7b-hf

  # Spherical ShiftDC on Qwen
  python run_intervention.py --method spherical --model Qwen/Qwen2.5-VL-7B-Instruct --layer 27

  # Only SSU samples, limit to first 50
  python run_intervention.py --method spherical --labels SSU --limit 50

  # Skip if results already exist
  python run_intervention.py --method spherical --skip_if_exists
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR   = Path(__file__).resolve().parent
_INTERVENTION = _SCRIPT_DIR.parent           # intervention/
_PROJECT_ROOT = _INTERVENTION.parent         # VLM_Safety_new/
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_INTERVENTION))

from src.dataset import load_holisafe, filter_subsets, load_image_for_sample
from src.model import create_wrapper, _normalize_model_name
from src.extraction import ActivationCache, cleanup_gpu
from diagnostic_experiments.experiment_scripts.classify_responses import (
    REFUSAL_PHRASES, classify_keyword_twoaxis,
)
from intervention.steering import (
    compute_shiftdc_calibration,
    compute_spherical_shiftdc,
    calibration_stats,
)

_DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"

# ── Path resolution ────────────────────────────────────────────────────────────

def _resolve_cache_dir(model_name: str) -> Path:
    """Find the TT/VL activation cache directory for a given model."""
    candidates = [
        _PROJECT_ROOT / "data" / "holisafe-bench" / "activations" / model_name,
        _PROJECT_ROOT / "outputs" / model_name / "activations" / "holisafe",
    ]
    for p in candidates:
        if p.exists() and any(p.iterdir()):
            return p
    raise FileNotFoundError(
        "Could not locate activation cache. Expected one of:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\nRun data_scripts/extract_vl.py and extract_tt.py first."
    )


def _resolve_safety_vecs(model_name: str) -> Path:
    candidates = [
        _PROJECT_ROOT / "experiment_artifacts" / model_name /
            "vl_activation_shift" / "safety_direction_vectors.npz",
        _PROJECT_ROOT / "outputs" / model_name /
            "method2_shiftdc" / "safety_direction_vectors.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Safety direction vectors not found. Expected one of:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def _load_train_eval_split() -> set:
    """Return set of sample IDs in the eval split, or None to use all samples."""
    p = _PROJECT_ROOT / "data" / "holisafe-bench" / "train_eval_split.json"
    if not p.exists():
        return None
    with open(p) as f:
        split = json.load(f)
    return set(split.get("eval", []))


# ── Forward hook factory ───────────────────────────────────────────────────────

def make_intervention_hook(
    x_tt_layer: np.ndarray,
    s_l: np.ndarray,
    method: str,
    t: float = 1.0,
    gate_by_alignment: bool = False,
):
    """Return a forward hook that modifies the last-token hidden state in place.

    Compatible with any transformer decoder layer whose output is either:
      - a tuple (hidden_states, ...) with hidden_states: (batch, seq, d)
      - just hidden_states: (batch, seq, d)

    Only the last token position is modified.
    """
    norm_ratios = []

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        x_vl = h[0, -1, :].detach().cpu().float().numpy()  # (d,)

        if method == "original":
            x_hat = compute_shiftdc_calibration(x_vl, x_tt_layer, s_l)
        elif method == "spherical":
            x_hat = compute_spherical_shiftdc(
                x_vl, x_tt_layer, s_l, t=t,
                gate_by_alignment=gate_by_alignment,
            )
        else:
            return output  # no-op

        norm_vl  = float(np.linalg.norm(x_vl))
        norm_hat = float(np.linalg.norm(x_hat))
        norm_ratios.append(norm_hat / (norm_vl + 1e-12))

        h[0, -1, :] = torch.tensor(x_hat, dtype=h.dtype, device=h.device)
        return (h,) + output[1:] if isinstance(output, tuple) else h

    hook_fn.norm_ratios = norm_ratios
    return hook_fn


# ── Refusal detection ──────────────────────────────────────────────────────────

def is_refusal(response: str) -> bool:
    if not response:
        return False
    lower = response.lower()
    return any(p in lower for p in REFUSAL_PHRASES)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=_DEFAULT_MODEL,
                   help="HuggingFace model ID (default: llava-hf/llava-1.5-7b-hf)")
    p.add_argument("--method", default="spherical",
                   choices=["none", "original", "spherical"],
                   help="Intervention method (default: spherical)")
    p.add_argument("--layer", type=int, default=None,
                   help="Transformer layer to intervene at (default: last layer)")
    p.add_argument("--t", type=float, default=1.0,
                   help="Rotation strength for spherical method, in [0, 1]")
    p.add_argument("--gate_by_alignment", action="store_true",
                   help="Scale t by cosine alignment between m^l and s^l")
    p.add_argument("--labels", default="SSU,SSS",
                   help="Comma-separated labels to evaluate (default: SSU,SSS)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit total number of samples (for quick tests)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--cache_dir", default=None,
                   help="Override activation cache directory")
    p.add_argument("--skip_if_exists", action="store_true",
                   help="Skip if results file already exists")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)

    out_dir = _INTERVENTION / model_name / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    method_tag = args.method
    if args.method == "spherical":
        method_tag = f"spherical_t{args.t:.2f}"
        if args.gate_by_alignment:
            method_tag += "_gated"
    results_path = out_dir / f"{method_tag}_results.json"
    summary_path = out_dir / f"{method_tag}_summary.json"

    if args.skip_if_exists and results_path.exists():
        print(f"Skipping — {results_path} already exists.")
        return

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading HoliSafe-Bench ...")
    entries, images_base = load_holisafe(cache_dir=args.cache_dir)
    sss_all, ssu_all = filter_subsets(entries, images_base)

    eval_ids = _load_train_eval_split()
    if eval_ids is not None:
        print(f"  Using eval split: {len(eval_ids)} samples")
        sss = [s for s in sss_all if s["id"] in eval_ids]
        ssu = [s for s in ssu_all if s["id"] in eval_ids]
    else:
        print("  No train/eval split found — using all samples")
        sss, ssu = sss_all, ssu_all

    requested_labels = {l.strip().upper() for l in args.labels.split(",")}
    samples = []
    if "SSS" in requested_labels:
        samples += sss
    if "SSU" in requested_labels:
        samples += ssu
    if args.limit:
        samples = samples[:args.limit]
    print(f"  Samples: {sum(s['label']=='SSS' for s in samples)} SSS + "
          f"{sum(s['label']=='SSU' for s in samples)} SSU")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model} ...")
    wrapper = create_wrapper(args.model).load()

    # Resolve intervention layer: default to last layer if not specified
    layer = args.layer if args.layer is not None else wrapper.num_layers - 1
    print(f"  Intervention layer: {layer} (model has {wrapper.num_layers} layers)")

    llm_layers = wrapper.llm_layers

    # ── Load precomputed TT activations ───────────────────────────────────────
    cache_dir = Path(args.cache_dir) if args.cache_dir else _resolve_cache_dir(model_name)
    print(f"Activation cache: {cache_dir}")
    cache = ActivationCache(str(cache_dir))

    # ── Load safety direction ─────────────────────────────────────────────────
    sd_path = _resolve_safety_vecs(model_name)
    print(f"Safety directions: {sd_path}")
    safety_vecs = dict(np.load(sd_path))
    s_l = safety_vecs[f"layer_{layer}"].astype(np.float64)
    print(f"  s^l (layer {layer}) shape: {s_l.shape}")

    # ── Checkpoint-based resume ───────────────────────────────────────────────
    checkpoint_path = results_path.with_suffix(".checkpoint.json")
    results = []
    done_ids = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"  Resuming from checkpoint: {len(done_ids)} already done")

    # ── Inference loop ────────────────────────────────────────────────────────
    for sample in tqdm(samples, desc=f"Inference [{method_tag}]"):
        sid = sample["id"]
        if sid in done_ids:
            continue

        record = {
            "id":         sid,
            "label":      sample["label"],
            "category":   sample["category"],
            "method":     method_tag,
            "response":   "",
            "is_refusal": None,
            "norm_ratio": None,
        }

        tt_acts = cache.load_or_none(sid, suffix="tt")
        if tt_acts is None or layer not in tt_acts:
            print(f"  Warning: TT activations missing for sample {sid} — skipping")
            continue
        x_tt_layer = tt_acts[layer].astype(np.float64)

        hook_fn = make_intervention_hook(
            x_tt_layer, s_l,
            method=args.method,
            t=args.t,
            gate_by_alignment=args.gate_by_alignment,
        )
        hook_handle = llm_layers[layer].register_forward_hook(hook_fn)

        try:
            image = load_image_for_sample(sample)
            if image is None:
                record["response"] = "[IMAGE_NOT_AVAILABLE]"
            else:
                record["response"] = wrapper.generate_vl(
                    image, sample["text"],
                    max_new_tokens=args.max_new_tokens,
                )
        except Exception as e:
            print(f"  Warning: generation failed for sample {sid}: {e}")
            record["response"] = "[GENERATION_ERROR]"
        finally:
            hook_handle.remove()
            cleanup_gpu()

        record["is_refusal"] = is_refusal(record["response"])
        if hook_fn.norm_ratios:
            record["norm_ratio"] = float(np.mean(hook_fn.norm_ratios))

        results.append(record)
        done_ids.add(sid)

        if len(results) % 20 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Saved → {results_path}")

    summary = _compute_summary(results, method_tag)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {summary_path}")
    _print_summary(summary)


def _compute_summary(results: list, method_tag: str) -> dict:
    summary = {"method": method_tag, "total": len(results)}
    for label in ["SSS", "SSU"]:
        group = [r for r in results if r["label"] == label]
        n = len(group)
        if n == 0:
            summary[label] = {"n": 0}
            continue
        n_refusal   = sum(1 for r in group if r["is_refusal"])
        norm_ratios = [r["norm_ratio"] for r in group if r["norm_ratio"] is not None]
        summary[label] = {
            "n":               n,
            "refusal_rate":    n_refusal / n,
            "asr":             1.0 - n_refusal / n,
            "mean_norm_ratio": float(np.mean(norm_ratios)) if norm_ratios else None,
            "std_norm_ratio":  float(np.std(norm_ratios))  if norm_ratios else None,
        }
    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"n": 0, "n_refusal": 0})
    for r in results:
        if r["label"] != "SSU":
            continue
        cat = r["category"]
        cat_stats[cat]["n"] += 1
        if r["is_refusal"]:
            cat_stats[cat]["n_refusal"] += 1
    summary["SSU_per_category"] = {
        cat: {
            "n":            s["n"],
            "refusal_rate": s["n_refusal"] / s["n"],
            "asr":          1.0 - s["n_refusal"] / s["n"],
        }
        for cat, s in sorted(cat_stats.items())
    }
    return summary


def _print_summary(summary: dict):
    print(f"\n{'='*60}")
    print(f"  Method: {summary['method']}")
    print(f"  Total samples: {summary['total']}")
    print(f"{'='*60}")
    for label in ["SSS", "SSU"]:
        g = summary.get(label, {})
        if not g or g.get("n", 0) == 0:
            continue
        nr_str = (f"  norm_ratio={g['mean_norm_ratio']:.4f}±{g['std_norm_ratio']:.4f}"
                  if g.get("mean_norm_ratio") is not None else "")
        print(f"  {label} (n={g['n']}): ASR={g['asr']:.1%}  "
              f"refusal={g['refusal_rate']:.1%}{nr_str}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
