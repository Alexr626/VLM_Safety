#!/usr/bin/env python3
"""
Extract Reference Dataset Activations
=======================================
Runs text-only (TT-style) forward passes on the two reference datasets used
by ShiftDC to compute the safety direction vector s^l:

  s^l = mean(LLaVA-Instruct activations^l) - mean(MM-SafetyBench activations^l)

Both datasets are as specified in ShiftDC Appendix A.3:
  - MM-SafetyBench (unsafe): Scenarios 01-07 & 09, 160 samples
  - LLaVA-Instruct-80k (safe): 160 samples

Results are also read by followup experiments B and C (subspace analysis).

Prerequisites
-------------
  1. generate_captions.py --dataset mm-safetybench
  2. generate_captions.py --dataset llava-instruct

Outputs (under data/reference/ by default)
-------
  {ref_name}/
    activation_matrices.npz  — keys "{role}_layer_{l}", shape (N, hidden_dim)
    metadata.json

Usage
-----
  python data_scripts/extract_ref_activations.py
  python data_scripts/extract_ref_activations.py --ref_samples 160 --skip_if_exists
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import REFERENCE_REGISTRY
from src.model import VLMWrapper
from src.extraction import get_last_token_activations, cleanup_gpu, save_json, save_npz


def build_tt_prompt(text, caption):
    return f"Image description: {caption}\n\n{text}" if caption else text


def load_captions(path, dataset_name):
    if not path.exists():
        raise FileNotFoundError(
            f"Captions not found: {path}\n"
            f"Run first: python data_scripts/generate_captions.py --dataset {dataset_name}"
        )
    with open(path) as f:
        return json.load(f)


def extract_per_layer(samples, captions, wrapper, desc):
    """Run TT forward passes, return {layer_idx: [vec, ...]}."""
    per_layer = {}
    for sample in tqdm(samples, desc=desc):
        tt_text = build_tt_prompt(sample["text"], captions.get(str(sample["id"]), ""))
        try:
            hidden, _, _ = wrapper.forward_text(tt_text)
            for l, vec in get_last_token_activations(hidden).items():
                per_layer.setdefault(l, []).append(vec)
            del hidden
        except Exception as e:
            print(f"Warning: {sample['id']} — {e}")
        cleanup_gpu()
    return per_layer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--output_dir", default=None,
                   help="Direct output directory. Defaults to data/catqa-contrastive/activations/{model}/")
    p.add_argument("--captions_dir", default=str(_PROJECT_ROOT / "data" / "captions"))
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--ref_samples", type=int, default=160)
    p.add_argument("--ref_seed", type=int, default=42)
    p.add_argument("--skip_if_exists", action="store_true")
    p.add_argument("--safe_ref",   default="llava-instruct",
                   help="Name of the safe reference dataset (must be in REFERENCE_REGISTRY)")
    p.add_argument("--unsafe_ref", default="mm-safetybench",
                   help="Name of the unsafe reference dataset (must be in REFERENCE_REGISTRY)")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    ref_base = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "data" / "catqa-contrastive" / "activations" / model_name)
    captions_dir = Path(args.captions_dir)

    ref_configs = []
    for name in (args.unsafe_ref, args.safe_ref):
        if name not in REFERENCE_REGISTRY:
            raise ValueError(
                f"Unknown reference dataset '{name}'. "
                f"Available: {list(REFERENCE_REGISTRY)}. "
                f"Add new datasets to REFERENCE_REGISTRY in src/dataset.py."
            )
        entry = REFERENCE_REGISTRY[name]
        samples = entry["loader"](n_samples=args.ref_samples, seed=args.ref_seed)
        ref_configs.append((name, entry["role"], entry.get("text_only", False), samples))

    wrapper = VLMWrapper(args.model).load()

    for name, role, text_only, samples in ref_configs:
        ref_dir = ref_base / name
        out_npz = ref_dir / "activation_matrices.npz"

        if args.skip_if_exists and out_npz.exists():
            print(f"Skipping '{name}' — already exists.")
            continue

        captions = {} if text_only else load_captions(captions_dir / f"{name}.json", name)
        per_layer = extract_per_layer(samples, captions, wrapper, desc=name)

        matrices = {f"{role}_layer_{l}": np.stack(vecs)
                    for l, vecs in per_layer.items() if vecs}

        ref_dir.mkdir(parents=True, exist_ok=True)
        save_npz(matrices, str(out_npz))
        save_json(
            {"source": name, "role": role, "n_samples": len(samples),
             "seed": args.ref_seed, "sample_ids": [s["id"] for s in samples]},
            str(ref_dir / "metadata.json"),
        )
        print(f"Done '{name}' → {out_npz}")


if __name__ == "__main__":
    main()
