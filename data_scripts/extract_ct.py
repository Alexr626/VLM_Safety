#!/usr/bin/env python3
"""
Extract CT (Cohesive Text) Activations
========================================
For each sample, runs a text-only forward pass using the cohesive text
(fused caption + query) to obtain activations for the cohesive representation.

Prerequisites
-------------
  1. extract_vl.py             -- provides sample list via sample_metadata.json
  2. generate_cohesive_text.py -- produces data/captions/holisafe_cohesive.json

Outputs (under data/holisafe-bench/activations/{model}/)
-------
  sample_{id}_ct.npz  -- same format as _vl.npz and _tt.npz

Usage
-----
  python data_scripts/extract_ct.py
  python data_scripts/extract_ct.py --skip_if_exists
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import load_holisafe, filter_subsets
from src.model import create_wrapper, _normalize_model_name
from src.extraction import ActivationCache, get_last_token_activations, cleanup_gpu


def load_samples(dataset, cache_dir, limit):
    if dataset == "holisafe":
        entries, images_base = load_holisafe(cache_dir=cache_dir)
        sss, ssu = filter_subsets(entries, images_base)
        if limit:
            sss, ssu = sss[:limit], ssu[:limit]
        return sss + ssu
    raise ValueError(f"Unknown dataset: {dataset}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--dataset", default="holisafe")
    p.add_argument("--output_dir", default=None,
                   help="Direct output directory. Defaults to data/holisafe-bench/activations/{model}/")
    p.add_argument("--cohesive_path", default=None,
                   help="Path to cohesive text JSON. Defaults to data/captions/holisafe_cohesive.json")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip_if_exists", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    act_dir = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "data" / "holisafe-bench" / model_name / "activations")
    cohesive_path = Path(args.cohesive_path) if args.cohesive_path else (
        _PROJECT_ROOT / "data" / "captions" / "holisafe_cohesive.json")

    if not cohesive_path.exists():
        raise FileNotFoundError(
            f"Cohesive text not found: {cohesive_path}\n"
            "Run first: python data_scripts/generate_cohesive_text.py"
        )

    samples = load_samples(args.dataset, args.cache_dir, args.limit)
    with open(cohesive_path) as f:
        cohesive = json.load(f)

    act_dir.mkdir(parents=True, exist_ok=True)
    cache = ActivationCache(str(act_dir))
    wrapper = create_wrapper(args.model).load()

    todo = [s for s in samples if not (args.skip_if_exists and cache.exists(s["id"], "ct"))]
    print(f"Extracting CT activations: {len(todo)}/{len(samples)} samples")

    for sample in tqdm(todo):
        ct_text = cohesive.get(str(sample["id"]), "")
        if not ct_text:
            ct_text = sample["text"]  # fallback
        try:
            hidden, _, _ = wrapper.forward_text(ct_text)
            cache.save(sample["id"], get_last_token_activations(hidden), suffix="ct")
            del hidden
        except Exception as e:
            print(f"Warning: {sample['id']} -- {e}")
        cleanup_gpu()

    print(f"Done -> {act_dir}/")


if __name__ == "__main__":
    main()
