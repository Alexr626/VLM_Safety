#!/usr/bin/env python3
"""
Extract VL (Vision-Language) Activations
  image+text → activations/*_vl.npz
=========================================
Runs image+text forward passes on a dataset and caches the last-token
hidden state at every transformer layer for each sample.

Outputs (under data/activations/{dataset_name}/ by default)
-------
  {sample_id}_vl.npz    — {layer_idx: np.ndarray (hidden_dim,)} per sample
  sample_metadata.json  — id / label / category per sample

Usage
-----
  python data_scripts/extract_vl.py
  python data_scripts/extract_vl.py --dataset holisafe --limit 50
  python data_scripts/extract_vl.py --skip_extraction   # resume interrupted run
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import json

from src.dataset import (
    load_holisafe, filter_subsets, filter_reference_subsets,
    load_image_for_sample, inspect_schema,
)
from src.model import create_wrapper, _normalize_model_name
from src.extraction import ActivationCache, get_last_token_activations, cleanup_gpu, save_json


def load_samples(dataset, cache_dir, limit,
                 holisafe_subsets=None, holisafe_eval_only=False):
    if dataset == "holisafe":
        entries, images_base = load_holisafe(cache_dir=cache_dir)
        if holisafe_subsets is None and not holisafe_eval_only:
            sss, ssu = filter_subsets(entries, images_base)
            if limit:
                sss, ssu = sss[:limit], ssu[:limit]
            return sss + ssu
        buckets = filter_reference_subsets(entries, images_base)
        wanted = (set(holisafe_subsets) if holisafe_subsets
                  else set(buckets.keys()))
        samples = [s for k, lst in buckets.items() if k in wanted
                   for s in lst]
        if holisafe_eval_only:
            split_path = (_PROJECT_ROOT / "data" / "holisafe-bench"
                          / "train_eval_split.json")
            if not split_path.exists():
                raise FileNotFoundError(
                    "--holisafe_eval_only requires "
                    f"{split_path}. Run: python -m src.dataset"
                )
            with open(split_path) as f:
                split = json.load(f)
            eval_ids = set()
            for k in ("sss", "ssu", "suu", "usu", "uuu"):
                eval_ids.update(split.get(f"{k}_eval_ids", []))
            samples = [s for s in samples if s["id"] in eval_ids]
        # Each sample built via filter_reference_subsets has label="OTHER";
        # promote it to the raw subset_type so downstream consumers (and
        # sample_metadata.json) carry useful labels.
        for s in samples:
            s["label"] = s.get("subset_type") or s.get("label", "OTHER")
        return samples[:limit] if limit else samples
    raise ValueError(f"Unknown dataset: {dataset}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--dataset", default="holisafe")
    p.add_argument("--output_dir", default=None,
                   help="Direct output directory. Defaults to data/activations/{dataset}/")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--skip_extraction", action="store_true")
    p.add_argument("--holisafe_eval_only", action="store_true",
                   help="When --dataset=holisafe, restrict to samples in the "
                        "eval splits of train_eval_split.json.")
    p.add_argument("--holisafe_subsets", nargs="+", default=None,
                   choices=["SSS", "SSU", "SUU", "USU", "UUU"],
                   help="When --dataset=holisafe, restrict to samples whose "
                        "raw HoliSafe `type` matches one of these.")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    act_dir = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "data" / "holisafe-bench" / "activations" / model_name)
    act_dir.mkdir(parents=True, exist_ok=True)

    if args.inspect:
        entries, _ = load_holisafe(cache_dir=args.cache_dir)
        inspect_schema(entries)
        sys.exit(0)

    samples = load_samples(
        args.dataset, args.cache_dir, args.limit,
        holisafe_subsets=args.holisafe_subsets,
        holisafe_eval_only=args.holisafe_eval_only,
    )

    # Merge sample_metadata.json by id so prior runs (e.g. SSS+SSU) are
    # preserved when this run extends with new subsets.
    meta_path = act_dir / "sample_metadata.json"
    merged: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            for r in json.load(f):
                merged[r["id"]] = r
    for s in samples:
        merged[s["id"]] = {
            "id": s["id"], "label": s["label"], "category": s["category"],
        }
    save_json(list(merged.values()), str(meta_path))

    cache = ActivationCache(str(act_dir))
    wrapper = create_wrapper(args.model).load()

    # Always skip samples whose VL activations are already cached. Prior
    # behavior was to skip only when --skip_extraction was set; that flag
    # is now redundant (kept for CLI backward compat) since the cache check
    # is cheap and avoids wasted re-extraction when extending to new subsets.
    todo = [s for s in samples if not cache.exists(s["id"], "vl")]
    print(f"Extracting VL activations: {len(todo)}/{len(samples)} samples")

    for sample in tqdm(todo):
        image = load_image_for_sample(sample)
        if image is None:
            continue
        try:
            hidden, _, _ = wrapper.forward_vl(image, sample["text"])
            cache.save(sample["id"], get_last_token_activations(hidden), suffix="vl")
            del hidden
        except Exception as e:
            print(f"Warning: {sample['id']} — {e}")
        cleanup_gpu()

    print(f"Done → {act_dir}/")


if __name__ == "__main__":
    main()
