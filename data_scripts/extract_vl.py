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

from src.dataset import load_holisafe, filter_subsets, load_image_for_sample, inspect_schema
from src.model import VLMWrapper
from src.extraction import ActivationCache, get_last_token_activations, cleanup_gpu, save_json


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
                   help="Direct output directory. Defaults to data/activations/{dataset}/")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--skip_extraction", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    act_dir = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "data" / "holisafe-bench" / "activations" / model_name)
    act_dir.mkdir(parents=True, exist_ok=True)

    if args.inspect:
        entries, _ = load_holisafe(cache_dir=args.cache_dir)
        inspect_schema(entries)
        sys.exit(0)

    samples = load_samples(args.dataset, args.cache_dir, args.limit)
    save_json(
        [{"id": s["id"], "label": s["label"], "category": s["category"]} for s in samples],
        str(act_dir / "sample_metadata.json"),
    )

    cache = ActivationCache(str(act_dir))
    wrapper = VLMWrapper(args.model).load()

    todo = [s for s in samples if not (args.skip_extraction and cache.exists(s["id"], "vl"))]
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
