#!/usr/bin/env python3
"""
Generate Image Captions
========================
Runs the VLM to produce text descriptions of images for a dataset.
Captions are a prerequisite for TT extraction and safety direction computation.

Supported datasets
------------------
  holisafe       — main evaluation dataset (SSS/SSU samples)
  mm-safetybench — unsafe reference (ShiftDC Appendix A.3, Scenarios 01-07 & 09)
  llava-instruct — safe reference (ShiftDC Appendix A.3)

Outputs (under data/captions/)
-------
  {dataset_name}.json   — {sample_id: caption_str}

Usage
-----
  python data_scripts/generate_captions.py --dataset holisafe
  python data_scripts/generate_captions.py --dataset mm-safetybench
  python data_scripts/generate_captions.py --dataset holisafe --skip_if_exists
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import (
    load_holisafe, filter_subsets, load_image_for_sample,
    REFERENCE_REGISTRY,
)
from src.model import create_wrapper
from src.extraction import cleanup_gpu


def load_samples(dataset, cache_dir, limit, ref_samples, ref_seed):
    if dataset == "holisafe":
        entries, images_base = load_holisafe(cache_dir=cache_dir)
        sss, ssu = filter_subsets(entries, images_base)
        samples = sss + ssu
    elif dataset in REFERENCE_REGISTRY:
        loader = REFERENCE_REGISTRY[dataset]["loader"]
        samples = loader(n_samples=ref_samples, seed=ref_seed)
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset}'. "
            f"Available reference datasets: {list(REFERENCE_REGISTRY)}. "
            f"To add a new dataset, register a loader in REFERENCE_REGISTRY (src/dataset.py)."
        )
    return samples[:limit] if limit else samples


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--dataset", default="holisafe",
                   help="Dataset to caption. 'holisafe' or any key in REFERENCE_REGISTRY.")
    p.add_argument("--output_dir", default=str(_PROJECT_ROOT / "data" / "captions"))
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ref_samples", type=int, default=160)
    p.add_argument("--ref_seed", type=int, default=42)
    p.add_argument("--skip_if_exists", action="store_true")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output_dir) / f"{args.dataset}.json"

    if args.skip_if_exists and out_path.exists():
        print(f"Skipping — {out_path} already exists.")
        return

    samples = load_samples(args.dataset, args.cache_dir, args.limit,
                           args.ref_samples, args.ref_seed)
    wrapper = create_wrapper(args.model).load()

    captions = {}
    bs = args.batch_size
    for i in tqdm(range(0, len(samples), bs), desc=f"Captioning {args.dataset}"):
        batch = samples[i:i + bs]
        images = [load_image_for_sample(s) for s in batch]
        valid_pairs = [(s, img) for s, img in zip(batch, images) if img is not None]
        for s, img in zip(batch, images):
            if img is None:
                captions[str(s["id"])] = ""
        if not valid_pairs:
            continue
        try:
            batch_caps = wrapper.generate_captions_batch(
                [img for _, img in valid_pairs],
                max_new_tokens=args.max_new_tokens,
            )
            for (s, _), cap in zip(valid_pairs, batch_caps):
                captions[str(s["id"])] = cap
        except Exception as e:
            print(f"Warning: batch {i//bs} — {e}")
            for s, _ in valid_pairs:
                captions[str(s["id"])] = ""
        cleanup_gpu()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
