#!/usr/bin/env python3
"""
Extract TT (Text-only Counterpart) Activations
================================================
ShiftDC-specific step. For each sample, replaces the image with its caption
and runs a text-only forward pass to obtain the "text-only counterpart" activations.

  ttt = "Image description: {caption}\\n\\n{original_text}"

The modality-induced shift is then: m^l = x_vl^l - x_tt^l per sample.
TT activations are cached alongside VL activations in the same directory.

Prerequisites
-------------
  1. extract_vl.py             — provides sample list via sample_metadata.json
  2. generate_captions.py      --dataset {dataset}

Outputs (under outputs/{model_name}/activations/{dataset_name}/)
-------
  {sample_id}_tt.npz   — same format as _vl.npz

Usage
-----
  python extraction/extract_tt.py
  python extraction/extract_tt.py --dataset holisafe --skip_extraction
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
from src.model import VLMWrapper
from src.extraction import ActivationCache, get_last_token_activations, cleanup_gpu


def load_samples(dataset, cache_dir, limit):
    if dataset == "holisafe":
        entries, images_base = load_holisafe(cache_dir=cache_dir)
        sss, ssu = filter_subsets(entries, images_base)
        if limit:
            sss, ssu = sss[:limit], ssu[:limit]
        return sss + ssu
    raise ValueError(f"Unknown dataset: {dataset}")


def build_tt_prompt(text, caption):
    return f"Image description: {caption}\n\n{text}" if caption else text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--dataset", default="holisafe")
    p.add_argument("--output_dir", default=str(_PROJECT_ROOT / "outputs"))
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip_extraction", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    out_base = Path(args.output_dir) / model_name
    act_dir = out_base / "activations" / args.dataset
    captions_path = out_base / "captions" / f"{args.dataset}.json"

    if not captions_path.exists():
        raise FileNotFoundError(
            f"Captions not found: {captions_path}\n"
            f"Run first: python extraction/generate_captions.py --dataset {args.dataset}"
        )

    samples = load_samples(args.dataset, args.cache_dir, args.limit)
    with open(captions_path) as f:
        captions = json.load(f)

    act_dir.mkdir(parents=True, exist_ok=True)
    cache = ActivationCache(str(act_dir))
    wrapper = VLMWrapper(args.model).load()

    todo = [s for s in samples if not (args.skip_extraction and cache.exists(s["id"], "tt"))]
    print(f"Extracting TT activations: {len(todo)}/{len(samples)} samples")

    for sample in tqdm(todo):
        tt_text = build_tt_prompt(sample["text"], captions.get(str(sample["id"]), ""))
        try:
            hidden, _, _ = wrapper.forward_text(tt_text)
            cache.save(sample["id"], get_last_token_activations(hidden), suffix="tt")
            del hidden
        except Exception as e:
            print(f"Warning: {sample['id']} — {e}")
        cleanup_gpu()

    print(f"Done → {act_dir}/")


if __name__ == "__main__":
    main()
