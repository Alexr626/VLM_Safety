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

Outputs (under data/activations/{dataset_name}/ by default)
-------
  {sample_id}_tt.npz   — same format as _vl.npz

Usage
-----
  python data_scripts/extract_tt.py
  python data_scripts/extract_tt.py --dataset holisafe --skip_extraction
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
    load_holisafe, filter_subsets, filter_reference_subsets,
    load_mssbench, DATASET_DATA_DIRS,
)
from src.model import create_wrapper, _normalize_model_name
from src.extraction import ActivationCache, get_last_token_activations, cleanup_gpu


def load_samples(dataset, cache_dir, limit,
                 holisafe_subsets=None, holisafe_eval_only=False,
                 mssbench_split: str = "all"):
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
        for s in samples:
            s["label"] = s.get("subset_type") or s.get("label", "OTHER")
        return samples[:limit] if limit else samples
    if dataset == "mssbench":
        samples = load_mssbench()
        if mssbench_split != "all":
            split_path = _PROJECT_ROOT / "data" / "mssbench" / "train_eval_split.json"
            if not split_path.exists():
                raise FileNotFoundError(
                    f"--mssbench_split={mssbench_split} requires {split_path}. "
                    "Run: python -m src.dataset --mssbench_split"
                )
            with open(split_path) as f:
                split = json.load(f)
            wanted = set(split[f"{mssbench_split}_sample_ids"])
            samples = [s for s in samples if s["id"] in wanted]
        return samples[:limit] if limit else samples
    raise ValueError(f"Unknown dataset: {dataset}")


def build_tt_prompt(text, caption):
    return f"Image description: {caption}\n\n{text}" if caption else text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--dataset", default="holisafe")
    p.add_argument("--output_dir", default=None,
                   help="Direct output directory. Defaults to data/activations/{dataset}/")
    p.add_argument("--captions_dir", default=str(_PROJECT_ROOT / "data" / "captions"))
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip_extraction", action="store_true")
    p.add_argument("--holisafe_eval_only", action="store_true",
                   help="When --dataset=holisafe, restrict to samples in the "
                        "eval splits of train_eval_split.json.")
    p.add_argument("--holisafe_subsets", nargs="+", default=None,
                   choices=["SSS", "SSU", "SUU", "USU", "UUU"],
                   help="When --dataset=holisafe, restrict to samples whose "
                        "raw HoliSafe `type` matches one of these.")
    p.add_argument("--mssbench_split", choices=["all", "train", "eval"],
                   default="all",
                   help="When --dataset=mssbench, restrict to a split. "
                        "Defaults to 'all' (extracts both train and eval).")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    dataset_dir = DATASET_DATA_DIRS.get(args.dataset, args.dataset)
    act_dir = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "data" / dataset_dir / "activations" / model_name)
    captions_path = Path(args.captions_dir) / f"{args.dataset}.json"

    if not captions_path.exists():
        raise FileNotFoundError(
            f"Captions not found: {captions_path}\n"
            f"Run first: python data_scripts/generate_captions.py --dataset {args.dataset}"
        )

    samples = load_samples(
        args.dataset, args.cache_dir, args.limit,
        holisafe_subsets=args.holisafe_subsets,
        holisafe_eval_only=args.holisafe_eval_only,
        mssbench_split=args.mssbench_split,
    )
    with open(captions_path) as f:
        captions = json.load(f)

    missing = [s["id"] for s in samples if str(s["id"]) not in captions]
    if missing:
        raise RuntimeError(
            f"{len(missing)} samples missing captions in {captions_path} "
            f"(e.g. {missing[:5]}). Run generate_captions.py with the same "
            "--holisafe_subsets / --holisafe_eval_only flags first."
        )

    act_dir.mkdir(parents=True, exist_ok=True)
    cache = ActivationCache(str(act_dir))
    wrapper = create_wrapper(args.model).load()

    # Always skip already-cached samples (see note in extract_vl.py).
    todo = [s for s in samples if not cache.exists(s["id"], "tt")]
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
