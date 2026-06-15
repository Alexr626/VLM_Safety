#!/usr/bin/env python3
"""Extract TT (caption+text) activations for hallucination benchmarks."""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import load_benchmark, DATASET_DATA_DIRS, ALL_BENCHMARKS
from src.model import create_wrapper, _normalize_model_name
from src.extraction import ActivationCache, get_last_token_activations, cleanup_gpu


def build_tt_prompt(text, caption):
    return f"Image description: {caption}\n\n{text}" if caption else text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--dataset", default="pope", choices=ALL_BENCHMARKS)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--captions_dir", default=str(_PROJECT_ROOT / "data" / "captions"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip_extraction", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    dataset_dir = DATASET_DATA_DIRS.get(args.dataset, args.dataset)
    act_dir = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "data" / dataset_dir / model_name / "activations")
    captions_path = Path(args.captions_dir) / f"{args.dataset}.json"

    if not captions_path.exists():
        raise FileNotFoundError(
            f"Captions not found: {captions_path}\n"
            f"Run: python data_scripts/generate_captions.py --dataset {args.dataset}"
        )

    samples = load_benchmark(args.dataset, limit=args.limit)
    with open(captions_path) as f:
        captions = json.load(f)

    act_dir.mkdir(parents=True, exist_ok=True)
    cache = ActivationCache(str(act_dir))
    wrapper = create_wrapper(args.model).load()
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

    print(f"Done -> {act_dir}/")


if __name__ == "__main__":
    main()
