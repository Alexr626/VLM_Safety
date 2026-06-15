#!/usr/bin/env python3
"""Extract VL (image+text) activations for hallucination benchmarks."""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import load_benchmark, load_image_for_sample, DATASET_DATA_DIRS, ALL_BENCHMARKS
from src.model import create_wrapper, _normalize_model_name
from src.extraction import ActivationCache, get_last_token_activations, cleanup_gpu, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--dataset", default="pope", choices=ALL_BENCHMARKS)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip_extraction", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    dataset_dir = DATASET_DATA_DIRS.get(args.dataset, args.dataset)
    act_dir = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "data" / dataset_dir / model_name / "activations")
    act_dir.mkdir(parents=True, exist_ok=True)

    samples = load_benchmark(args.dataset, limit=args.limit)

    meta_path = act_dir / "sample_metadata.json"
    merged = {}
    if meta_path.exists():
        with open(meta_path) as f:
            for r in json.load(f):
                merged[r["id"]] = r
    for s in samples:
        merged[s["id"]] = {
            "id": s["id"], "label": s["label"], "category": s.get("category"),
            "task": s.get("task"),
        }
    save_json(list(merged.values()), str(meta_path))

    cache = ActivationCache(str(act_dir))
    wrapper = create_wrapper(args.model).load()
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

    print(f"Done -> {act_dir}/")


if __name__ == "__main__":
    main()
