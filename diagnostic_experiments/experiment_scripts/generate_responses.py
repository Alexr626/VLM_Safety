#!/usr/bin/env python3
"""
Generate Behavioral Ground Truth Responses
===========================================
For each HoliSafe SSS + SSU sample, generate model outputs under three
input conditions: VL (image+text), TT (caption+text), CT (cohesive text).

Outputs
-------
  diagnostic_experiments/{model}/behavioral_ground_truth/outputs/results/
    holisafe_responses.json
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_DIAGNOSTIC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DIAGNOSTIC_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import load_holisafe, filter_subsets, load_image_for_sample
from src.model import create_wrapper, _normalize_model_name
from src.extraction import cleanup_gpu

_EXPERIMENT_NAME = "behavioral_ground_truth"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--skip_if_exists", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    _DATA = _PROJECT_ROOT / "data"
    experiment_dir = _DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME
    results_dir = experiment_dir / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "holisafe_responses.json"

    if args.skip_if_exists and out_path.exists():
        print(f"Skipping -- {out_path} already exists.")
        return

    entries, images_base = load_holisafe(cache_dir=args.cache_dir)
    sss, ssu = filter_subsets(entries, images_base)
    samples = sss + ssu
    if args.limit:
        samples = samples[:args.limit]

    captions_path = _DATA / "captions" / "holisafe.json"
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions not found: {captions_path}")
    with open(captions_path) as f:
        captions = json.load(f)

    cohesive_path = _DATA / "captions" / "holisafe_cohesive.json"
    cohesive = {}
    if cohesive_path.exists():
        with open(cohesive_path) as f:
            cohesive = json.load(f)
    else:
        print(f"  Warning: cohesive text not found at {cohesive_path}; CT skipped.")

    checkpoint_path = out_path.with_suffix(".checkpoint.json")
    results = []
    done_ids = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"  Resuming from checkpoint: {len(done_ids)} already done")

    wrapper = create_wrapper(args.model).load()
    mnt = args.max_new_tokens

    for sample in tqdm(samples, desc="Generating responses"):
        sid = sample["id"]
        if sid in done_ids:
            continue

        record = {
            "id": sid, "label": sample["label"], "category": sample["category"],
            "text": sample["text"],
            "response_vl": "", "response_tt": "", "response_ct": "",
        }

        try:
            image = load_image_for_sample(sample)
            if image is not None:
                record["response_vl"] = wrapper.generate_vl(image, sample["text"], max_new_tokens=mnt)
            else:
                record["response_vl"] = "[IMAGE_NOT_AVAILABLE]"
        except Exception as e:
            print(f"  Warning: VL {sid} -- {e}")
        cleanup_gpu()

        try:
            caption = captions.get(str(sid), "")
            tt_text = f"Image description: {caption}\n\n{sample['text']}" if caption else sample["text"]
            record["response_tt"] = wrapper.generate_text(tt_text, max_new_tokens=mnt)
        except Exception as e:
            print(f"  Warning: TT {sid} -- {e}")
        cleanup_gpu()

        ct_text = cohesive.get(str(sid), "")
        if ct_text:
            try:
                record["response_ct"] = wrapper.generate_text(ct_text, max_new_tokens=mnt)
            except Exception as e:
                print(f"  Warning: CT {sid} -- {e}")
            cleanup_gpu()

        results.append(record)
        done_ids.add(sid)

        if len(results) % 20 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Done -> {out_path} ({len(results)} samples)")


if __name__ == "__main__":
    main()
