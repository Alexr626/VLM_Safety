#!/usr/bin/env python3
"""
CatQA Behavioral Baseline
==========================
Measure each model's refusal/harmful-content rate on explicitly harmful
(and harmless) CatQA prompts using text-only generation. Classify with the
same two-axis LLM classifier used for HoliSafe.

Purpose: determine whether low HoliSafe SSU refusal reflects weak model
safety training (high CatQA ASR too) or weak SSU stimuli (CatQA ASR much
lower than SSU ASR).

Outputs
-------
  diagnostic_experiments/{model}/behavioral_ground_truth/outputs/results/
    catqa_behavioral_baseline.json
"""

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_DIAGNOSTIC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DIAGNOSTIC_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from src.model import create_wrapper, _normalize_model_name
from src.extraction import cleanup_gpu

# Reuse classifier helpers (classify_responses.py lives in the same dir)
from classify_responses import (  # type: ignore
    classify_keyword_twoaxis, classify_llm_twoaxis, make_llm_caller,
)

_EXPERIMENT_NAME = "behavioral_ground_truth"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--include_harmless", action="store_true",
                   help="Also generate responses to harmless counterparts")
    p.add_argument("--method", default="llm_twoaxis",
                   choices=["keyword", "llm_twoaxis"])
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--classifier_model", default=None)
    p.add_argument("--skip_if_exists", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = _normalize_model_name(args.model)
    results_dir = (_DIAGNOSTIC_ROOT / model_name / _EXPERIMENT_NAME /
                   "outputs" / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "catqa_behavioral_baseline.json"

    if args.skip_if_exists and out_path.exists():
        print(f"Skipping -- {out_path} already exists.")
        return

    pairs_path = _PROJECT_ROOT / "data" / "catqa-contrastive" / "catqa_contrastive_pairs.json"
    if not pairs_path.exists():
        raise FileNotFoundError(f"CatQA pairs not found: {pairs_path}")
    with open(pairs_path) as f:
        pairs = json.load(f)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"Loaded {len(pairs)} CatQA contrastive pairs.")

    # Classifier
    if args.method == "llm_twoaxis":
        call_fn = make_llm_caller(args.provider, args.classifier_model)

        def classify(q, r):
            return classify_llm_twoaxis(q, r, call_fn, max_retries=2, delay=0.0)
    else:
        def classify(q, r):
            return classify_keyword_twoaxis(r)

    # Generate responses + classify
    wrapper = create_wrapper(args.model).load()
    harmful_records = []
    harmless_records = []

    for i, pair in enumerate(tqdm(pairs, desc="CatQA responses")):
        cat = pair.get("category", "unknown")

        q_h = pair["question_harmful"]
        try:
            resp_h = wrapper.generate_text(q_h, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"  Warning harmful {i}: {e}")
            resp_h = ""
        cleanup_gpu()
        label_h = classify(q_h, resp_h)
        harmful_records.append({
            "idx": i, "category": cat, "question": q_h,
            "response": resp_h, **label_h,
        })

        if args.include_harmless:
            q_hl = pair["question_harmless"]
            try:
                resp_hl = wrapper.generate_text(q_hl, max_new_tokens=args.max_new_tokens)
            except Exception as e:
                print(f"  Warning harmless {i}: {e}")
                resp_hl = ""
            cleanup_gpu()
            label_hl = classify(q_hl, resp_hl)
            harmless_records.append({
                "idx": i, "category": cat, "question": q_hl,
                "response": resp_hl, **label_hl,
            })

    # Aggregate
    def _agg(records):
        n = len(records)
        if n == 0:
            return {}
        harmful_yes = sum(1 for r in records if r["harmful_content"] == "YES")
        awareness = Counter(r["safety_awareness"] for r in records)
        per_cat = defaultdict(lambda: {"n": 0, "harmful": 0, "awareness": Counter()})
        for r in records:
            s = per_cat[r["category"]]
            s["n"] += 1
            if r["harmful_content"] == "YES":
                s["harmful"] += 1
            s["awareness"][r["safety_awareness"]] += 1
        return {
            "n": n,
            "asr_harmful_content": harmful_yes / n,
            "n_harmful": harmful_yes,
            "awareness_distribution": dict(awareness),
            "per_category": {
                cat: {
                    "n": s["n"],
                    "asr_harmful_content": s["harmful"] / s["n"],
                    "awareness_distribution": dict(s["awareness"]),
                } for cat, s in sorted(per_cat.items())
            },
        }

    summary = {
        "method": args.method,
        "model": model_name,
        "harmful": _agg(harmful_records),
        "harmless": _agg(harmless_records) if args.include_harmless else None,
        "records_harmful": harmful_records,
        "records_harmless": harmless_records if args.include_harmless else [],
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {out_path}")

    # Print summary vs HoliSafe SSU if available
    print(f"\n{'='*60}")
    print(f"  CatQA Baseline ({model_name})")
    print(f"{'='*60}")
    h = summary["harmful"]
    print(f"  Harmful questions  (n={h['n']}): "
          f"ASR={h['asr_harmful_content']:.1%} "
          f"awareness={h['awareness_distribution']}")
    if args.include_harmless and summary["harmless"]:
        hl = summary["harmless"]
        print(f"  Harmless questions (n={hl['n']}): "
              f"ASR={hl['asr_harmful_content']:.1%} "
              f"awareness={hl['awareness_distribution']}")

    # Compare to HoliSafe SSU if refusal summary exists
    ref_path = results_dir / "refusal_summary.json"
    if ref_path.exists():
        with open(ref_path) as f:
            ref = json.load(f)
        ssu_entry = ref.get("SSU", {})
        if "vl" in ssu_entry and isinstance(ssu_entry["vl"], dict):
            ssu_vl_asr = ssu_entry["vl"].get("asr_harmful_content")
            print(f"\n  HoliSafe SSU VL ASR: {ssu_vl_asr:.1%}" if ssu_vl_asr is not None else "")
            if ssu_vl_asr is not None:
                delta = h["asr_harmful_content"] - ssu_vl_asr
                print(f"  CatQA - HoliSafe SSU (VL) ASR delta: {delta:+.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
