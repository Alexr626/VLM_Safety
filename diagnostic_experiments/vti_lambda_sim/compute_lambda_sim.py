#!/usr/bin/env python3
"""Run gated_rotation generation with per-token lambda_sim logging."""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import create_wrapper, _normalize_model_name
from src.paths import diagnostic_results_dir, experiment_artifacts_dir
from evaluation.benchmarks import load_pope_eval
from evaluation.interventions.vti import VTITextualIntervention
from src.extraction import save_json

EXPERIMENT = "vti_lambda_sim"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark", default="pope", choices=["pope"])
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--hook_site", default="mlp", choices=["mlp", "layer"])
    p.add_argument("--pope_split", default="random")
    return p.parse_args()


def main():
    args = parse_args()
    model_short = _normalize_model_name(args.model)
    results_dir = diagnostic_results_dir(EXPERIMENT, model_short)
    results_dir.mkdir(parents=True, exist_ok=True)

    wrapper = create_wrapper(args.model).load()
    intervention = VTITextualIntervention(
        args.model,
        variant="gated_rotation",
        hook_site=args.hook_site,
        log_lambda_sim=True,
    )

    samples = load_pope_eval(split=args.pope_split, limit=args.limit)
    per_sample = []

    for sample in tqdm(samples, desc="lambda_sim"):
        intervention._lambda_log = []
        response = intervention.generate(
            wrapper, sample.image, sample.question, max_new_tokens=256,
        )
        per_sample.append({
            "id": sample.id,
            "question": sample.question,
            "response": response,
            "lambda_records": list(intervention.lambda_log),
        })

    out_path = results_dir / f"lambda_sim_{args.hook_site}_{args.pope_split}.json"
    save_json(per_sample, str(out_path))
    save_json({
        "model": model_short,
        "benchmark": args.benchmark,
        "hook_site": args.hook_site,
        "pope_split": args.pope_split,
        "n_samples": len(per_sample),
        "use_cache_at_generation": True,
        "note": (
            "gated_rotation lambda_sim branch fires per decode step when "
            "use_cache=True (all current wrappers)."
        ),
    }, str(results_dir / f"run_metadata_{args.hook_site}_{args.pope_split}.json"))

    art_dir = experiment_artifacts_dir(EXPERIMENT, model_short)
    art_dir.mkdir(parents=True, exist_ok=True)
    save_json(per_sample, str(art_dir / out_path.name))
    print(f"Wrote {len(per_sample)} samples -> {out_path}")


if __name__ == "__main__":
    main()
