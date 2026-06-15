#!/usr/bin/env python3
"""FCCT-style causal mediation on POPE yes/no pairs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import load_benchmark
from src.dataset import load_image_for_sample
from src.mediation import (
    COMPONENTS, ScoringTarget, capture_clean_activations,
    get_dispatch, patch_hook_ctx, target_token_probs, verify_layout,
)
from src.model import create_wrapper, _normalize_model_name
from src.paths import diagnostic_results_dir
from src.extraction import save_json

EXPERIMENT = "causal_mediation"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark", default="pope")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--layers", type=int, nargs=2, default=[0, 8],
                   help="Layer range [start, end) for sweep.")
    p.add_argument("--prompt", default="{text} Please answer Yes or No.")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    model_short = _normalize_model_name(args.model)
    results_dir = diagnostic_results_dir(EXPERIMENT, model_short)
    results_dir.mkdir(parents=True, exist_ok=True)

    wrapper = create_wrapper(args.model).load()
    dispatch = get_dispatch(wrapper)
    verify_layout(wrapper, dispatch)
    yes_target, _ = ScoringTarget.yes_no(wrapper)

    samples = load_benchmark(args.benchmark, limit=args.limit)
    layer_range = list(range(args.layers[0], min(args.layers[1], dispatch.n_layers)))

    recovery: dict = {comp: {L: [] for L in layer_range} for comp in COMPONENTS}

    for s in tqdm(samples, desc="mediation"):
        image = load_image_for_sample(s)
        if image is None:
            continue
        text = args.prompt.format(text=s["text"])
        try:
            p_baseline, _ = target_token_probs(wrapper, image, text, yes_target)
            store, _ = capture_clean_activations(wrapper, dispatch, image, text, layers=layer_range)
            p_clean, _ = target_token_probs(wrapper, image, text, yes_target)
            denom = p_clean - p_baseline
            if abs(denom) < 1e-6:
                continue
            for L in layer_range:
                for comp in COMPONENTS:
                    key = (L, comp)
                    if key not in store:
                        continue
                    with patch_hook_ctx(wrapper, dispatch, L, comp, store[key]):
                        p_patch, _ = target_token_probs(wrapper, image, text, yes_target)
                    rr = (p_patch - p_baseline) / denom
                    recovery[comp][L].append(float(rr))
        except Exception as e:
            print(f"  skip {s['id']}: {e}")

    summary = {}
    for comp in COMPONENTS:
        summary[comp] = {}
        for L in layer_range:
            vals = recovery[comp][L]
            summary[comp][str(L)] = {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "n": len(vals),
            }

    save_json(summary, str(results_dir / "recovery_rates.json"))
    save_json({"model": model_short, "benchmark": args.benchmark,
               "n_samples": len(samples), "layers": layer_range},
              str(results_dir / "run_metadata.json"))
    print(f"Wrote -> {results_dir}")


if __name__ == "__main__":
    main()
