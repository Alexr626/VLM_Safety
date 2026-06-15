#!/usr/bin/env python3
"""Master data preparation: captions, activations, and baseline responses."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import _normalize_model_name
from src.dataset import DATASET_DATA_DIRS, ALL_BENCHMARKS

DEFAULT_MODELS = [
    "llava-hf/llava-1.5-7b-hf",
    "Lin-Chen/ShareGPT4V-7B",
    "Qwen/Qwen-VL-Chat",
]
ALL_PHASES = ["captions", "activations", "responses"]


def _run(cmd: list[str], desc: str) -> bool:
    print(f"\n  >>> {desc}")
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  [FAILED] {desc}")
        return False
    return True


def _caption_path(benchmark: str) -> Path:
    return _PROJECT_ROOT / "data" / "captions" / f"{benchmark}.json"


def _activation_dir(benchmark: str, model_short: str) -> Path:
    dataset_dir = DATASET_DATA_DIRS.get(benchmark, benchmark)
    return _PROJECT_ROOT / "data" / dataset_dir / model_short / "activations"


def _response_path(model_short: str, benchmark: str) -> Path:
    dataset_dir = DATASET_DATA_DIRS.get(benchmark, benchmark)
    return (_PROJECT_ROOT / "data" / dataset_dir / model_short
            / "responses" / "no_intervention" / "responses.json")


def phase_captions(benchmarks, provider, caption_model):
    print("\n=== Phase 1: Captions ===")
    for bm in benchmarks:
        cmd = [sys.executable, str(_SCRIPT_DIR / "generate_captions.py"),
               "--dataset", bm, "--provider", provider]
        if provider == "local":
            cmd += ["--model", caption_model]
        _run(cmd, f"Caption {bm}")


def phase_activations(benchmarks, models):
    print("\n=== Phase 2: Activations ===")
    for model_id in models:
        for bm in benchmarks:
            if not _caption_path(bm).exists():
                print(f"  SKIP {bm} — no captions")
                continue
            _run([sys.executable, str(_SCRIPT_DIR / "extract_vl.py"),
                  "--model", model_id, "--dataset", bm],
                 f"VL {model_id} × {bm}")
            _run([sys.executable, str(_SCRIPT_DIR / "extract_tt.py"),
                  "--model", model_id, "--dataset", bm],
                 f"TT {model_id} × {bm}")


def phase_responses(benchmarks, models, max_new_tokens):
    print("\n=== Phase 3: Responses (no_intervention) ===")
    from evaluation.runners.eval_runner import run_evaluation
    for model_id in models:
        run_evaluation(
            model_id=model_id,
            interventions=["no_intervention"],
            benchmarks=benchmarks,
            output_dir=_PROJECT_ROOT / "evaluation" / "results",
            max_new_tokens=max_new_tokens,
            skip_if_exists=True,
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmarks", nargs="+", default=ALL_BENCHMARKS, choices=ALL_BENCHMARKS)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--phases", nargs="+", default=ALL_PHASES, choices=ALL_PHASES)
    p.add_argument("--caption_provider", default="anthropic",
                   choices=["anthropic", "openai", "local"])
    p.add_argument("--caption_model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    if "captions" in args.phases:
        phase_captions(args.benchmarks, args.caption_provider, args.caption_model)
    if "activations" in args.phases:
        phase_activations(args.benchmarks, args.models)
    if "responses" in args.phases:
        phase_responses(args.benchmarks, args.models, args.max_new_tokens)
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
