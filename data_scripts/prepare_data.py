#!/usr/bin/env python3
"""
Master Data Preparation Script
================================
Single entry point that ensures all data artifacts needed by downstream
diagnostic experiments and the evaluation pipeline are present on disk.
Run this FIRST on a fresh workstation before any experiments.

Three phases (each idempotent — existing artifacts are never overwritten):

  Phase 1 — Captions    (CPU + API or GPU for local VLM)
  Phase 2 — Activations (GPU: VL + TT forward passes)
  Phase 3 — Responses   (GPU: vanilla generation)

Usage
-----
  # Everything, default models + benchmarks:
  python data_scripts/prepare_data.py

  # Just captions for two eval benchmarks:
  python data_scripts/prepare_data.py \\
      --benchmarks mm_safetybench figstep --phases captions

  # Activations + responses for a single model:
  python data_scripts/prepare_data.py \\
      --models llava-hf/llava-1.5-7b-hf \\
      --phases activations responses

  # Override caption provider (default: anthropic):
  python data_scripts/prepare_data.py \\
      --caption_provider local --caption_model llava-hf/llava-1.5-7b-hf
"""

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
from src.dataset import DATASET_DATA_DIRS


# ── Registries ────────────────────────────────────────────────────────────────

ALL_BENCHMARKS = ["holisafe", "mssbench", "mm_safetybench", "figstep"]

DEFAULT_MODELS = [
    "llava-hf/llava-1.5-7b-hf",
    "Lin-Chen/ShareGPT4V-7B",
    "Qwen/Qwen-VL-Chat",
]

ALL_PHASES = ["captions", "activations", "responses"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str], desc: str) -> bool:
    """Run a subprocess, streaming output. Returns True on success."""
    print(f"\n  >>> {desc}")
    print(f"  >>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  [FAILED] {desc} (exit {result.returncode})")
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
            / "responses" / "vanilla" / "responses.json")


def _count_captions(benchmark: str) -> int:
    p = _caption_path(benchmark)
    if not p.exists():
        return 0
    with open(p) as f:
        return len(json.load(f))


def _count_activations(benchmark: str, model_short: str, suffix: str) -> int:
    d = _activation_dir(benchmark, model_short)
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.name.endswith(f"_{suffix}.npz"))


# ── Phase 1: Captions ────────────────────────────────────────────────────────

def phase_captions(benchmarks: list[str], provider: str,
                   caption_model: str) -> None:
    print("\n" + "=" * 60)
    print(" Phase 1: Captions")
    print("=" * 60)
    for bm in benchmarks:
        existing = _count_captions(bm)
        if existing > 0:
            print(f"  [{bm}] {existing} captions already exist; "
                  f"will merge new ones if any.")
        cmd = [
            sys.executable, str(_SCRIPT_DIR / "generate_captions.py"),
            "--dataset", bm,
            "--provider", provider,
        ]
        if provider == "local":
            cmd += ["--model", caption_model]
        _run(cmd, f"Caption {bm}")


# ── Phase 2: Activations ─────────────────────────────────────────────────────

def phase_activations(benchmarks: list[str], models: list[str]) -> None:
    print("\n" + "=" * 60)
    print(" Phase 2: Activations (VL + TT)")
    print("=" * 60)
    for model_id in models:
        model_short = _normalize_model_name(model_id)
        for bm in benchmarks:
            n_vl = _count_activations(bm, model_short, "vl")
            n_tt = _count_activations(bm, model_short, "tt")
            print(f"  [{model_short}/{bm}] cached: {n_vl} VL, {n_tt} TT")

            # Check captions exist first (needed for TT)
            if _count_captions(bm) == 0:
                print(f"    SKIP — no captions for {bm}. "
                      f"Run: python data_scripts/prepare_data.py "
                      f"--benchmarks {bm} --phases captions")
                continue

            # VL extraction
            _run([
                sys.executable, str(_SCRIPT_DIR / "extract_vl.py"),
                "--model", model_id,
                "--dataset", bm,
            ], f"Extract VL: {model_short} × {bm}")

            # TT extraction
            _run([
                sys.executable, str(_SCRIPT_DIR / "extract_tt.py"),
                "--model", model_id,
                "--dataset", bm,
            ], f"Extract TT: {model_short} × {bm}")


# ── Phase 3: Vanilla Responses ───────────────────────────────────────────────

def phase_responses(benchmarks: list[str], models: list[str],
                    max_new_tokens: int) -> None:
    print("\n" + "=" * 60)
    print(" Phase 3: Vanilla Responses")
    print("=" * 60)

    # Lazy imports — only needed if this phase runs.
    from src.model import create_wrapper
    from src.extraction import cleanup_gpu
    from evaluation.classifiers.keyword import is_refusal_keyword

    # Benchmark loaders (lazy, to avoid loading large parquets when skipped).
    def _load_benchmark_samples(bm):
        if bm == "holisafe":
            from src.dataset import load_holisafe, filter_subsets
            entries, images_base = load_holisafe()
            sss, ssu = filter_subsets(entries, images_base)
            return sss + ssu
        if bm == "mssbench":
            from src.dataset import load_mssbench as _load_ms
            return _load_ms()
        if bm == "mm_safetybench":
            from evaluation.benchmarks import load_mm_safetybench
            return load_mm_safetybench()
        if bm == "figstep":
            from evaluation.benchmarks import load_figstep
            return load_figstep()
        raise ValueError(f"Unknown benchmark for responses: {bm}")

    for model_id in models:
        model_short = _normalize_model_name(model_id)
        # Check which benchmarks still need responses.
        bms_todo = []
        for bm in benchmarks:
            rp = _response_path(model_short, bm)
            if rp.exists():
                with open(rp) as f:
                    n = len(json.load(f))
                print(f"  [{model_short}/{bm}] {n} responses already exist.")
            else:
                bms_todo.append(bm)

        if not bms_todo:
            print(f"  [{model_short}] all benchmarks complete; skipping model load.")
            continue

        print(f"  Loading {model_id} ...")
        wrapper = create_wrapper(model_id).load()

        for bm in bms_todo:
            rp = _response_path(model_short, bm)
            rp.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path = rp.with_suffix(".checkpoint.json")

            # Resume from checkpoint or existing partial responses.
            records_by_id: dict = {}
            for src in (rp, checkpoint_path):
                if src.exists():
                    try:
                        for r in json.load(open(src)):
                            records_by_id[r["id"]] = r
                    except Exception:
                        pass
            records = list(records_by_id.values())
            done_ids = set(records_by_id.keys())

            raw_samples = _load_benchmark_samples(bm)
            # Normalize to dicts with id/image/text.
            samples = []
            for s in raw_samples:
                if isinstance(s, dict):
                    samples.append(s)
                else:
                    # EvalSample dataclass from evaluation.benchmarks
                    samples.append({
                        "id": s.id, "image_pil": s.image,
                        "text": s.question,
                        "label": getattr(s, "safety_label", None)
                               or getattr(s, "image_type", None) or "eval",
                        "category": getattr(s, "scenario_name", None)
                                   or getattr(s, "benchmark", ""),
                    })

            todo = [s for s in samples if s["id"] not in done_ids]
            print(f"  [{model_short}/{bm}] generating {len(todo)}/{len(samples)} "
                  f"vanilla responses ...")

            from tqdm import tqdm
            from src.dataset import load_image_for_sample
            for s in tqdm(todo, desc=f"{model_short}/{bm}"):
                img = s.get("image_pil") or load_image_for_sample(s)
                try:
                    if img is not None:
                        resp = wrapper.generate_vl(
                            img, s["text"], max_new_tokens=max_new_tokens)
                    else:
                        resp = wrapper.generate_text(
                            s["text"], max_new_tokens=max_new_tokens)
                except Exception as e:
                    print(f"    [error] {s['id']}: {e}")
                    resp = ""
                records.append({
                    "id": s["id"],
                    "question": s["text"],
                    "response": resp,
                    "is_refusal": is_refusal_keyword(resp),
                })
                cleanup_gpu()
                # Checkpoint every sample.
                with open(checkpoint_path, "w") as f:
                    json.dump(records, f)

            with open(rp, "w") as f:
                json.dump(records, f, indent=2)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            print(f"  [{model_short}/{bm}] done → {rp} ({len(records)} records)")

        wrapper.cleanup()
        del wrapper


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Master data preparation: captions, activations, responses."
    )
    p.add_argument("--benchmarks", nargs="+", default=ALL_BENCHMARKS,
                   choices=ALL_BENCHMARKS,
                   help=f"Benchmarks to process (default: {ALL_BENCHMARKS}).")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help=f"Models for activation extraction + response "
                        f"generation (default: {DEFAULT_MODELS}).")
    p.add_argument("--phases", nargs="+", default=ALL_PHASES,
                   choices=ALL_PHASES,
                   help=f"Which phases to run (default: {ALL_PHASES}).")
    p.add_argument("--caption_provider", default="anthropic",
                   choices=["anthropic", "openai", "local"],
                   help="Caption generation backend (default: anthropic).")
    p.add_argument("--caption_model", default="llava-hf/llava-1.5-7b-hf",
                   help="Model for local captioning (only used when "
                        "--caption_provider=local).")
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Max tokens for vanilla response generation.")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    print("=" * 60)
    print(" prepare_data.py")
    print("=" * 60)
    print(f"  benchmarks      : {args.benchmarks}")
    print(f"  models          : {args.models}")
    print(f"  phases          : {args.phases}")
    print(f"  caption_provider: {args.caption_provider}")

    if "captions" in args.phases:
        phase_captions(args.benchmarks, args.caption_provider,
                       args.caption_model)

    if "activations" in args.phases:
        phase_activations(args.benchmarks, args.models)

    if "responses" in args.phases:
        phase_responses(args.benchmarks, args.models, args.max_new_tokens)

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\n{'=' * 60}")
    print(f" Done. Total elapsed: {h:02d}:{m:02d}:{s:02d}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
