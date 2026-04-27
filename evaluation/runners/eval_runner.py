"""
Evaluation runner: drives benchmark × intervention generation for a single
model, writes per-sample responses + ASR summaries, and prints a comparison
table across interventions.

Note on data isolation: the compositional safety direction `c^l` used by
CompSafetyShift is estimated from HoliSafe TT activations, NOT MM-SafetyBench /
FigStep / MSSBench. There is therefore no overlap between calibration data and
evaluation data.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import create_wrapper, _normalize_model_name  # noqa: E402
from src.extraction import cleanup_gpu  # noqa: E402

from evaluation.benchmarks import (  # noqa: E402
    load_mm_safetybench, load_figstep, load_mssbench,
)
from evaluation.classifiers.keyword import (  # noqa: E402
    is_refusal_keyword, compute_asr_records,
)
from evaluation.interventions import (  # noqa: E402
    get_intervention, ALL_INTERVENTIONS,
)


_BENCHMARK_LOADERS = {
    "mm_safetybench": load_mm_safetybench,
    "figstep":        load_figstep,
    "mssbench":       load_mssbench,
}

_CHECKPOINT_EVERY = 50


# ── Sample → record helpers ───────────────────────────────────────────────────

def _sample_to_meta(sample) -> dict:
    """Subset of EvalSample fields safe to write to JSON (excluding the image)."""
    return {
        "id": sample.id,
        "question": sample.question,
        "benchmark": sample.benchmark,
        "scenario_id": sample.scenario_id,
        "scenario_name": sample.scenario_name,
        "image_type": sample.image_type,
        "safety_label": sample.safety_label,
    }


def _load_benchmark(name: str, **kwargs) -> list:
    if name not in _BENCHMARK_LOADERS:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available: {sorted(_BENCHMARK_LOADERS)}"
        )
    return _BENCHMARK_LOADERS[name](**kwargs)


def _benchmark_kwargs(name: str, opts: dict) -> dict:
    if name == "mm_safetybench":
        return {
            "scenarios": opts.get("mm_safetybench_scenarios"),
            "image_types": opts.get("mm_safetybench_image_types"),
            "limit_per_scenario": opts.get("limit"),
        }
    if name == "figstep":
        return {"limit": opts.get("limit")}
    if name == "mssbench":
        return {
            "safety_labels": opts.get("mssbench_safety_labels"),
            "limit": opts.get("limit"),
        }
    return {}


# ── I/O ───────────────────────────────────────────────────────────────────────

def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ── Per-(benchmark, intervention) drive ───────────────────────────────────────

def _run_one(
    wrapper,
    intervention,
    samples: list,
    out_dir: Path,
    max_new_tokens: int,
    skip_if_exists: bool,
    benchmark_name: str,
    model_short: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "responses.json"
    checkpoint_path = out_dir / "responses.checkpoint.json"
    summary_path = out_dir / "asr_summary.json"

    if skip_if_exists and responses_path.exists() and summary_path.exists():
        print(f"  [skip] Existing results at {responses_path}")
        return _load_json(summary_path)

    # Resume from checkpoint if available.
    completed_ids: set[str] = set()
    records: list[dict] = []
    if checkpoint_path.exists():
        try:
            records = _load_json(checkpoint_path)
            completed_ids = {r["id"] for r in records}
            print(f"  [resume] {len(completed_ids)} samples already in checkpoint")
        except Exception:
            records = []
            completed_ids = set()

    t0 = time.time()
    for i, sample in enumerate(samples, 1):
        if sample.id in completed_ids:
            continue
        try:
            response = intervention.generate(
                wrapper, sample.image, sample.question,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            print(f"  [error] sample {sample.id}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            response = ""
        rec = _sample_to_meta(sample)
        rec["response"] = response
        rec["is_refusal"] = is_refusal_keyword(response)
        records.append(rec)
        cleanup_gpu()

        if i % _CHECKPOINT_EVERY == 0:
            _save_json(records, checkpoint_path)
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            print(f"  [{i}/{len(samples)}]  {rate:.2f} samples/s")

    _save_json(records, responses_path)
    summary = {
        "model": model_short,
        "benchmark": benchmark_name,
        "intervention": intervention.name,
        "intervention_config": intervention.config,
        **compute_asr_records(records),
    }
    _save_json(summary, summary_path)
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            pass

    print(f"  [done] {benchmark_name} × {intervention.name}: "
          f"asr={summary['asr_overall']:.3f} (n={summary['n_total']})")
    return summary


# ── Public entry point ────────────────────────────────────────────────────────

def run_evaluation(
    model_id: str,
    interventions: list[str],
    benchmarks: list[str],
    output_dir: str | Path,
    max_new_tokens: int = 256,
    limit: Optional[int] = None,
    skip_if_exists: bool = False,
    mm_safetybench_scenarios: Optional[list[int]] = None,
    mm_safetybench_image_types: Optional[list[str]] = None,
    mssbench_safety_labels: Optional[list[str]] = None,
) -> dict:
    interventions = interventions or list(ALL_INTERVENTIONS)
    benchmarks = benchmarks or list(_BENCHMARK_LOADERS)

    output_dir = Path(output_dir)
    model_short = _normalize_model_name(model_id)
    print(f"\n=== Evaluating {model_id} ({model_short}) ===")
    print(f"  benchmarks   : {benchmarks}")
    print(f"  interventions: {interventions}")

    # ── Pre-flight checks (fail fast before model load) ──────────────────────
    opts = {
        "limit": limit,
        "mm_safetybench_scenarios": mm_safetybench_scenarios,
        "mm_safetybench_image_types": mm_safetybench_image_types,
        "mssbench_safety_labels": mssbench_safety_labels,
    }
    benchmark_samples: dict[str, list] = {}
    for b in benchmarks:
        print(f"  loading benchmark: {b} ...")
        benchmark_samples[b] = _load_benchmark(b, **_benchmark_kwargs(b, opts))
        print(f"    -> {len(benchmark_samples[b])} samples")

    # Validate intervention artefacts before loading the model.
    intervention_objs: dict[str, "InterventionBase"] = {}
    for iv_name in interventions:
        intervention_objs[iv_name] = get_intervention(iv_name, model_id=model_id)
    print(f"  interventions ready: {list(intervention_objs)}")

    # ── Load model once and reuse across all (benchmark, intervention) ───────
    print(f"  loading wrapper for {model_id} ...")
    wrapper = create_wrapper(model_id).load()
    print(f"    num_layers={wrapper.num_layers}  hidden_dim={wrapper.hidden_dim}")

    # Validate CompSafetyShift direction shape against the loaded model.
    for iv in intervention_objs.values():
        if iv.name == "comp_safety_shift":
            iv._validate_hidden_dim(wrapper)

    # ── Drive ────────────────────────────────────────────────────────────────
    summaries: dict[tuple[str, str], dict] = {}
    for b_name in benchmarks:
        samples = benchmark_samples[b_name]
        for iv_name in interventions:
            iv = intervention_objs[iv_name]
            out_dir = output_dir / model_short / b_name / iv_name
            print(f"\n--- {b_name} × {iv_name} ---")
            summaries[(b_name, iv_name)] = _run_one(
                wrapper, iv, samples, out_dir,
                max_new_tokens=max_new_tokens,
                skip_if_exists=skip_if_exists,
                benchmark_name=b_name,
                model_short=model_short,
            )
    return {
        "model": model_id,
        "model_short": model_short,
        "summaries": {f"{b}__{i}": v for (b, i), v in summaries.items()},
    }


# ── Comparison table ──────────────────────────────────────────────────────────

_TABLE_COLUMNS = [
    ("MM(SD)",        "mm_safetybench", lambda s: s.get("by_image_type", {}).get("SD",      {}).get("asr")),
    ("MM(OCR)",       "mm_safetybench", lambda s: s.get("by_image_type", {}).get("OCR",     {}).get("asr")),
    ("MM(SD_TYPO)",   "mm_safetybench", lambda s: s.get("by_image_type", {}).get("SD_TYPO", {}).get("asr")),
    ("FigStep",       "figstep",        lambda s: s.get("asr_overall")),
    ("MSSB(SSU)",     "mssbench",       lambda s: s.get("by_safety_label", {}).get("SSU", {}).get("asr") or s.get("asr_overall")),
]


def print_comparison_table(model_short: str, results_root: str | Path) -> None:
    """Read all asr_summary.json files for a model and print a comparison table."""
    root = Path(results_root) / model_short
    if not root.exists():
        print(f"No results found at {root}")
        return

    interventions: list[str] = []
    by_iv: dict[str, dict[str, dict]] = {}
    for benchmark_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        b_name = benchmark_dir.name
        for iv_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
            iv = iv_dir.name
            summary_path = iv_dir / "asr_summary.json"
            if not summary_path.exists():
                continue
            if iv not in interventions:
                interventions.append(iv)
            by_iv.setdefault(iv, {})[b_name] = _load_json(summary_path)

    if not interventions:
        print(f"No asr_summary.json files under {root}")
        return

    label_w = max(len(iv) for iv in interventions) + 2
    col_headers = [c[0] for c in _TABLE_COLUMNS]
    col_w = [max(len(h), 9) for h in col_headers]

    print(f"\nModel: {model_short}")
    header = " " * label_w + "".join(
        h.rjust(w + 2) for h, w in zip(col_headers, col_w)
    )
    print(header)
    print(" " * label_w + "".join("-" * (w + 2) for w in col_w))
    for iv in interventions:
        cells = []
        for (_, b_name, getter), w in zip(_TABLE_COLUMNS, col_w):
            s = by_iv.get(iv, {}).get(b_name)
            val = getter(s) if s else None
            cells.append(("--" if val is None else f"{val * 100:.1f}%").rjust(w + 2))
        print(iv.ljust(label_w) + "".join(cells))
