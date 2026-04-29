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
            "eval_only": opts.get("mssbench_eval_only", False),
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


def _load_mssbench_eval_ids(project_root: Path) -> Optional[set[str]]:
    """Return the set of EvalSample.id strings in the MSSBench eval split,
    or None if the split file doesn't exist."""
    split_path = project_root / "data" / "mssbench" / "train_eval_split.json"
    if not split_path.exists():
        return None
    with open(split_path) as f:
        split = json.load(f)
    ids = split.get("eval_sample_ids") or []
    return set(ids) if ids else None


def write_eval_only_asr_summary(
    out_dir: Path,
    eval_ids: set,
    base_summary: Optional[dict] = None,
    subset_label: str = "mssbench_eval",
) -> Optional[dict]:
    """Filter `responses.json` in `out_dir` to records whose id is in
    `eval_ids`, recompute ASR via `compute_asr_records`, and persist the
    result as `asr_summary_eval.json`.

    Used both inline (immediately after a generation run) and post-hoc (by
    `evaluation/scripts/recompute_mssbench_eval_asr.py`) — the operation is
    a pure function of `responses.json`, so the existing responses are
    reused without any model load.

    Returns the summary dict that was written, or None if `responses.json`
    is missing.
    """
    responses_path = out_dir / "responses.json"
    if not responses_path.exists():
        return None
    records = _load_json(responses_path)
    if not isinstance(records, list):
        return None
    filtered = [r for r in records if r.get("id") in eval_ids]
    eval_summary: dict = {
        "subset": subset_label,
        "n_eval_ids": len(eval_ids),
        "n_responses_total": len(records),
        "n_responses_in_eval": len(filtered),
        **compute_asr_records(filtered),
    }
    if base_summary is not None:
        for k in ("model", "benchmark", "intervention", "intervention_config"):
            if k in base_summary:
                eval_summary.setdefault(k, base_summary[k])
    _save_json(eval_summary, out_dir / "asr_summary_eval.json")
    return eval_summary


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
    mssbench_eval_ids: Optional[set] = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "responses.json"
    checkpoint_path = out_dir / "responses.checkpoint.json"
    summary_path = out_dir / "asr_summary.json"

    if skip_if_exists and responses_path.exists() and summary_path.exists():
        print(f"  [skip] Existing results at {responses_path}")
        # Even on skip, refresh the eval-only summary so changes to the split
        # propagate without forcing a full regeneration.
        if benchmark_name == "mssbench" and mssbench_eval_ids:
            write_eval_only_asr_summary(
                out_dir, mssbench_eval_ids,
                base_summary=_load_json(summary_path),
            )
        return _load_json(summary_path)

    # Resume: merge records from both responses.json (partial prior run) and
    # responses.checkpoint.json (mid-run checkpoint), keyed by sample id so
    # duplicates are impossible. This handles every crash/interrupt scenario.
    records_by_id: dict[str, dict] = {}
    for src_path in (responses_path, checkpoint_path):
        if src_path.exists():
            try:
                for r in _load_json(src_path):
                    records_by_id[r["id"]] = r
            except Exception:
                pass
    records = list(records_by_id.values())
    completed_ids = set(records_by_id.keys())
    n_todo = sum(1 for s in samples if s.id not in completed_ids)
    if completed_ids:
        print(f"  [resume] {len(completed_ids)}/{len(samples)} already done, "
              f"{n_todo} remaining")

    t0 = time.time()
    n_done_this_run = 0
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
        n_done_this_run += 1
        cleanup_gpu()

        # Checkpoint after every sample — disk I/O is negligible vs. a VLM
        # forward pass, and losing work on interrupt is not.
        _save_json(records, checkpoint_path)

        if n_done_this_run % _CHECKPOINT_EVERY == 0:
            elapsed = time.time() - t0
            rate = n_done_this_run / elapsed if elapsed > 0 else 0.0
            print(f"  [{i}/{len(samples)}]  {rate:.2f} samples/s  "
                  f"({n_done_this_run} new this run)")

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

    if benchmark_name == "mssbench" and mssbench_eval_ids:
        eval_summary = write_eval_only_asr_summary(
            out_dir, mssbench_eval_ids, base_summary=summary,
        )
        if eval_summary:
            print(f"  [done] {benchmark_name} × {intervention.name} eval-only: "
                  f"asr={eval_summary['asr_overall']:.3f} "
                  f"(n={eval_summary['n_responses_in_eval']})")

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
    mssbench_eval_only: bool = False,
    mssbench_compare_on_eval: bool = True,
    comp_safety_sources: Optional[list[str]] = None,
) -> dict:
    interventions = interventions or list(ALL_INTERVENTIONS)
    benchmarks = benchmarks or list(_BENCHMARK_LOADERS)
    comp_safety_sources = comp_safety_sources or ["mssbench_vl"]

    output_dir = Path(output_dir)
    model_short = _normalize_model_name(model_id)
    print(f"\n=== Evaluating {model_id} ({model_short}) ===")
    print(f"  benchmarks   : {benchmarks}")
    print(f"  interventions: {interventions}")
    if "comp_safety_shift" in interventions:
        print(f"  comp_safety_sources: {comp_safety_sources}")
    if "mssbench" in benchmarks:
        print(f"  mssbench_eval_only:        {mssbench_eval_only}")
        print(f"  mssbench_compare_on_eval:  {mssbench_compare_on_eval}")

    # Pre-load the MSSBench eval id set once (cheap; same set used by every
    # per-(intervention, sample) write).
    mssbench_eval_ids: Optional[set] = None
    if "mssbench" in benchmarks and mssbench_compare_on_eval:
        mssbench_eval_ids = _load_mssbench_eval_ids(_PROJECT_ROOT)
        if mssbench_eval_ids is None:
            print("  WARN: --mssbench_compare_on_eval requested but "
                  "data/mssbench/train_eval_split.json is missing. "
                  "Run: python -m src.dataset --mssbench_split")

    # ── Pre-flight checks (fail fast before model load) ──────────────────────
    opts = {
        "limit": limit,
        "mm_safetybench_scenarios": mm_safetybench_scenarios,
        "mm_safetybench_image_types": mm_safetybench_image_types,
        "mssbench_safety_labels": mssbench_safety_labels,
        "mssbench_eval_only": mssbench_eval_only,
    }
    benchmark_samples: dict[str, list] = {}
    for b in benchmarks:
        print(f"  loading benchmark: {b} ...")
        benchmark_samples[b] = _load_benchmark(b, **_benchmark_kwargs(b, opts))
        print(f"    -> {len(benchmark_samples[b])} samples")

    # Build (output_subdir, intervention_obj) tuples in execution order.
    # comp_safety_shift expands into one entry per --comp_safety_sources value;
    # other interventions appear once at their canonical name.
    iv_runs: list[tuple[str, "InterventionBase"]] = []
    for iv_name in interventions:
        if iv_name == "comp_safety_shift":
            for source in comp_safety_sources:
                iv = get_intervention(
                    "comp_safety_shift", model_id=model_id,
                    direction_source=source,
                )
                iv_runs.append((f"comp_safety_shift_{source}", iv))
        else:
            iv = get_intervention(iv_name, model_id=model_id)
            iv_runs.append((iv_name, iv))
    print(f"  interventions ready: {[name for name, _ in iv_runs]}")

    # ── Load model once and reuse across all (benchmark, intervention) ───────
    print(f"  loading wrapper for {model_id} ...")
    wrapper = create_wrapper(model_id).load()
    print(f"    num_layers={wrapper.num_layers}  hidden_dim={wrapper.hidden_dim}")

    # Validate CompSafetyShift direction shape against the loaded model.
    for _, iv in iv_runs:
        if iv.name == "comp_safety_shift":
            iv._validate_hidden_dim(wrapper)

    # ── Drive ────────────────────────────────────────────────────────────────
    summaries: dict[tuple[str, str], dict] = {}
    for b_name in benchmarks:
        samples = benchmark_samples[b_name]
        for output_subdir, iv in iv_runs:
            out_dir = output_dir / model_short / b_name / output_subdir
            print(f"\n--- {b_name} × {output_subdir} ---")
            summaries[(b_name, output_subdir)] = _run_one(
                wrapper, iv, samples, out_dir,
                max_new_tokens=max_new_tokens,
                skip_if_exists=skip_if_exists,
                benchmark_name=b_name,
                model_short=model_short,
                mssbench_eval_ids=(mssbench_eval_ids if b_name == "mssbench" else None),
            )
    return {
        "model": model_id,
        "model_short": model_short,
        "summaries": {f"{b}__{i}": v for (b, i), v in summaries.items()},
    }


# ── Comparison table ──────────────────────────────────────────────────────────

def _mssb_ssu(s: dict) -> Optional[float]:
    return s.get("by_safety_label", {}).get("SSU", {}).get("asr") or s.get("asr_overall")


# (column header, benchmark, getter, source_file).
# source_file=None -> always read asr_summary.json
# source_file="eval" -> read asr_summary_eval.json (the held-out subset)
_TABLE_COLUMNS_FULL = [
    ("MM(SD)",          "mm_safetybench", lambda s: s.get("by_image_type", {}).get("SD",      {}).get("asr"), None),
    ("MM(OCR)",         "mm_safetybench", lambda s: s.get("by_image_type", {}).get("OCR",     {}).get("asr"), None),
    ("MM(SD_TYPO)",     "mm_safetybench", lambda s: s.get("by_image_type", {}).get("SD_TYPO", {}).get("asr"), None),
    ("FigStep",         "figstep",        lambda s: s.get("asr_overall"),                                    None),
    ("MSSB(SSU)",       "mssbench",       _mssb_ssu,                                                         None),
]
_MSSBENCH_EVAL_COL = ("MSSB(eval/SSU)", "mssbench", _mssb_ssu, "eval")


def print_comparison_table(
    model_short: str,
    results_root: str | Path,
    mssbench_view: str = "eval",
) -> None:
    """Read all asr_summary*.json files for a model and print a comparison table.

    `mssbench_view`:
      - "full" : MSSB column reads asr_summary.json (responses across the
                 full MSSBench).
      - "eval" : MSSB column reads asr_summary_eval.json (responses filtered
                 to the held-out eval split). Default — matches the fair-
                 comparison setup used when comp_safety_shift is trained on
                 MSSBench.
      - "both" : show both columns side by side.
    """
    if mssbench_view not in ("full", "eval", "both"):
        raise ValueError(
            f"mssbench_view must be one of 'full', 'eval', 'both' "
            f"(got {mssbench_view!r})"
        )

    root = Path(results_root) / model_short
    if not root.exists():
        print(f"No results found at {root}")
        return

    # Build the column list according to mssbench_view.
    columns: list[tuple] = []
    for col in _TABLE_COLUMNS_FULL:
        header, benchmark, getter, _ = col
        if benchmark == "mssbench":
            if mssbench_view == "full":
                columns.append(col)
            elif mssbench_view == "eval":
                columns.append(_MSSBENCH_EVAL_COL)
            else:  # both
                columns.append(col)                   # full
                columns.append(_MSSBENCH_EVAL_COL)    # eval-only
        else:
            columns.append(col)

    interventions: list[str] = []
    # by_iv[iv][benchmark][source] = summary dict
    # source ∈ {"full", "eval"} so a single iv dir can carry both.
    by_iv: dict[str, dict[str, dict[str, dict]]] = {}
    for benchmark_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        b_name = benchmark_dir.name
        for iv_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
            iv = iv_dir.name
            full_path = iv_dir / "asr_summary.json"
            eval_path = iv_dir / "asr_summary_eval.json"
            if not full_path.exists() and not eval_path.exists():
                continue
            if iv not in interventions:
                interventions.append(iv)
            slot = by_iv.setdefault(iv, {}).setdefault(b_name, {})
            if full_path.exists():
                slot["full"] = _load_json(full_path)
            if eval_path.exists():
                slot["eval"] = _load_json(eval_path)

    if not interventions:
        print(f"No asr_summary*.json files under {root}")
        return

    label_w = max(len(iv) for iv in interventions) + 2
    col_headers = [c[0] for c in columns]
    col_w = [max(len(h), 9) for h in col_headers]

    print(f"\nModel: {model_short}  (mssbench_view={mssbench_view})")
    header = " " * label_w + "".join(
        h.rjust(w + 2) for h, w in zip(col_headers, col_w)
    )
    print(header)
    print(" " * label_w + "".join("-" * (w + 2) for w in col_w))
    for iv in interventions:
        cells = []
        for col, w in zip(columns, col_w):
            _, b_name, getter, source = col
            slot = by_iv.get(iv, {}).get(b_name) or {}
            src = source or "full"
            s = slot.get(src)
            # Graceful fallback: when "eval" requested but missing, leave
            # blank so the user sees that the eval summary hasn't been
            # written yet (e.g. forgot to run the post-hoc tool).
            val = getter(s) if s else None
            cells.append(("--" if val is None else f"{val * 100:.1f}%").rjust(w + 2))
        print(iv.ljust(label_w) + "".join(cells))
