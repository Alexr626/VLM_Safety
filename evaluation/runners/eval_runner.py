"""
Evaluation runner: drives benchmark × intervention generation for a single
model and writes per-sample responses + metric summaries.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import create_wrapper, _normalize_model_name  # noqa: E402
from src.extraction import cleanup_gpu  # noqa: E402

from evaluation.benchmarks import (  # noqa: E402
    load_pope_eval, load_amber_eval, load_chair_eval,
    load_hallusionbench_eval, load_mmhal_bench_eval,
)
from evaluation.classifiers.metrics import compute_metric_records  # noqa: E402
from evaluation.interventions import get_intervention, ALL_INTERVENTIONS  # noqa: E402


_BENCHMARK_LOADERS = {
    "pope": load_pope_eval,
    "amber": load_amber_eval,
    "chair": load_chair_eval,
    "hallusionbench": load_hallusionbench_eval,
    "mmhal_bench": load_mmhal_bench_eval,
}

_CHECKPOINT_EVERY = 50


def _sample_to_meta(sample) -> dict:
    return {
        "id": sample.id,
        "question": sample.question,
        "benchmark": sample.benchmark,
        "task": sample.task,
        "ground_truth": sample.ground_truth,
        "metadata": sample.metadata or {},
    }


def _load_benchmark(name: str, **kwargs) -> list:
    if name not in _BENCHMARK_LOADERS:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available: {sorted(_BENCHMARK_LOADERS)}"
        )
    return _BENCHMARK_LOADERS[name](**kwargs)


def _benchmark_kwargs(name: str, opts: dict) -> dict:
    kw = {"limit": opts.get("limit")}
    if name == "amber" and opts.get("amber_task"):
        kw["task"] = opts["amber_task"]
    if name == "pope" and opts.get("pope_split"):
        kw["split"] = opts["pope_split"]
    return kw


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _run_one(
    wrapper,
    intervention,
    samples: list,
    out_dir: Path,
    max_new_tokens: int,
    skip_if_exists: bool,
    benchmark_name: str,
    model_short: str,
    captions: Optional[dict] = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "responses.json"
    checkpoint_path = out_dir / "responses.checkpoint.json"
    summary_path = out_dir / "metric_summary.json"

    if skip_if_exists and responses_path.exists() and summary_path.exists():
        print(f"  [skip] Existing results at {responses_path}")
        return _load_json(summary_path)

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
        print(f"  [resume] {len(completed_ids)}/{len(samples)} done, {n_todo} remaining")

    t0 = time.time()
    n_done_this_run = 0
    for i, sample in enumerate(samples, 1):
        if sample.id in completed_ids:
            continue
        caption = None
        if captions:
            caption = captions.get(str(sample.id)) or captions.get(sample.id)
        try:
            response = intervention.generate(
                wrapper, sample.image, sample.question,
                max_new_tokens=max_new_tokens,
                caption=caption,
            )
        except Exception as e:
            print(f"  [error] sample {sample.id}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            response = ""
        rec = _sample_to_meta(sample)
        rec["response"] = response
        records.append(rec)
        n_done_this_run += 1
        cleanup_gpu()
        _save_json(records, checkpoint_path)
        if n_done_this_run % _CHECKPOINT_EVERY == 0:
            elapsed = time.time() - t0
            rate = n_done_this_run / elapsed if elapsed > 0 else 0.0
            print(f"  [{i}/{len(samples)}]  {rate:.2f} samples/s")

    _save_json(records, responses_path)
    metrics = compute_metric_records(records, benchmark_name)
    summary = {
        "model": model_short,
        "benchmark": benchmark_name,
        "intervention": intervention.name,
        "intervention_config": intervention.config,
        **metrics,
    }
    _save_json(summary, summary_path)
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            pass

    acc = summary.get("accuracy_overall", summary.get("metric"))
    print(f"  [done] {benchmark_name} × {intervention.name}: metric={acc}")
    return summary


def run_evaluation(
    model_id: str,
    interventions: list[str],
    benchmarks: list[str],
    output_dir: str | Path,
    max_new_tokens: int = 256,
    limit: Optional[int] = None,
    skip_if_exists: bool = False,
    pope_split: str = "random",
    amber_task: Optional[str] = None,
    run_date: Optional[str] = None,
    beta: Optional[float] = None,
) -> dict:
    interventions = interventions or list(ALL_INTERVENTIONS)
    benchmarks = benchmarks or list(_BENCHMARK_LOADERS)
    output_dir = Path(output_dir)
    run_date = run_date or datetime.now().strftime("%Y-%m-%d")
    model_short = _normalize_model_name(model_id)
    print(f"\n=== Evaluating {model_id} ({model_short}) ===")
    print(f"  run_date     : {run_date}")
    print(f"  benchmarks   : {benchmarks}")
    print(f"  interventions: {interventions}")

    opts = {"limit": limit, "pope_split": pope_split, "amber_task": amber_task}
    benchmark_samples: dict[str, list] = {}
    for b in benchmarks:
        print(f"  loading benchmark: {b} ...")
        benchmark_samples[b] = _load_benchmark(b, **_benchmark_kwargs(b, opts))
        print(f"    -> {len(benchmark_samples[b])} samples")

    iv_kwargs = {"beta": beta} if beta is not None else {}
    iv_runs = [(name, get_intervention(name, model_id=model_id, **iv_kwargs))
               for name in interventions]

    print(f"  loading wrapper for {model_id} ...")
    wrapper = create_wrapper(model_id).load()
    print(f"    num_layers={wrapper.num_layers}  hidden_dim={wrapper.hidden_dim}")

    benchmark_captions: dict[str, dict] = {}
    for b_name in benchmarks:
        caption_path = _PROJECT_ROOT / "data" / "captions" / f"{b_name}.json"
        benchmark_captions[b_name] = _load_json(caption_path) if caption_path.exists() else {}

    summaries: dict[tuple[str, str], dict] = {}
    for b_name in benchmarks:
        samples = benchmark_samples[b_name]
        captions = benchmark_captions.get(b_name, {})
        # POPE writes per-split result trees so the three splits do not collide.
        bench_key = f"pope_{pope_split}" if b_name == "pope" else b_name
        for iv_name, iv in iv_runs:
            # Encode beta in the result dir for interventions that use it, so
            # grid points do not collide. no_intervention (config has no 'beta')
            # stays at the bare {iv} path and is computed once across a sweep.
            iv_dir = (f"{iv_name}__b{beta}"
                      if beta is not None and "beta" in iv.config else iv_name)
            out_dir = output_dir / run_date / model_short / bench_key / iv_dir
            print(f"\n--- {bench_key} × {iv_dir} ---")
            summaries[(bench_key, iv_dir)] = _run_one(
                wrapper, iv, samples, out_dir,
                max_new_tokens=max_new_tokens,
                skip_if_exists=skip_if_exists,
                benchmark_name=b_name,
                model_short=model_short,
                captions=captions,
            )
    return {
        "model": model_id,
        "model_short": model_short,
        "run_date": run_date,
        "summaries": {f"{b}__{i}": v for (b, i), v in summaries.items()},
    }


def _looks_like_date(name: str) -> bool:
    parts = name.split("-")
    return len(parts) == 3 and all(p.isdigit() for p in parts)


def _fmt_pope(s: dict) -> str:
    return f"{s.get('accuracy_overall', 0) * 100:.1f}/{s.get('f1_overall', 0) * 100:.1f}"


def _fmt_pct(s: dict) -> Optional[str]:
    v = s.get("accuracy_overall")
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else None


def _fmt_count(s: dict) -> Optional[str]:
    v = s.get("n_total")
    return str(v) if v is not None else None


_POPE_SPLIT_ABBR = {"random": "rnd", "popular": "pop", "adversarial": "adv"}
_POPE_SPLIT_ORDER = ["random", "popular", "adversarial"]
_OTHER_COLUMNS = [
    ("AMBER", "amber", _fmt_pct),
    ("CHAIR", "chair", _fmt_count),
    ("Hallusion", "hallusionbench", _fmt_pct),
    ("MMHal", "mmhal_bench", _fmt_pct),
]


def _resolve_model_root(results_root: Path, model_short: str,
                        run_date: Optional[str]) -> Optional[Path]:
    if run_date is not None:
        cand = results_root / run_date / model_short
        return cand if cand.exists() else None
    # No date given: prefer the most recent date dir that has this model, else
    # fall back to the legacy (pre-date) layout results_root/model_short.
    date_dirs = sorted(
        (p for p in results_root.iterdir()
         if p.is_dir() and _looks_like_date(p.name) and (p / model_short).is_dir()),
        reverse=True,
    ) if results_root.exists() else []
    if date_dirs:
        return date_dirs[0] / model_short
    legacy = results_root / model_short
    return legacy if legacy.exists() else None


def print_comparison_table(model_short: str, results_root: str | Path,
                           run_date: Optional[str] = None) -> None:
    results_root = Path(results_root)
    root = _resolve_model_root(results_root, model_short, run_date)
    if root is None:
        print(f"No results found for {model_short} under {results_root}")
        return

    interventions: list[str] = []
    by_iv: dict[str, dict[str, dict]] = {}
    bench_keys: set[str] = set()
    for benchmark_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        b_key = benchmark_dir.name
        for iv_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
            iv = iv_dir.name
            summary_path = iv_dir / "metric_summary.json"
            if not summary_path.exists():
                continue
            if iv not in interventions:
                interventions.append(iv)
            by_iv.setdefault(iv, {})[b_key] = _load_json(summary_path)
            bench_keys.add(b_key)

    if not interventions:
        print(f"No metric_summary.json files under {root}")
        return

    # Build columns: per-split POPE first (in canonical order), then the others.
    columns: list[tuple[str, str, object]] = []
    pope_keys = {k for k in bench_keys if k == "pope" or k.startswith("pope_")}
    for split in _POPE_SPLIT_ORDER:
        key = f"pope_{split}"
        if key in pope_keys:
            columns.append((f"POPE/{_POPE_SPLIT_ABBR[split]}", key, _fmt_pope))
    for key in sorted(pope_keys - {f"pope_{s}" for s in _POPE_SPLIT_ORDER}):
        label = "POPE" if key == "pope" else f"POPE/{key[5:]}"
        columns.append((label, key, _fmt_pope))
    for header, key, fmt in _OTHER_COLUMNS:
        if key in bench_keys:
            columns.append((header, key, fmt))

    label_w = max(len(iv) for iv in interventions) + 2
    col_w = [max(len(h), 11) for h, _, _ in columns]

    print(f"\nModel: {model_short}  ({root.parent.name})")
    print(" " * label_w + "".join(h.rjust(w + 2) for (h, _, _), w in zip(columns, col_w)))
    print(" " * label_w + "".join("-" * (w + 2) for w in col_w))
    for iv in interventions:
        cells = []
        for (_, key, fmt), w in zip(columns, col_w):
            s = by_iv.get(iv, {}).get(key)
            val = fmt(s) if s else None
            cells.append((val if val is not None else "--").rjust(w + 2))
        print(iv.ljust(label_w) + "".join(cells))
