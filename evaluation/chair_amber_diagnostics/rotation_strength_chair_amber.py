#!/usr/bin/env python3
"""Experiment 2 — CHAIR / AMBER rotation-strength sweep (layer-hook site).

The `vti_rotation_strength/rotation_strength.py` analog on the new benchmarks:
sweeps the textual steering coefficient beta for the layer-site uniform_rotation
variant — the residual-stream site where the stark length / yes-bias / collapse
effects were seen on POPE — over the PINNED CHAIR-500 or AMBER-450 subset.

For each beta every steered response is classified vs the true no-hook baseline:
  - empty     : whitespace-only generation (degenerate collapse)
  - identical : byte-identical to the no-hook baseline (steering inert here)
  - changed   : non-empty and different from baseline
and per-beta benchmark metrics are reported as metric-vs-beta curves:

  CHAIR : chair_s, chair_i (over NON-EMPTY captions, §C), avg_objects_mentioned,
          avg_caption_len_chars (over ALL, shows collapse), n_empty/empty_fraction,
          mean_len.
  AMBER : accuracy, f1, yes_ratio, neg_item_accuracy, pos_item_accuracy,
          n_unparsed, by_qtype{existence,attribute,relation}, plus POPE-style
          decision-flip accounting vs baseline (induced/removed hallucinations).

Mitigation probes at the strongest beta (decode-only steering; leaving sequence
position 0 unsteered) are also run, mirroring the POPE sweep.

Generic gen / classify / flip helpers are imported from the POPE rotation-
strength module so the steering path + flip confusion-matrix logic have a single
source of truth.

Results land under the dated results tree (per the diagnostics plan):
  evaluation/results/{run_date}/{model_short}/{benchmark}_rotation_strength/
      sweep_uniform_rotation_layer_n{N}.json

Usage:
  CUDA_VISIBLE_DEVICES=0 python evaluation/chair_amber_diagnostics/\\
      rotation_strength_chair_amber.py --benchmark chair --model llava-hf/llava-1.5-7b-hf
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import create_wrapper, _normalize_model_name  # noqa: E402
from src.extraction import save_json  # noqa: E402
from src.dataset import benchmark_data_dir  # noqa: E402
from evaluation.benchmarks import load_chair_eval, load_amber_eval  # noqa: E402
from evaluation.classifiers.metrics import (  # noqa: E402
    score_chair_records, score_amber_discriminative_records,
)
from evaluation.interventions.vti import VTITextualIntervention  # noqa: E402
# Reuse the generic steering / classification / flip helpers (single source).
from evaluation.vti_rotation_strength.rotation_strength import (  # noqa: E402
    _gen, _gen_steered, _classify, _flip_counts, _ZERO_FLIPS,
)

EXPERIMENT = "chair_amber_rotation_strength"
HOOK_SITE = "layer"
VTI_CHAIR_PROMPT = "Please Describe this image in detail."

_PINNED = {
    "chair": "data/chair/pinned_chair_500.json",
    "amber": "data/amber/pinned_amber_disc_450.json",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("MODEL", "llava-hf/llava-1.5-7b-hf"))
    p.add_argument("--benchmark", required=True, choices=["chair", "amber"])
    p.add_argument("--variant", default="uniform_rotation",
                   choices=["uniform_rotation", "gated_rotation"])
    p.add_argument("--betas", type=float, nargs="+",
                   default=[0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.1])
    p.add_argument("--max_new_tokens", type=int, default=64,
                   help="Frozen generation budget (CHAIR cap = 64 per Step 0; "
                        "AMBER yes/no needs little).")
    p.add_argument("--subset_file", default=None,
                   help="Pinned subset JSON; defaults to the benchmark's pin.")
    p.add_argument("--chair_prompt", default=VTI_CHAIR_PROMPT)
    p.add_argument("--run_date", default=None)
    p.add_argument("--max_pixels", type=int, default=None,
                   help="Cap the Qwen2/2.5-VL visual-token budget (pixels = "
                        "tokens * 28*28). Native resolution is uncapped, so some "
                        "CHAIR/AMBER images explode the attention matrix and OOM; "
                        "e.g. 1003520 (=1280*28*28) bounds it. Ignored for "
                        "non-Qwen2 models.")
    p.add_argument("--skip_if_exists", action="store_true",
                   help="Skip this (model,benchmark) sweep if its final JSON "
                        "already exists; otherwise resume from a per-sample "
                        "checkpoint so an interrupted sweep does not restart.")
    return p.parse_args()


def _rec_complete(rec: dict, betas: list) -> bool:
    """A checkpointed per-sample record is reusable only if it has every field
    the current sweep needs (baseline, all betas, both beta-max probes)."""
    if not rec or "baseline" not in rec:
        return False
    bb = rec.get("by_beta") or {}
    if any(f"{b}" not in bb or "response" not in bb[f"{b}"] for b in betas):
        return False
    return "decode_only_beta_max" in rec and "skip_pos0_beta_max" in rec


def _load_subset_ids(benchmark: str, path: str) -> set:
    spec = json.load(open(path))
    ids = spec.get(benchmark) if isinstance(spec, dict) else spec
    if not ids:
        raise ValueError(f"No '{benchmark}' ids in subset file {path}")
    return set(ids)


def _round_group(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def _score_chair(responses, samples) -> dict:
    records = [{"response": r, "metadata": {"raw": (s.metadata or {}).get("raw") or {}}}
               for r, s in zip(responses, samples)]
    m = score_chair_records(records)
    mean_len = sum(len(r) for r in responses) / len(responses) if responses else 0.0
    return {
        "chair_s": round(m["chair_s"], 4),
        "chair_i": round(m["chair_i"], 4),
        "avg_objects_mentioned": round(m["avg_objects_mentioned"], 3),
        "avg_caption_len_chars": round(m["avg_caption_len_chars"], 1),
        "n_empty": m["n_empty"],
        "n_nonempty": m["n_nonempty"],
        "empty_fraction": round(m["empty_fraction"], 4),
        "mean_len": round(mean_len, 1),
    }


def _score_amber(responses, samples) -> dict:
    records = [{"response": r, "ground_truth": s.ground_truth, "task": "discriminative",
                "metadata": {"category": (s.metadata or {}).get("category")}}
               for r, s in zip(responses, samples)]
    m = score_amber_discriminative_records(records)
    return {
        "accuracy": round(m["accuracy_overall"], 4),
        "f1": round(m["f1_overall"], 4),
        "yes_ratio": round(m["yes_ratio"], 4),
        "neg_item_accuracy": round(m["neg_item_accuracy"], 4),
        "pos_item_accuracy": round(m["pos_item_accuracy"], 4),
        "n_unparsed": m["n_unparsed"],
        "n_total": m["n_total"],
        "by_qtype": {q: _round_group(v) for q, v in m["by_qtype"].items()},
    }


def main():
    args = parse_args()
    run_date = args.run_date or datetime.now().strftime("%Y-%m-%d")
    model_short = _normalize_model_name(args.model)
    bench = args.benchmark
    subset_file = args.subset_file or (_PROJECT_ROOT / _PINNED[bench])
    subset_ids = _load_subset_ids(bench, str(subset_file))

    is_chair = bench == "chair"
    score_fn = _score_chair if is_chair else _score_amber

    if is_chair:
        samples = load_chair_eval(subset_ids=subset_ids,
                                  prompt_override=args.chair_prompt)
    else:
        samples = load_amber_eval(task="discriminative", subset_ids=subset_ids)
    n = len(samples)

    print(f"\nmodel={args.model}  benchmark={bench}  variant={args.variant} @ {HOOK_SITE}")
    print(f"samples={n}  subset={subset_file}  run_date={run_date}  betas={args.betas}")
    if is_chair:
        print(f"chair_prompt={args.chair_prompt!r}  max_new_tokens={args.max_new_tokens}")

    out_dir = (_PROJECT_ROOT / "evaluation" / "results" / run_date
               / model_short / f"{bench}_rotation_strength")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sweep_{args.variant}_{HOOK_SITE}_n{n}.json"
    checkpoint_path = out_dir / f"sweep_{args.variant}_{HOOK_SITE}_n{n}.checkpoint.json"

    # Whole-sweep skip: a completed (model,benchmark) sweep is not re-run.
    if args.skip_if_exists and out_path.exists():
        print(f"  [skip] existing sweep at {out_path}")
        return

    # Per-sample resume: reuse any complete records from a prior interrupted run
    # so the sweep does not restart from the very beginning. `failed_ids` records
    # samples that OOM'd under the same config so resume does not retry them.
    done_by_id: dict = {}
    prev_failed: set = set()
    if checkpoint_path.exists():
        try:
            ck = json.loads(checkpoint_path.read_text())
            for rec in ck.get("per_sample", []):
                if _rec_complete(rec, args.betas):
                    done_by_id[rec["id"]] = rec
            if ck.get("max_pixels") == args.max_pixels:
                prev_failed = set(ck.get("failed_ids", []))
            print(f"  [resume] loaded {len(done_by_id)} completed samples"
                  + (f", {len(prev_failed)} known-OOM skipped" if prev_failed else "")
                  + f" from {checkpoint_path.name}")
        except Exception as e:
            print(f"  [resume] ignoring unreadable checkpoint ({e})")

    # Only Qwen2/2.5-VL accepts max_pixels; the base wrapper __init__ does not.
    wrapper_kwargs = {}
    if args.max_pixels is not None and "qwen2" in args.model.lower():
        wrapper_kwargs["max_pixels"] = args.max_pixels
        print(f"  [max_pixels] capping Qwen visual budget at {args.max_pixels} px")
    wrapper = create_wrapper(args.model, **wrapper_kwargs).load()
    base_iv = VTITextualIntervention(args.model, variant=args.variant,
                                     hook_site=HOOK_SITE)
    directions = base_iv.ensure_directions(wrapper)
    beta_max = max(args.betas)

    def _save_checkpoint(recs, failed, idx: int | None = None):
        payload = {"benchmark": bench, "model": model_short,
                   "betas": args.betas, "max_pixels": args.max_pixels,
                   "failed_ids": failed, "per_sample": recs}
        tmp = checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(checkpoint_path)
        # Human-readable sidecar (no caption text) for monitoring mid-run.
        progress_path = checkpoint_path.parent / checkpoint_path.name.replace(
            ".checkpoint.json", ".progress.json")
        gens_per = 1 + len(args.betas) + 2
        progress = {
            "benchmark": bench,
            "model": model_short,
            "run_date": run_date,
            "variant": args.variant,
            "hook_site": HOOK_SITE,
            "n_total": n,
            "n_completed": len(recs),
            "n_failed_oom": len(failed),
            "pct_complete": round(100 * len(recs) / n, 1) if n else 0.0,
            "last_sample_index": idx,
            "last_id": recs[-1]["id"] if recs else None,
            "betas": args.betas,
            "max_pixels": args.max_pixels,
            "generations_per_sample": gens_per,
            "generations_done_approx": len(recs) * gens_per,
            "generations_total_approx": n * gens_per,
            "failed_ids": failed,
            "checkpoint_file": checkpoint_path.name,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")

    # Generate (or reuse) one record per sample. `kept_samples` mirrors per_sample
    # (1:1, in order) so the column-wise arrays below stay aligned for scoring even
    # when some samples are dropped for OOM.
    per_sample = []
    kept_samples = []
    failed_ids = list(prev_failed)
    n_new = 0
    for i, sample in enumerate(samples, 1):
        if sample.id in prev_failed:
            continue
        cached = done_by_id.get(sample.id)
        if cached is not None:
            per_sample.append(cached)
            kept_samples.append(sample)
            continue
        try:
            baseline = _gen(wrapper, sample, args.max_new_tokens)
            rec = {"id": sample.id, "ground_truth": sample.ground_truth,
                   "category": (sample.metadata or {}).get("category"),
                   "baseline": baseline, "by_beta": {}}
            for b in args.betas:
                resp = _gen_steered(wrapper, sample, directions, args.variant, b,
                                    args.max_new_tokens)
                rec["by_beta"][f"{b}"] = {"response": resp,
                                          "class": _classify(resp, baseline)}
            rec["decode_only_beta_max"] = _gen_steered(
                wrapper, sample, directions, args.variant, beta_max,
                args.max_new_tokens, steer_prefill=False)
            rec["skip_pos0_beta_max"] = _gen_steered(
                wrapper, sample, directions, args.variant, beta_max,
                args.max_new_tokens, skip_first_token=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            failed_ids.append(sample.id)
            print(f"  [oom] skipping sample {sample.id} "
                  f"({type(e).__name__}); {len(failed_ids)} dropped so far", flush=True)
            _save_checkpoint(per_sample, failed_ids, idx=i)
            continue
        per_sample.append(rec)
        kept_samples.append(sample)
        n_new += 1
        if n_new % 25 == 0:
            print(f"  [{i}/{n}] ({n_new} new) done", flush=True)
            _save_checkpoint(per_sample, failed_ids, idx=i)
    _save_checkpoint(per_sample, failed_ids, idx=len(samples) if per_sample else None)
    n_scored = len(per_sample)
    if failed_ids:
        print(f"  [oom] {len(failed_ids)} sample(s) dropped for OOM; "
              f"scoring over {n_scored}/{n}")

    # Rebuild the column-wise arrays + collapse aggregates from per_sample.
    baseline_resps = [r["baseline"] for r in per_sample]
    resp_by_beta = {f"{b}": [r["by_beta"][f"{b}"]["response"] for r in per_sample]
                    for b in args.betas}
    decode_only_resps = [r["decode_only_beta_max"] for r in per_sample]
    skip_pos0_resps = [r["skip_pos0_beta_max"] for r in per_sample]
    agg = {f"{b}": {"empty": 0, "identical": 0, "changed": 0} for b in args.betas}
    changed_examples = {f"{b}": [] for b in args.betas}
    for r in per_sample:
        for b in args.betas:
            key = f"{b}"
            cls = r["by_beta"][key]["class"]
            agg[key][cls] += 1
            if cls == "changed" and len(changed_examples[key]) < 5:
                changed_examples[key].append(
                    {"id": r["id"], "baseline": r["baseline"],
                     "steered": r["by_beta"][key]["response"]})

    # Per-beta metric-vs-beta curves (+ collapse split; AMBER adds flip accounting).
    def _flips(resps):
        return {} if is_chair else _flip_counts(resps, baseline_resps, kept_samples)

    metrics_by_beta = {
        "baseline": {**score_fn(baseline_resps, kept_samples),
                     "empty": 0, "identical": n_scored, "changed": 0,
                     **({} if is_chair else dict(_ZERO_FLIPS))},
    }
    for b in args.betas:
        key = f"{b}"
        resps = resp_by_beta[key]
        metrics_by_beta[key] = {**score_fn(resps, kept_samples), **agg[key],
                                **_flips(resps)}

    probe_metrics = {
        "decode_only_beta_max": {**score_fn(decode_only_resps, kept_samples),
                                 **_flips(decode_only_resps)},
        "skip_pos0_beta_max": {**score_fn(skip_pos0_resps, kept_samples),
                               **_flips(skip_pos0_resps)},
    }

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n=== {bench} | {args.variant} @ {HOOK_SITE}  "
          f"(n={n_scored}/{n} scored) ===")
    if is_chair:
        print(f"{'beta':>8} {'chair_s':>8} {'chair_i':>8} {'avg_obj':>8} "
              f"{'avg_len':>8} {'mlen':>7} {'empty':>6} {'chng':>6}")
        for key in ["baseline"] + [f"{b}" for b in args.betas]:
            m = metrics_by_beta[key]
            print(f"{key:>8} {m['chair_s']:>8} {m['chair_i']:>8} "
                  f"{m['avg_objects_mentioned']:>8} {m['avg_caption_len_chars']:>8} "
                  f"{m['mean_len']:>7} {m.get('empty', 0):>6} {m.get('changed', 0):>6}")
    else:
        print(f"{'beta':>8} {'acc':>6} {'f1':>6} {'yes_r':>6} {'neg_acc':>8} "
              f"{'pos_acc':>8} {'unp':>4} {'empty':>6} {'h+':>4} {'h-':>4}")
        for key in ["baseline"] + [f"{b}" for b in args.betas]:
            m = metrics_by_beta[key]
            print(f"{key:>8} {m['accuracy']:>6} {m['f1']:>6} {m['yes_ratio']:>6} "
                  f"{m['neg_item_accuracy']:>8} {m['pos_item_accuracy']:>8} "
                  f"{m['n_unparsed']:>4} {m.get('empty', 0):>6} "
                  f"{m.get('flip_tn_to_fp', 0):>4} {m.get('flip_fp_to_tn', 0):>4}")
        print("  (h+ = induced hallucinations [tn->fp]; h- = removed [fp->tn])")

    # ── Sanity examples (~3): verify synonym map / gold-join alignment ─────────
    sanity = []
    if is_chair:
        from evaluation.classifiers.chair_objects import (
            parse_caption_objects, gt_objects_for_image)
        for s, base in list(zip(kept_samples, baseline_resps))[:3]:
            cid = (s.metadata or {}).get("raw", {}).get("coco_id")
            sanity.append({
                "id": s.id, "caption": base,
                "mentioned": sorted(parse_caption_objects(base)),
                "gt": sorted(gt_objects_for_image(cid)) if cid is not None else [],
            })
    else:
        from evaluation.classifiers.metrics import _normalize_yes_no
        for s, base in list(zip(kept_samples, baseline_resps))[:3]:
            sanity.append({
                "id": s.id, "response": base,
                "parsed": _normalize_yes_no(base),
                "gold": s.ground_truth,
                "dimension": (s.metadata or {}).get("category"),
            })

    save_json({
        "experiment": EXPERIMENT,
        "benchmark": bench,
        "model": model_short,
        "run_date": run_date,
        "variant": args.variant,
        "hook_site": HOOK_SITE,
        "n_samples": n,
        "n_scored": n_scored,
        "n_failed_oom": len(failed_ids),
        "failed_ids": failed_ids,
        "max_pixels": args.max_pixels,
        "betas": args.betas,
        "max_new_tokens": args.max_new_tokens,
        "chair_prompt": args.chair_prompt if is_chair else None,
        "subset_file": str(subset_file),
        "metrics_by_beta": metrics_by_beta,
        "mitigation_probes_beta_max": probe_metrics,
        "changed_examples_by_beta": changed_examples,
        "sanity_examples": sanity,
        "per_sample": per_sample,
    }, str(out_path))
    # Sweep complete: drop the per-sample checkpoint + progress sidecar.
    for stale in (checkpoint_path,
                  checkpoint_path.parent / checkpoint_path.name.replace(
                      ".checkpoint.json", ".progress.json")):
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
