#!/usr/bin/env python3
"""VTI rotation-strength experiment (layer-hook site).

Characterizes how the textual steering strength beta modulates a layer-site
rotation variant (`uniform_rotation` / `gated_rotation`) across N POPE samples.
For each beta it classifies every steered response relative to the true
no-intervention baseline (no hooks):

  - empty      : model emitted EOS immediately (degenerate collapse)
  - identical  : byte-identical to the no-hook baseline (steering inert here)
  - changed    : non-empty and different from baseline (steering had an effect)

and reports POPE accuracy / yes_ratio / F1 / mean length plus decision-flip
accounting vs the no-hook baseline (n_decision_flips, correct->wrong,
wrong->correct). Two mitigation probes are run at the strongest beta:
decode-only steering (`steer_prefill=False`) and leaving sequence position 0
(BOS / attention sink) unsteered during prefill (`skip_first_token=True`).

Naming: `beta` is the textual steering coefficient (paper's notation; the
vision arm uses `alpha`). It is passed into the geometry-agnostic
`steer()` / `vti_hook_ctx(alpha=...)` coefficient.

Results land under
  evaluation/vti_rotation_strength/results/{run_date}/{model_short}/
      sweep_{variant}_{site}_{split}_n{N}.json

Usage:
  CUDA_VISIBLE_DEVICES=0 \\
    python evaluation/vti_rotation_strength/rotation_strength.py --num_samples 30
  python evaluation/vti_rotation_strength/rotation_strength.py \\
    --variant gated_rotation --num_samples 30 --betas 0.9 0.5 0.3 0.1 0.05
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import create_wrapper, _normalize_model_name  # noqa: E402
from src.extraction import save_json  # noqa: E402
from evaluation.benchmarks import load_pope_eval  # noqa: E402
from evaluation.classifiers.metrics import (  # noqa: E402
    _normalize_yes_no, score_pope_records,
)
from evaluation.interventions.vti import VTITextualIntervention  # noqa: E402
from evaluation.interventions.vti.hooks import vti_hook_ctx  # noqa: E402

EXPERIMENT = "vti_rotation_strength"
HOOK_SITE = "layer"


def results_dir(run_date: str, model_short: str) -> Path:
    return (_PROJECT_ROOT / "evaluation" / EXPERIMENT / "results"
            / run_date / model_short)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("MODEL", "llava-hf/llava-1.5-7b-hf"))
    p.add_argument("--variant", default=os.environ.get("VARIANT", "uniform_rotation"),
                   choices=["uniform_rotation", "gated_rotation", "additive"])
    p.add_argument("--num_samples", type=int,
                   default=int(os.environ.get("NUM_SAMPLES", "30")))
    p.add_argument("--pope_split", default="random")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--betas", type=float, nargs="+",
                   default=[0.9, 0.5, 0.3, 0.1, 0.05, 0.01],
                   help="Textual steering coefficients to sweep.")
    p.add_argument("--run_date", default=None,
                   help="Results subdir (YYYY-MM-DD); defaults to today.")
    return p.parse_args()


def _gen(wrapper, sample, max_new):
    return wrapper.generate_vl(sample.image, sample.question, max_new_tokens=max_new)


def _gen_steered(wrapper, sample, directions, variant, beta, max_new,
                 steer_prefill=True, skip_first_token=False):
    with vti_hook_ctx(
        wrapper, directions,
        variant=variant, alpha=beta, hook_site=HOOK_SITE,
        steer_prefill=steer_prefill, skip_first_token=skip_first_token,
    ):
        return _gen(wrapper, sample, max_new)


def _classify(resp, baseline):
    if len(resp.strip()) == 0:
        return "empty"
    if resp == baseline:
        return "identical"
    return "changed"


def _score(responses, samples):
    """POPE accuracy / yes_ratio / mean_len for a list of responses."""
    records = [
        {"response": r, "ground_truth": s.ground_truth,
         "metadata": {"category": (s.metadata or {}).get("category")}}
        for r, s in zip(responses, samples)
    ]
    m = score_pope_records(records)
    mean_len = sum(len(r) for r in responses) / len(responses) if responses else 0.0
    return {
        "accuracy": round(m["accuracy_overall"], 4),
        "precision": round(m["precision_overall"], 4),
        "recall": round(m["recall_overall"], 4),
        "f1": round(m["f1_overall"], 4),
        "yes_ratio": round(m["yes_ratio"], 4),
        "n_unparsed": m["n_unparsed"],
        "mean_len": round(mean_len, 1),
    }


# Zeroed flip-accounting block (also the baseline row's value).
_ZERO_FLIPS = {
    "n_decision_flips": 0,
    "n_flip_correct_to_wrong": 0,
    "n_flip_wrong_to_correct": 0,
    "flip_tp_to_fn": 0,   # correct yes -> wrong no  (suppressed a true detection)
    "flip_tn_to_fp": 0,   # correct no  -> wrong yes (INDUCED a hallucination)
    "flip_fn_to_tp": 0,   # wrong no    -> correct yes (recovered a missed object)
    "flip_fp_to_tn": 0,   # wrong yes   -> correct no  (REMOVED a hallucination)
}


def _flip_counts(steered_responses, baseline_responses, samples):
    """Decision-flip accounting vs the no-hook baseline (per sample).

    Decomposes each correctness flip by confusion-matrix direction (positive =
    yes), so suppression (true yes -> no) is distinguished from hallucination
    removal/induction (no/yes errors), per answers/concepts/jun_19_2026.
    """
    out = dict(_ZERO_FLIPS)
    for resp, base, s in zip(steered_responses, baseline_responses, samples):
        gt = _normalize_yes_no(s.ground_truth or "")
        bl = _normalize_yes_no(base)
        st = _normalize_yes_no(resp)
        if bl is None or st is None or bl == st:
            continue
        out["n_decision_flips"] += 1
        if gt is None:
            continue
        if bl == gt and st != gt:
            out["n_flip_correct_to_wrong"] += 1
            if gt == "yes":
                out["flip_tp_to_fn"] += 1
            else:
                out["flip_tn_to_fp"] += 1
        elif bl != gt and st == gt:
            out["n_flip_wrong_to_correct"] += 1
            if gt == "yes":
                out["flip_fn_to_tp"] += 1
            else:
                out["flip_fp_to_tn"] += 1
    return out


def main():
    args = parse_args()
    run_date = args.run_date or datetime.now().strftime("%Y-%m-%d")
    model_short = _normalize_model_name(args.model)

    wrapper = create_wrapper(args.model).load()
    samples = load_pope_eval(split=args.pope_split, limit=args.num_samples)
    print(f"\nmodel={args.model}  variant={args.variant}  site={HOOK_SITE}")
    print(f"samples={len(samples)}  split={args.pope_split}  "
          f"run_date={run_date}  betas={args.betas}\n")

    base_iv = VTITextualIntervention(args.model, variant=args.variant,
                                     hook_site=HOOK_SITE)
    directions = base_iv.ensure_directions(wrapper)

    beta_max = max(args.betas)
    # Collect responses column-wise so we can score whole sets per beta.
    baseline_resps = []
    resp_by_beta = {f"{b}": [] for b in args.betas}
    decode_only_resps = []
    skip_pos0_resps = []
    per_sample = []
    agg = {f"{b}": {"empty": 0, "identical": 0, "changed": 0} for b in args.betas}
    changed_examples = {f"{b}": [] for b in args.betas}

    for i, sample in enumerate(samples, 1):
        baseline = _gen(wrapper, sample, args.max_new_tokens)
        baseline_resps.append(baseline)
        rec = {"id": sample.id, "question": sample.question,
               "ground_truth": sample.ground_truth,
               "baseline": baseline, "by_beta": {}}
        for b in args.betas:
            resp = _gen_steered(wrapper, sample, directions, args.variant, b,
                                args.max_new_tokens)
            resp_by_beta[f"{b}"].append(resp)
            cls = _classify(resp, baseline)
            agg[f"{b}"][cls] += 1
            rec["by_beta"][f"{b}"] = {"response": resp, "class": cls}
            if cls == "changed" and len(changed_examples[f"{b}"]) < 5:
                changed_examples[f"{b}"].append(
                    {"id": sample.id, "baseline": baseline, "steered": resp})

        # Mitigation probes at the strongest beta.
        r_decode = _gen_steered(wrapper, sample, directions, args.variant,
                                beta_max, args.max_new_tokens, steer_prefill=False)
        r_skip0 = _gen_steered(wrapper, sample, directions, args.variant,
                               beta_max, args.max_new_tokens, skip_first_token=True)
        decode_only_resps.append(r_decode)
        skip_pos0_resps.append(r_skip0)
        rec["decode_only_beta_max"] = r_decode
        rec["skip_pos0_beta_max"] = r_skip0
        per_sample.append(rec)
        if i % 10 == 0:
            print(f"  [{i}/{len(samples)}] done")

    n = len(samples)

    # Per-beta scoring (accuracy / yes_ratio / mean_len / flips).
    metrics_by_beta = {"baseline": {**_score(baseline_resps, samples),
                                    "empty": 0, "identical": n, "changed": 0,
                                    **_ZERO_FLIPS}}
    for b in args.betas:
        key = f"{b}"
        resps = resp_by_beta[key]
        metrics_by_beta[key] = {
            **_score(resps, samples),
            **agg[key],
            **_flip_counts(resps, baseline_resps, samples),
        }
    probe_metrics = {
        "decode_only_beta_max": {
            **_score(decode_only_resps, samples),
            **_flip_counts(decode_only_resps, baseline_resps, samples),
        },
        "skip_pos0_beta_max": {
            **_score(skip_pos0_resps, samples),
            **_flip_counts(skip_pos0_resps, baseline_resps, samples),
        },
    }

    print(f"\n=== {args.variant} @ {HOOK_SITE}  (n={n}) ===")
    print("  (h+ = hallucinations induced [tn->fp];  h- = hallucinations removed [fp->tn])")
    hdr = (f"{'beta':>9} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} "
           f"{'yes_r':>6} {'mlen':>6} {'empty':>6} {'flips':>6} "
           f"{'c->w':>5} {'w->c':>5} {'h+':>4} {'h-':>4}")
    print(hdr)
    for key in ["baseline"] + [f"{b}" for b in args.betas]:
        m = metrics_by_beta[key]
        print(f"{key:>9} {m['accuracy']:>6} {m['precision']:>6} {m['recall']:>6} "
              f"{m['f1']:>6} {m['yes_ratio']:>6} {m['mean_len']:>6} "
              f"{m['empty']:>6} {m['n_decision_flips']:>6} "
              f"{m['n_flip_correct_to_wrong']:>5} {m['n_flip_wrong_to_correct']:>5} "
              f"{m['flip_tn_to_fp']:>4} {m['flip_fp_to_tn']:>4}")
    print(f"\nMitigation probes at beta={beta_max}:")
    for k, m in probe_metrics.items():
        print(f"  {k:>22}: acc={m['accuracy']} prec={m['precision']} "
              f"rec={m['recall']} f1={m['f1']} yes_ratio={m['yes_ratio']} "
              f"mlen={m['mean_len']} flips={m['n_decision_flips']} "
              f"c->w={m['n_flip_correct_to_wrong']} w->c={m['n_flip_wrong_to_correct']} "
              f"h+={m['flip_tn_to_fp']} h-={m['flip_fp_to_tn']}")

    out_dir = results_dir(run_date, model_short)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sweep_{args.variant}_{HOOK_SITE}_{args.pope_split}_n{n}.json"
    save_json({
        "experiment": EXPERIMENT,
        "model": model_short,
        "run_date": run_date,
        "variant": args.variant,
        "hook_site": HOOK_SITE,
        "pope_split": args.pope_split,
        "n_samples": n,
        "betas": args.betas,
        "max_new_tokens": args.max_new_tokens,
        "metrics_by_beta": metrics_by_beta,
        "mitigation_probes_beta_max": probe_metrics,
        "changed_examples_by_beta": changed_examples,
        "per_sample": per_sample,
    }, str(out_path))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
