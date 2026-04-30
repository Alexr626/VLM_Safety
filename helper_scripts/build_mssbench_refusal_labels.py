#!/usr/bin/env python3
"""
Build MSSBench Refusal Labels (diagnostic-side bridge)
=======================================================
Eval pipeline produces `evaluation/results/{model}/mssbench/{intervention}/responses.json`
with an `is_refusal: bool` field per record (computed via the canonical ShiftDC
keyword classifier). The diagnostic-side `safety_probes.py` consumes refusal
labels in a different shape — keyed by sample id, with the same flat boolean
schema written by `classify_responses.py --method keyword`.

This script bridges them: filter the eval responses to the MSSBench eval split
and rewrite as a per-sample-id list compatible with `_read_refusal_key`.

Output schema (matches the `keyword`-method output of classify_responses.py):

    [
      {"id": "mssbench_0006_SSS_12_q0", "label": "SSS",
       "category": "harmful", "method": "keyword", "refused_vl": false},
      ...
    ]

Output path:
    diagnostic_experiments/{model_short}/behavioral_ground_truth/outputs/results/mssbench_refusal_labels.json

Usage
-----
    python helper_scripts/build_mssbench_refusal_labels.py \\
        --model llava-hf/llava-1.5-7b-hf

    # Use a different intervention's responses (e.g. comp_safety_shift_mssbench_vl/):
    python helper_scripts/build_mssbench_refusal_labels.py \\
        --model llava-hf/llava-1.5-7b-hf --intervention comp_safety_shift_mssbench_vl

    # All eval responses, not just the eval-split (rare; usually only useful
    # when comparing across the full benchmark):
    python helper_scripts/build_mssbench_refusal_labels.py \\
        --model llava-hf/llava-1.5-7b-hf --no_eval_only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import _normalize_model_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf",
                   help="HuggingFace model id (used to resolve the eval results dir).")
    p.add_argument("--intervention", default="vanilla",
                   help="Subdirectory under evaluation/results/{model}/mssbench/. "
                        "Default 'vanilla' — the no-intervention responses produced "
                        "by the full eval pipeline are the natural source of "
                        "behavioral refusal labels.")
    p.add_argument("--results_root",
                   default=str(_PROJECT_ROOT / "evaluation" / "results"),
                   help="Root of evaluation/results/. Override only for non-default "
                        "result locations.")
    p.add_argument("--mssbench_split_path",
                   default=str(_PROJECT_ROOT / "data" / "mssbench" / "train_eval_split.json"),
                   help="Path to the MSSBench train/eval split JSON. Required when "
                        "--no_eval_only is NOT set.")
    p.add_argument("--no_eval_only", action="store_true",
                   help="Don't filter to the eval split; emit labels for every "
                        "response in responses.json.")
    p.add_argument("--skip_if_exists", action="store_true",
                   help="No-op if the output file already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_short = _normalize_model_name(args.model)

    responses_path = (Path(args.results_root) / model_short / "mssbench"
                      / args.intervention / "responses.json")
    if not responses_path.exists():
        print(f"ERROR: {responses_path} not found.\n"
              f"Run: python evaluation/run_eval.py --model {args.model} "
              f"--interventions {args.intervention} --benchmarks mssbench")
        sys.exit(1)

    with open(responses_path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} responses from {responses_path}")

    if args.no_eval_only:
        eval_ids = None
    else:
        split_path = Path(args.mssbench_split_path)
        if not split_path.exists():
            print(f"ERROR: {split_path} not found.\n"
                  "Run: python -m src.dataset --mssbench_split")
            sys.exit(1)
        with open(split_path) as f:
            split = json.load(f)
        eval_ids = set(split.get("eval_sample_ids") or [])
        if not eval_ids:
            print(f"ERROR: {split_path} has no eval_sample_ids.")
            sys.exit(1)
        print(f"Filtering to {len(eval_ids)} MSSBench eval-split sample ids.")

    out_path = (_PROJECT_ROOT / "diagnostic_experiments" / model_short
                / "behavioral_ground_truth" / "outputs" / "results"
                / "mssbench_refusal_labels.json")
    if args.skip_if_exists and out_path.exists():
        print(f"Skipping — {out_path} already exists.")
        return

    labels: list[dict] = []
    seen_ids: set[str] = set()
    for r in records:
        sid = r.get("id")
        if not sid or sid in seen_ids:
            continue
        if eval_ids is not None and sid not in eval_ids:
            continue
        seen_ids.add(sid)
        labels.append({
            "id": sid,
            "label": r.get("safety_label"),       # "SSS" or "SSU"
            "category": r.get("scenario_name"),   # MSSBench Type field
            "method": "keyword",
            # The eval pipeline's vanilla intervention generates VL responses
            # only — there's no TT/CT counterpart at the eval-side. The probe
            # uses `refused_vl` as the target regardless of which activation
            # modality is on the input side.
            "refused_vl": bool(r.get("is_refusal", False)),
            "intervention": args.intervention,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved -> {out_path}  ({len(labels)} samples)")

    refused = sum(1 for l in labels if l["refused_vl"])
    by_label: dict[str, list[int]] = {"SSS": [0, 0], "SSU": [0, 0]}
    for l in labels:
        bucket = by_label.setdefault(l["label"] or "?", [0, 0])
        bucket[0] += 1
        if l["refused_vl"]:
            bucket[1] += 1
    print(f"\nRefusal-rate summary (intervention={args.intervention}):")
    for k in sorted(by_label):
        n, n_ref = by_label[k]
        if n:
            print(f"  {k}: refused={n_ref}/{n} ({n_ref/n:.1%})")
    print(f"  Overall: refused={refused}/{len(labels)} "
          f"({refused/len(labels):.1%})" if labels else "")


if __name__ == "__main__":
    main()
