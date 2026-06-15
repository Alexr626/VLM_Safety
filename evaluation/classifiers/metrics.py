"""Hallucination benchmark metrics (replaces ASR/refusal scoring)."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional


def _normalize_yes_no(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    if not t:
        return None
    if re.match(r"^(yes|yeah|yep|y)\b", t):
        return "yes"
    if re.match(r"^(no|nope|n)\b", t):
        return "no"
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t and "yes" not in t:
        return "no"
    return None


def score_pope_records(records: List[dict]) -> dict:
    correct = 0
    total = 0
    by_split: Dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        gt = _normalize_yes_no(r.get("ground_truth") or "")
        pred = _normalize_yes_no(r.get("response") or "")
        if gt is None:
            continue
        total += 1
        split = (r.get("metadata") or {}).get("category") or "all"
        by_split[split]["total"] += 1
        if pred == gt:
            correct += 1
            by_split[split]["correct"] += 1
    acc = correct / total if total else 0.0
    return {
        "metric": "accuracy",
        "accuracy_overall": acc,
        "n_correct": correct,
        "n_total": total,
        "by_category": {
            k: {"accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
                "n_total": v["total"]}
            for k, v in by_split.items()
        },
    }


def score_amber_records(records: List[dict]) -> dict:
    by_task: Dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        task = r.get("task") or "discriminative"
        gt = (r.get("ground_truth") or "").strip().lower()
        pred = (r.get("response") or "").strip().lower()
        if not gt:
            continue
        by_task[task]["total"] += 1
        if task == "discriminative":
            p = _normalize_yes_no(pred) or pred[:20]
            g = _normalize_yes_no(gt) or gt[:20]
            if p == g:
                by_task[task]["correct"] += 1
        else:
            if gt in pred or pred in gt:
                by_task[task]["correct"] += 1
    overall_correct = sum(v["correct"] for v in by_task.values())
    overall_total = sum(v["total"] for v in by_task.values())
    return {
        "metric": "task_accuracy",
        "accuracy_overall": overall_correct / overall_total if overall_total else 0.0,
        "n_correct": overall_correct,
        "n_total": overall_total,
        "by_task": {
            k: {"accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
                "n_total": v["total"]}
            for k, v in by_task.items()
        },
    }


def score_chair_records(records: List[dict]) -> dict:
    """Placeholder CHAIR scoring — full CHAIR-s/i needs COCO object inventory."""
    return {
        "metric": "chair_pending",
        "n_total": len(records),
        "note": "CHAIR-s/i requires post-hoc object inventory check against COCO annotations.",
    }


def score_hallusionbench_records(records: List[dict]) -> dict:
    correct = 0
    total = 0
    for r in records:
        gt = (r.get("ground_truth") or "").strip().lower()
        pred = (r.get("response") or "").strip().lower()
        if not gt:
            continue
        total += 1
        yn_gt = _normalize_yes_no(gt)
        yn_pred = _normalize_yes_no(pred)
        if yn_gt and yn_pred:
            if yn_gt == yn_pred:
                correct += 1
        elif gt in pred or pred in gt:
            correct += 1
    return {
        "metric": "accuracy",
        "accuracy_overall": correct / total if total else 0.0,
        "n_correct": correct,
        "n_total": total,
    }


def score_mmhal_records(records: List[dict]) -> dict:
    correct = 0
    total = 0
    by_category: Dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        gt = (r.get("ground_truth") or "").strip()
        pred = (r.get("response") or "").strip()
        if not gt:
            continue
        total += 1
        cat = (r.get("metadata") or {}).get("category") or "all"
        by_category[cat]["total"] += 1
        if gt.lower() in pred.lower() or pred.lower() in gt.lower():
            correct += 1
            by_category[cat]["correct"] += 1
    return {
        "metric": "reference_match",
        "accuracy_overall": correct / total if total else 0.0,
        "n_correct": correct,
        "n_total": total,
        "by_category": {
            k: {"accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
                "n_total": v["total"]}
            for k, v in by_category.items()
        },
    }


_BENCHMARK_SCORERS = {
    "pope": score_pope_records,
    "amber": score_amber_records,
    "chair": score_chair_records,
    "hallusionbench": score_hallusionbench_records,
    "mmhal_bench": score_mmhal_records,
}


def compute_metric_records(records: List[dict], benchmark: str) -> dict:
    scorer = _BENCHMARK_SCORERS.get(benchmark)
    if scorer is None:
        return {"metric": "unknown", "n_total": len(records)}
    # Enrich records with ground_truth/task from stored meta if needed
    enriched = []
    for r in records:
        rec = dict(r)
        if "ground_truth" not in rec and "safety_label" in rec:
            rec["ground_truth"] = rec["safety_label"]
        enriched.append(rec)
    return scorer(enriched)
