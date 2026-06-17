"""Hallucination benchmark metrics (replaces ASR/refusal scoring)."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional


def _normalize_yes_no(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    if not t:
        return None
    # POPE convention: the answer is the leading token. Authoritative.
    if re.match(r"^\s*yes\b", t):
        return "yes"
    if re.match(r"^\s*no\b", t):
        return "no"
    # Fallback for preambled answers ("There is no ..."). Word-boundaried,
    # so "not"/"cannot"/"none"/"known" do NOT spuriously match "no".
    has_yes = re.search(r"\byes\b", t) is not None
    has_no = re.search(r"\bno\b", t) is not None
    if has_no and not has_yes:
        return "no"
    if has_yes and not has_no:
        return "yes"
    return None  # ambiguous / unparseable


def _prf1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def score_pope_records(records: List[dict]) -> dict:
    # Positive class = "yes" (POPE convention, matches VTI's reporting).
    correct = total = unparsed = 0
    tp = fp = fn = tn = 0
    by_split: Dict[str, dict] = defaultdict(
        lambda: {"correct": 0, "total": 0, "unparsed": 0,
                 "tp": 0, "fp": 0, "fn": 0, "tn": 0})

    for r in records:
        gt = _normalize_yes_no(r.get("ground_truth") or "")
        if gt is None:
            continue  # malformed label; skip entirely
        pred = _normalize_yes_no(r.get("response") or "")
        split = (r.get("metadata") or {}).get("category") or "all"
        s = by_split[split]
        total += 1
        s["total"] += 1

        if pred is None:
            # Unparseable model answer: count as incorrect (conservative,
            # matches field convention) but track separately. For P/R/F1,
            # an unparsed answer to a "yes" gt is a missed positive (fn);
            # to a "no" gt it is not a false positive (model didn't say yes).
            unparsed += 1
            s["unparsed"] += 1
            if gt == "yes":
                fn += 1; s["fn"] += 1
            # gt == "no" with no pred -> neither tp/fp/fn (stays uncounted in P,
            # correctly penalized only in accuracy)
            continue

        if pred == gt:
            correct += 1
            s["correct"] += 1
        # confusion matrix (positive = yes)
        if gt == "yes" and pred == "yes":
            tp += 1; s["tp"] += 1
        elif gt == "no" and pred == "yes":
            fp += 1; s["fp"] += 1
        elif gt == "yes" and pred == "no":
            fn += 1; s["fn"] += 1
        elif gt == "no" and pred == "no":
            tn += 1; s["tn"] += 1

    acc = correct / total if total else 0.0
    prf = _prf1(tp, fp, fn)
    n_yes_pred = tp + fp
    yes_ratio = n_yes_pred / total if total else 0.0

    by_category = {}
    for k, v in by_split.items():
        vacc = v["correct"] / v["total"] if v["total"] else 0.0
        vprf = _prf1(v["tp"], v["fp"], v["fn"])
        by_category[k] = {
            "accuracy": vacc,
            "precision": vprf["precision"],
            "recall": vprf["recall"],
            "f1": vprf["f1"],
            "n_total": v["total"],
            "n_unparsed": v["unparsed"],
            "yes_ratio": ((v["tp"] + v["fp"]) / v["total"]
                          if v["total"] else 0.0),
        }

    return {
        "metric": "accuracy",
        "accuracy_overall": acc,
        "precision_overall": prf["precision"],
        "recall_overall": prf["recall"],
        "f1_overall": prf["f1"],
        "yes_ratio": yes_ratio,
        "n_correct": correct,
        "n_total": total,
        "n_unparsed": unparsed,
        "by_category": by_split and by_category,
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
