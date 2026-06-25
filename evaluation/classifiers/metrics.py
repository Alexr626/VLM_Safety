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


def _amber_disc_group():
    return {"correct": 0, "total": 0, "unparsed": 0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "yes_pred": 0,
            "neg_correct": 0, "neg_total": 0,
            "pos_correct": 0, "pos_total": 0}


def _amber_disc_finalize(g: dict) -> dict:
    total = g["total"]
    prf = _prf1(g["tp"], g["fp"], g["fn"])
    return {
        "accuracy": g["correct"] / total if total else 0.0,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "yes_ratio": g["yes_pred"] / total if total else 0.0,
        "neg_item_accuracy": (g["neg_correct"] / g["neg_total"]
                              if g["neg_total"] else 0.0),
        "pos_item_accuracy": (g["pos_correct"] / g["pos_total"]
                              if g["pos_total"] else 0.0),
        "n_correct": g["correct"],
        "n_total": total,
        "n_neg_total": g["neg_total"],
        "n_pos_total": g["pos_total"],
        "n_unparsed": g["unparsed"],
    }


def score_amber_discriminative_records(records: List[dict]) -> dict:
    """POPE-style scoring for AMBER's discriminative (yes/no) split.

    Reuses `_normalize_yes_no` (same parser as POPE, so the two benchmarks stay
    comparable) and the `yes`-positive convention (matches POPE; AMBER-official
    uses `no`-positive — we keep POPE's so AMBER/POPE read on the same axis, and
    `neg_item_accuracy` below captures AMBER-official's negative-detection signal
    regardless of P/R convention).

    The headline grounding-isolation pair is `yes_ratio` + `neg_item_accuracy`:
    an agreeableness (yes-drift) confound raises yes_ratio while neg_item_accuracy
    falls (saying "yes" to absent things); genuine grounding holds negatives at
    flat yes_ratio. `by_qtype` (existence/attribute/relation) localizes this.
    """
    overall = _amber_disc_group()
    by_q: Dict[str, dict] = defaultdict(_amber_disc_group)

    for r in records:
        gt = _normalize_yes_no(r.get("ground_truth") or "")
        if gt is None:
            continue  # malformed/absent gold; skip entirely (as POPE does)
        pred = _normalize_yes_no(r.get("response") or "")
        qtype = (r.get("metadata") or {}).get("category") or "unknown"
        groups = (overall, by_q[qtype])

        for g in groups:
            g["total"] += 1
            if gt == "yes":
                g["pos_total"] += 1
            else:
                g["neg_total"] += 1

        if pred is None:
            # Unparseable answer: incorrect for accuracy; FN if gold is yes,
            # uncounted (not FP) if gold is no — identical to the POPE scorer.
            for g in groups:
                g["unparsed"] += 1
                if gt == "yes":
                    g["fn"] += 1
            continue

        is_correct = pred == gt
        for g in groups:
            if pred == "yes":
                g["yes_pred"] += 1
            if is_correct:
                g["correct"] += 1
                if gt == "yes":
                    g["pos_correct"] += 1
                else:
                    g["neg_correct"] += 1
            if gt == "yes" and pred == "yes":
                g["tp"] += 1
            elif gt == "no" and pred == "yes":
                g["fp"] += 1
            elif gt == "yes" and pred == "no":
                g["fn"] += 1
            elif gt == "no" and pred == "no":
                g["tn"] += 1

    fin = _amber_disc_finalize(overall)
    return {
        "metric": "amber_discriminative",
        "positive_class": "yes",
        "metric_convention": "pope",  # positive=yes; AMBER-official uses positive=no
        "accuracy_overall": fin["accuracy"],
        "precision_overall": fin["precision"],
        "recall_overall": fin["recall"],
        "f1_overall": fin["f1"],
        "yes_ratio": fin["yes_ratio"],
        "n_correct": fin["n_correct"],
        "n_total": fin["n_total"],
        "n_unparsed": fin["n_unparsed"],
        "neg_item_accuracy": fin["neg_item_accuracy"],
        "pos_item_accuracy": fin["pos_item_accuracy"],
        "n_neg_total": fin["n_neg_total"],
        "n_pos_total": fin["n_pos_total"],
        "by_qtype": {q: _amber_disc_finalize(g) for q, g in sorted(by_q.items())},
    }


def _score_amber_generative(records: List[dict]) -> dict:
    """Existing (out-of-scope) generative AMBER scoring, kept unchanged.

    NOTE: AMBER generative gold is NOT a yes/no label and is not surfaced into
    `ground_truth` by the loader, so this branch is a placeholder; AMBER
    generative hallucination-rate scoring is a separate, future task.
    """
    by_task: Dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        gt = (r.get("ground_truth") or "").strip().lower()
        pred = (r.get("response") or "").strip().lower()
        if not gt:
            continue
        by_task["generative"]["total"] += 1
        if gt in pred or pred in gt:
            by_task["generative"]["correct"] += 1
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


def score_amber_records(records: List[dict]) -> dict:
    """Dispatch AMBER scoring by task.

    Discriminative -> the enriched POPE-style yes-bias/grounding breakdown.
    Generative -> the existing placeholder (out of scope here). Runs are normally
    single-task via `--amber_task`; for a mixed record set the discriminative
    summary is returned with the generative summary nested under `generative`.
    """
    disc = [r for r in records if (r.get("task") or "discriminative") == "discriminative"]
    gen = [r for r in records if r.get("task") == "generative"]
    if disc and not gen:
        return score_amber_discriminative_records(disc)
    if gen and not disc:
        return _score_amber_generative(gen)
    summary = score_amber_discriminative_records(disc) if disc else {
        "metric": "amber_discriminative", "n_total": 0}
    if gen:
        summary["generative"] = _score_amber_generative(gen)
    return summary


def score_chair_records(records: List[dict]) -> dict:
    """Rule-based CHAIR-s / CHAIR-i against COCO val2014 instance annotations.

    Per caption: mentioned = COCO objects named in the caption (synonym map);
    gt = COCO objects actually in the image; hallucinated = mentioned - gt.

      CHAIR_i = sum(|hallucinated|) / sum(|mentioned|)   (instance level)
      CHAIR_s = (#captions with >=1 hallucination) / (#captions)  (sentence level)

    LENGTH/COVERAGE CONTROLS (do not drop): `avg_objects_mentioned` and
    `avg_caption_len_chars` ship in every summary because CHAIR confounds with
    caption length in BOTH directions (terse captions game CHAIR_s down; long
    captions inflate object counts). For this project the rotation intervention
    changes caption length per-model with opposite sign, so CHAIR_i/s must be
    read JOINTLY with these two columns, never in isolation, and `max_new_tokens`
    must be fixed across baseline and interventions (the runner freezes it).

    EMPTY-CAPTION HANDLING (§C, layer-rotation collapse regime): at high steering
    strength the model can emit an empty / whitespace-only caption. CHAIR_i's
    denominator is sum(|mentioned|), so empties drive it toward 0/0 and make a
    collapsed run look hallucination-free. We therefore treat empty captions as
    UNDEFINED for CHAIR: `chair_i`, `chair_s`, and `avg_objects_mentioned` are
    computed over NON-EMPTY captions only (`n_nonempty`), while `n_empty` /
    `empty_fraction` are reported separately so "low CHAIR" can never silently
    mean "the model said nothing." `avg_caption_len_chars` is kept over ALL
    scoreable captions (incl. empties) precisely so the length collapse stays
    visible in that column.
    """
    from evaluation.classifiers.chair_objects import (
        parse_caption_objects, gt_objects_for_image)

    n_total = 0
    n_empty = 0
    n_missing_image_id = 0
    n_caps_hallucinated = 0
    sum_mentioned = 0
    sum_hallucinated = 0
    sum_caption_len = 0
    for r in records:
        caption = r.get("response") or ""
        meta = r.get("metadata") or {}
        raw = meta.get("raw") or {}
        image_id = raw.get("coco_id")
        if image_id is None:
            n_missing_image_id += 1
            continue
        n_total += 1
        sum_caption_len += len(caption)          # over ALL scoreable (shows collapse)
        if not caption.strip():
            n_empty += 1
            continue                              # empty -> undefined for CHAIR
        mentioned = parse_caption_objects(caption)
        gt = gt_objects_for_image(image_id)
        hallucinated = mentioned - gt
        sum_mentioned += len(mentioned)
        sum_hallucinated += len(hallucinated)
        if hallucinated:
            n_caps_hallucinated += 1

    n_nonempty = n_total - n_empty
    chair_i = sum_hallucinated / sum_mentioned if sum_mentioned else 0.0
    chair_s = n_caps_hallucinated / n_nonempty if n_nonempty else 0.0
    return {
        "metric": "chair",
        "chair_i": chair_i,           # lower is better (over non-empty)
        "chair_s": chair_s,           # lower is better (over non-empty)
        "n_total": n_total,
        "n_nonempty": n_nonempty,
        "n_empty": n_empty,
        "empty_fraction": (n_empty / n_total) if n_total else 0.0,
        "n_missing_image_id": n_missing_image_id,
        "avg_objects_mentioned": (sum_mentioned / n_nonempty) if n_nonempty else 0.0,
        "avg_caption_len_chars": (sum_caption_len / n_total) if n_total else 0.0,
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


def score_mmhal_records(records: List[dict], *, judge) -> dict:
    """Judge-based MMHal-Bench scoring (official 0-6 rating + hallucination flag).

    For each record the official MMHal judge prompt is assembled from the stored
    question / model response / gt answer / image-content description / question
    type, sent to ``judge`` (any provider), and parsed to a 0-6 rating. The
    8-way ``by_category`` breakdown is keyed by the MMHal *question type* (the
    record's ``task``: attribute/adversarial/comparison/counting/relation/
    environment/holistic/other) — this is what isolates the COUNTING subset.

    Caveats baked in as summary fields: n is tiny (96 total, ~12/category), so a
    per-category number is a directional probe, not a claim — every category
    carries its own ``n_total`` so the ~12 is always visible. ``judge_name`` is
    stamped in so the result is self-describing (absolute scores are only
    comparable across runs judged by the same model). Parse failures are surfaced
    as ``n_unparsed_judge`` (NOT scored 0, unlike the official script) so a judge
    that failed to emit a rating does not deflate ``avg_score``.
    """
    from evaluation.classifiers.judges import (
        build_mmhal_prompt, parse_mmhal_rating, rating_to_hallucination)

    n_total = 0
    n_unparsed = 0
    sum_score = 0
    n_hallucinated = 0
    by_category: Dict[str, dict] = defaultdict(
        lambda: {"sum_score": 0, "n_parsed": 0, "n_hallucinated": 0, "n_total": 0})

    for r in records:
        gt_answer = (r.get("ground_truth") or "").strip()
        if not gt_answer:
            continue
        n_total += 1
        meta = r.get("metadata") or {}
        raw = meta.get("raw") or {}
        # Question type (the 8 MMHal categories incl. 'counting'); the loader's
        # `category` field is the question *topic* (outdoor/indoor/...), not this.
        cat = r.get("task") or raw.get("question_type") or "unknown"
        image_content_list = raw.get("image_content") or []
        image_content = ", ".join(image_content_list) if isinstance(
            image_content_list, list) else str(image_content_list)
        prompt = build_mmhal_prompt(
            image_content=image_content,
            question=r.get("question") or raw.get("question") or "",
            gt_answer=gt_answer,
            model_response=(r.get("response") or "").strip(),
        )
        c = by_category[cat]
        c["n_total"] += 1
        judge_text = judge.score(prompt)
        rating = parse_mmhal_rating(judge_text)
        if rating is None:
            n_unparsed += 1
            continue
        hal = rating_to_hallucination(rating)
        sum_score += rating
        n_hallucinated += hal
        c["sum_score"] += rating
        c["n_parsed"] += 1
        c["n_hallucinated"] += hal

    n_parsed = n_total - n_unparsed
    avg_score = sum_score / n_parsed if n_parsed else 0.0
    hallucination_rate = n_hallucinated / n_parsed if n_parsed else 0.0
    return {
        "metric": "mmhal_judge",
        "judge_name": getattr(judge, "name", "unknown"),
        "avg_score": avg_score,                 # mean 0-6 over parsed; headline
        "hallucination_rate": hallucination_rate,
        "n_total": n_total,
        "n_unparsed_judge": n_unparsed,
        "by_category": {
            k: {
                "avg_score": (v["sum_score"] / v["n_parsed"]) if v["n_parsed"] else 0.0,
                "hallucination_rate": (v["n_hallucinated"] / v["n_parsed"]) if v["n_parsed"] else 0.0,
                "n_total": v["n_total"],
            }
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


def compute_metric_records(records: List[dict], benchmark: str,
                           *, judge=None) -> dict:
    """Dispatch to the per-benchmark scorer.

    ``judge`` is consumed only by ``mmhal_bench`` (the judge-based scorer); all
    other benchmarks ignore it. If MMHal is scored without a judge we fall back
    to the offline mock judge so a dry run never makes network calls.
    """
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
    if benchmark == "mmhal_bench":
        if judge is None:
            from evaluation.classifiers.judges import get_judge
            judge = get_judge("mock")
        return scorer(enriched, judge=judge)
    return scorer(enriched)
