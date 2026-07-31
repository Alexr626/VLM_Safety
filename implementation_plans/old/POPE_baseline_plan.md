# Experiment Plan — POPE Baselines (VTI-comparable), 3 Models

**Goal:** produce no-intervention POPE baselines for LLaVA-1.5, Qwen-VL-Chat, and Qwen2-VL/2.5 that are directly comparable to the VTI paper (Liu et al., ICLR 2025) — same metric set (accuracy, precision, recall, F1), same elicitation (free generation), same query template.

**Status:** Change 1 (metrics parser/scorer) already applied by Romanus. Remaining: confirm Change 1 matches the spec below, apply Changes 2–4.

**Today's run is accuracy/scoring only.** CPU offload / multi-GPU is acceptable today — it does NOT affect generated answer text or scoring. (Single-device is required only later, for activation extraction; not in scope here.)

---

## Change 1 — `evaluation/classifiers/metrics.py` (verify; reportedly already done)

Confirm `_normalize_yes_no`, `_prf1`, and `score_pope_records` match the following. The parser must prioritize the **leading token** (POPE convention), with a **word-boundaried** fallback so `not` / `cannot` / `none` do NOT match `no`.

```python
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
    yes_ratio = (tp + fp) / total if total else 0.0

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
        "by_category": by_category,
    }
```

**Convention to document:** unparseable response counts as incorrect for accuracy; for P/R/F1 it is a false negative when ground truth is `yes`, and uncounted when ground truth is `no` (the model never falsely affirmed). `n_unparsed` is the alarm — if it is near zero this choice is immaterial; if it is large, accuracy is suspect.

**Other scorers:** `score_hallusionbench_records` and `score_amber_records` also call `_normalize_yes_no`; the new version is strictly safer for them. No further change required there.

---

## Change 2 — `src/model.py` (Policy A: native resolution)

In `Qwen2VLWrapper.__init__`, change the `_max_pixels` default so the processor uses its **native default** instead of the 512-token cap:

```python
# BEFORE
self._max_pixels = max_pixels if max_pixels is not None else 512 * 28 * 28
# AFTER
self._max_pixels = max_pixels   # None -> processor's native default
```

- Keep the `max_pixels` kwarg so it can be constrained later if needed.
- Do NOT change LLaVA (already native) or Qwen-VL-Chat (architecturally fixed at 448×448).
- Leave all offload / `device_map` / `max_memory` logic AS-IS — acceptable for today's accuracy run.

---

## Change 3 — surface the new metrics

### 3a. `evaluation/eval_runner.py` — no change needed
The summary already spreads `**metrics`, so `f1_overall`, `precision_overall`, `recall_overall`, `n_unparsed`, `yes_ratio` flow into `metric_summary.json` automatically. Confirm this is the case.

### 3b. `evaluation/runners.py` (or wherever `print_comparison_table` lives) — show F1
The POPE column in `_TABLE_COLUMNS` currently shows accuracy only:

```python
("POPE", "pope", lambda s: s.get("accuracy_overall")),
```

Change it to display `acc/f1` (the string-formatting branch already handles non-float cells):

```python
("POPE", "pope",
 lambda s: f"{s.get('accuracy_overall', 0) * 100:.1f}/{s.get('f1_overall', 0) * 100:.1f}"),
```

(Header can stay "POPE"; the cell now reads e.g. `80.3/79.1` = acc/F1.)

---

## Change 4 — run invocation (3 splits × 200 samples, greedy)

```bash
for split in random popular adversarial; do
  python evaluation/run_eval.py \
    --model MODEL_ID \
    --benchmarks pope \
    --interventions no_intervention \
    --pope_split $split \
    --limit 200
done
```

Run for each of:
- `llava-hf/llava-1.5-7b-hf`
- `Qwen/Qwen-VL-Chat`
- `Qwen/Qwen2-VL-7B-Instruct` (or `Qwen/Qwen2.5-VL-7B-Instruct`)

**Verify before trusting results:** the three splits write to the same
`pope/no_intervention/` directory. POPE sample ids are split-prefixed
(`pope_adversarial_00000`, etc.), so records should accumulate rather than
collide, and the final `metric_summary.json` should show all three splits under
`by_category`. **Confirm the third run's summary contains random + popular +
adversarial in `by_category`** (not just the last split). If they do NOT
accumulate cleanly, instead run each split to a separate `--output_dir` and
average the three summaries manually.

---

## Verification checklist (before reporting numbers)

1. `n_unparsed` per model is near zero. If not, accuracy is suspect — investigate before reporting.
2. Spot-check ~5 raw responses per model in `responses.json` against the recorded verdict.
3. `by_category` in the final summary shows all three splits (random, popular, adversarial).
4. POPE query template is verbatim `"Is there a [object] in the image?"` (matches VTI). The loader appears to pull `raw.text` straight from the official POPE JSON — confirm it is unmodified.
5. Report per-split AND averaged acc/precision/recall/F1 (VTI averages the three splits).

---

## RESEARCH_LOG.md entry (Romanus to append)

Per model record: commit hash, exact command(s), per-split and averaged
acc/precision/recall/F1, `n_unparsed`, `yes_ratio`, vision-token count under
Policy A (LLaVA 576; Qwen-VL-Chat 448×448; Qwen2-VL/2.5 native default),
device config (note CPU offload / multi-GPU if used), and `--limit 200` per
split. Flag explicitly: **accuracy/F1 baseline only — NOT suitable for
activation extraction** (offload/multi-GPU environment). Note that the eval is
VTI-comparable (same metrics, free-generation elicitation, official POPE
template) but uses a more robust first-token parser than VTI's bare substring
match.
