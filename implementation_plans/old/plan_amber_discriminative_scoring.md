# Implementation plan — AMBER discriminative scoring (grounding-isolation breakdown)

**For:** Cursor implementation agent (Remote-SSH, lambdab2, repo open).
**Author:** research/analysis agent, via Romanus.
**Goal:** extend AMBER scoring so the **discriminative** split emits the same
yes-bias / negative-item decomposition that `score_pope_records` produces, so AMBER
can be used to isolate visual grounding from the rotation intervention's
agreeableness (yes-bias) and length confounds. AMBER discriminative is the
length-robust counterpart to CHAIR; together they cross-check each other.

**Why this benchmark / why this breakdown:** AMBER discriminative is short yes/no
(length-robust, unlike CHAIR), but built post-POPE on a richer question set
(existence + attribute + relation). The single confound it does *not* kill on its
own is agreeableness — like POPE, it hands the model a candidate. The fix is the
same one already built for POPE: expose **yes-ratio** and **accuracy on the
negative (gold = "no") items** as first-class numbers. A yes-drifting model craters
on negatives; genuine grounding holds them. Current AMBER scorer reports only
`accuracy_overall` + `by_task`, which **hides exactly this signal** — that aggregate
is the thing to replace.

**Scope discipline:** scorer + (if needed) loader extension only. Do **not** touch
`src/model.py`, the interventions, the POPE/CHAIR/MMHal paths, or the runner's
generation loop. Integration point is `compute_metric_records(records, benchmark)`
in `evaluation/classifiers/metrics.py` for benchmark key `amber`, task
`discriminative`. Generation already works (AMBER discriminative runs today and
produces `task_accuracy`); only the scoring is being enriched.

---

## Step 0 — Verify-in-repo before writing scorer logic (do this first, report back)

These determine the scorer. Resolve by reading the repo; paste findings into
`RESEARCH_LOG.md` / back to Romanus before implementing.

**0a. Where AMBER gold answers live, and in what form.** Open `load_amber` in
`src/dataset.py` and the current `task_accuracy` branch in
`evaluation/classifiers/metrics.py`. The existing scorer already knows right from
wrong, so a gold→sample join exists somewhere. Determine:
- Does the loaded **discriminative** sample carry its gold yes/no in `label` /
  `label_idx` (as the uniform sample dict suggests), or does the scorer join
  against the AMBER annotation file (`junyangwang0410/AMBER` query/annotations
  JSON) at score time?
- What are the exact gold label **string values** for discriminative items
  (e.g. `"yes"`/`"no"`, or AMBER's native encoding)? The new scorer must compare
  against these verbatim — do not assume POPE's encoding.

**0b. Per-question-type tag on discriminative items.** AMBER discriminative
subdivides into **existence / attribute / relation** (attribute further into
state / number / action in the AMBER taxonomy). Determine whether the loaded
sample surfaces this type (in `task`, `category`, or inside `raw`/`metadata`).
- If **yes**: the scorer breaks out per type for free (high value — see A3).
- If **no**: note it. The type lives in the AMBER source `query` file keyed by
  question id; surfacing it is a small loader extension (flag as new, do not
  silently skip — the per-type negative-item breakdown is the most diagnostic
  output here).

**0c. AMBER's own metric convention (do we need it).** AMBER's official scoring for
the discriminative split is **accuracy, plus precision/recall/F1 with a documented
positive class**. Confirm whether the repo intends AMBER-official metrics or the
POPE-style convention. Default below uses the **POPE convention** (positive class =
`yes`) so AMBER and POPE numbers are read on the same axis in this project; note
the choice in the summary so it is auditable. If strict AMBER-paper comparability
is wanted later, that is an additional output, not a replacement.

**0d. Parser reuse.** Confirm the yes/no normalizer `_normalize_yes_no` in
`evaluation/classifiers/metrics.py` (leading-token priority, word-boundaried
fallback, `not`/`cannot`/`none` guarded) is importable for reuse. AMBER
discriminative answers are yes/no free-text just like POPE, so the **same parser
must be reused** — do not write a second yes/no parser (divergent parsing is how
POPE and AMBER would silently become non-comparable).

---

## Implementation

### A1. Scorer: `score_amber_discriminative_records`

Add to `evaluation/classifiers/metrics.py`, wired into `compute_metric_records`
for `benchmark == "amber"` **when task == discriminative**. Leave the
**generative** AMBER path on its existing scorer untouched (generative AMBER is a
different metric — AMBER's CHAIR-like / hallucination-rate scoring — and is out of
scope here; this plan is discriminative only).

```python
def score_amber_discriminative_records(records) -> dict
```

Per record: parse the model response to yes/no with the **reused** `_normalize_yes_no`
(0d); compare to gold (0a). Positive class = `yes` (0c).

### A2. Top-level return shape

Mirror the POPE summary so downstream tooling and the analyst read AMBER and POPE
identically:

```python
{
    "metric": "amber_discriminative",
    "accuracy_overall", "precision_overall", "recall_overall", "f1_overall",
    "yes_ratio",                 # THE agreeableness signal — fraction of "yes" answers
    "n_correct", "n_total", "n_unparsed",

    # negative-item isolation (the reason this plan exists):
    "neg_item_accuracy",         # accuracy on gold == "no" items only
    "pos_item_accuracy",         # accuracy on gold == "yes" items only
    "n_neg_total", "n_pos_total",
}
```

- **`neg_item_accuracy` is the headline grounding-isolation number.** Under the
  agreeableness confound, a yes-drifting model's `yes_ratio` rises and
  `neg_item_accuracy` falls together (it's answering "yes" to absent things).
  Genuine grounding improvement raises or holds `neg_item_accuracy` while
  `yes_ratio` stays put. This pair is what AMBER contributes that CHAIR cannot.
- Unparsed handling: match POPE exactly (per IMPLEMENTATION.md POPE scorer —
  unparseable counts incorrect for accuracy; for P/R/F1, FN when gold is `yes`,
  uncounted when gold is `no`). Track `n_unparsed`; expect ≈0 on yes/no items.

### A3. Per-question-type breakdown (conditional on 0b)

If the type tag is available (existence / attribute / relation), add:

```python
    "by_qtype": {
        qtype: {
            "accuracy", "yes_ratio", "neg_item_accuracy", "pos_item_accuracy",
            "n_total", "n_neg_total", "n_pos_total", "n_unparsed",
        }
    }
```

This is the most diagnostic output: it answers "does rotation help **existence**
grounding while hurting **relation** grounding," which aggregate accuracy hides.
Keep qtype key spelling exactly as the loader/source surfaces it. If 0b found no
type tag and surfacing it is deferred, emit the A2 fields without `by_qtype` and
note the omission in the summary (e.g. `"by_qtype": null, "note": "qtype tag not
surfaced by loader"`) so its absence is explicit, not silent.

### A4. Runner / table wiring

- No new generation path, no judge, no new CLI flag. AMBER discriminative already
  runs via `--benchmarks amber --amber_task discriminative`; only the scorer
  output changes.
- `print_comparison_table`: render AMBER discriminative with at minimum
  `acc / neg_acc / yes_ratio` (e.g. `acc/neg/yr`), so the agreeableness signal is
  visible at a glance next to POPE's per-split columns — not buried in the JSON.
  Make column meaning explicit in the header/legend.
- Confirm `--amber_task` already threads through `run_evaluation` (IMPLEMENTATION.md
  lists `amber_task` as a `run_evaluation` kwarg, so this should be wired — verify,
  don't add a duplicate).

---

## Validation (before any sweep)

1. **Dry run, baseline, tiny limit.** `--benchmarks amber --amber_task
   discriminative --limit 16 --interventions no_intervention` on
   `llava-hf/llava-1.5-7b-hf`. Confirm: gold join resolves (0a), parser reused (0d),
   summary has the A2 fields, `n_neg_total + n_pos_total == n_total`,
   `neg_item_accuracy` populated, `by_qtype` present iff 0b found the tag.
2. **Gold-join sanity.** For ~3 of the 16, log (response, parsed yes/no, gold) so
   Romanus can eyeball that the gold answers are aligned to the right questions —
   an off-by-one in the annotation join would silently produce plausible-but-wrong
   accuracy. (AMBER's separate query/annotation files make this the most likely
   bug; check it explicitly.)
3. **yes_ratio sanity vs POPE.** On the same model, baseline AMBER discriminative
   `yes_ratio` should be in a sane range (not 0 or 1). A degenerate yes_ratio
   usually means the parser isn't firing or the gold encoding mismatched (0a/0d).
4. **Baseline-only first, all four models.** `no_intervention` on AMBER
   discriminative for LLaVA-1.5, Qwen-VL-Chat, Qwen2-VL, Qwen2.5-VL at Policy-A
   resolution. Record in `RESEARCH_LOG.md`. These baselines are the reference the
   intervention sweep is read against — especially each model's baseline
   `yes_ratio` and `neg_item_accuracy`, since the whole point is measuring how the
   rotation **moves** those two relative to baseline.
5. **One intervention smoke, then stop.** A single
   `vti_textual_uniform_rotation_mlp` run at one beta on Qwen2.5-VL, confirm the
   summary diffs sensibly against baseline (yes_ratio and neg_item_accuracy both
   present and moved), before launching the full beta grid.

Device policy: accuracy/scoring runs (generation + post-hoc yes/no scoring) — CPU
offload / multi-GPU allowed. No activation extraction, so the single-device
constraint does not apply.

---

## How this reads against CHAIR (for the analyst, not Cursor — context only)

The two benchmarks have **orthogonal confounds**, which is the point:
- **AMBER discriminative** kills the *length* confound (short yes/no), exposes
  *agreeableness* directly as `yes_ratio` + `neg_item_accuracy`.
- **CHAIR** kills the *agreeableness* confound (model volunteers objects, nothing
  to agree with), exposes *length* directly as `avg_objects_mentioned` + caption
  length.

Genuine grounding improvement from the rotation must show in **both**: AMBER
`neg_item_accuracy` up at flat `yes_ratio`, **and** CHAIR_i down at matched
`avg_objects_mentioned`. An effect that appears in only the benchmark whose confound
happens to favor it (AMBER accuracy up purely via yes-drift, or CHAIR better purely
via terseness) is the confound, not grounding. Agreement across the two orthogonal
failure modes is the strong signal.

---

## Open questions / decisions for Romanus

1. **0a gold encoding** — confirm exact gold label strings for AMBER discriminative
   so the scorer compares verbatim. (Cursor resolves in-repo; flag if AMBER uses a
   non-`yes`/`no` encoding that needs mapping.)
2. **0b qtype tag** — if existence/attribute/relation is not surfaced by
   `load_amber`, decide: extend the loader to pull it from the AMBER `query` file
   (preferred — `by_qtype` is the most diagnostic output), or ship A2 fields only
   for now and add per-type later.
3. **0c metric convention** — confirm POPE-convention (positive = `yes`, for
   same-axis reading with POPE) is what you want as the primary, vs AMBER-paper
   official metrics. Default is POPE-convention; AMBER-official can be an added
   output if a write-up needs paper comparability.
4. **Generative AMBER** — out of scope here (different metric). Confirm you only
   want **discriminative** enriched now; flag if you also want AMBER generative
   (its hallucination-rate scoring) built, which is a separate plan closer to CHAIR.
