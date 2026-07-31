# Detailed rejection reasons — paid 5-candidate pilot

Pipeline funnel: **5 → 4 → 3 → 1 → 0**. Below is *why* each drop happened, with the actual text.

---

## `000000002466` — Stage 1 reject (visual QC)

**Anchors from COCO:** count = 5× `potted plant`; relation = every `cup` left of every `wine glass`.

The stage-1 model failed **two** checks (either alone is enough to reject):

1. **Counting:** annotation says 5 potted plants, but the model says only 2–3 are clearly distinguishable in the background — not individually countable.
2. **Relation:** cups and wine glasses are intermixed on the table, so “every cup left of every wine glass” is not visually true.

Distractor (`bowl`) and attribute (`wine glass` / red vs white wine) would have passed.

**How to read this:** stage 0’s geometry/count gates are annotation-only; stage 1 is doing its job catching bad visual anchors. Not a code bug.

---

## `000000001145` — Stage 2 reject (caption structure)

**Anchors:** count category = **`person`** (N=2); attribute = red pendant lamp; distractor = sink.

Model wrote (both attempts):

> “…busy with **people**, a refrigerator, and a potted plant… There are at least two **people**…”

Deterministic validator requires the counting **category string** `person` to appear in the caption. It sees `people` only → reject: `counting category 'person' not found in caption`.

Otherwise the caption was well-formed (4 sentences, spans, etc.).

**How to read this:** strict string match vs pluralization. Easy prompt/validator fix (accept plural/lemma), or force the model to use the exact COCO category token.

---

## `000000000127` — Stage 3 reject (existence not minimal-pair)

**Truthful S1 enumeration:**  
`…with a slice of cake, a spoon, and a knife.`

**Existence target:** insert distractor `chair`.

**Model output (both attempts):**  
`…with a slice of cake, a spoon, a knife, and a chair.`

Semantically fine, but the word-level diff is **two** regions because it also rewrote the list conjunction:

| change | removed | inserted |
|--------|---------|----------|
| insert | — | `a knife,` |
| replace | `knife.` | `chair.` |

(i.e. it dropped the Oxford-style `and` before `knife` and appended `, and a chair`.) Validator requires **exactly one** edit region → reject.

**How to read this:** not a bad hallucination; our existence LLM is too free with list grammar. Deterministic insert (or a stricter prompt / post-edit) would recover these.

---

## `000000002429` — Stage 3 reject (same existence issue)

**Truthful:** `…with kitchen cabinets, dining tables, and chairs.`  
**Model:** `…with kitchen cabinets, dining tables, chairs, and a person.`

Again: dropped `and` before `chairs` and appended `, and a person` → two opcode regions → reject. Same class of failure as above.

---

## `000000003915` — Stage 3 pass, Stage 4 reject (faithfulness)

Only image that got four variants. Truthful caption:

1. S1: counter with potted plants, white bowls, refrigerator  
2. S2: refrigerator has a **silver** finish…  
3. S3: at least **two** potted plants…  
4. S4: bowl left of refrigerator  

Stage 4 asked eight true/false questions. Failures on the **truthful** side (must all be `true`):

| # | Statement (expected true) | Model said | Reason |
|---|---------------------------|------------|--------|
| 2 | S2 — fridge is silver | **false** | Looks stainless/dark, not “silver-reflective” |
| 3 | S3 — ≥2 potted plants | **unsure** | Flowers visible; unclear there are two distinct *potted plants* |

Hallucinated claims (5–8) were correctly judged false. So variants are fine; the **truthful** caption’s attribute + count wording didn’t survive an independent VLM check.

**How to read this:** borderline attribute language (`silver` vs stainless) + soft plant/flower ambiguity. Plan says stage 4 is a filter and manual review is authority — you could override this one after looking at the image, or treat it as correctly dropped.

---

## Implications for how to proceed

| Issue | Severity | Likely fix direction |
|-------|----------|----------------------|
| Stage 1 visual fails | Expected attrition | Need larger stage-0 pool (you already mined 300) |
| Stage 2 `person`/`people` | Easy yield leak | Relax category match or pin exact token in prompt |
| Stage 3 list-rewriting existence | Main mechanical loss (2/3 of stage-2 passes) | Deterministic string insert into `existence_insertion_hint` |
| Stage 4 attribute/count disputes | Qualitative | Accept as filter, or human override; prefer unambiguous attributes (clear colors on large objects) |

**Practical recommendation:** harden stage 3 existence (biggest cheap win), optionally soften stage 2 category matching, then run the new 300-candidate pool through stages 1–4 — don’t expect 5/5 on a kitchen-heavy micro-pilot.
