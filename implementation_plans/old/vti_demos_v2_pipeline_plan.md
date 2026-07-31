# Implementation Plan — `demos_v2`: minimal-pair, dimension-controlled VTI demo generation

**For:** Cursor (implementation agent)
**From:** analyst, via Romanus
**Date:** 2026-07-12
**Status:** ready to implement after the Open Questions at the bottom are resolved with Romanus

---

## 1. Goal

Replace the author-released VTI paired-caption demos (`data/vti/demos.jsonl`, 100 rows) with a
new, larger, controlled demo set. Problems with the author data (verified on the file):

1. **Not minimal pairs.** In 69/100 rows, a word-level diff between `value` and `h_value` shows
   more than two edit regions — pairs differ in structure and content beyond the hallucination,
   so the contrastive direction captures nuisance variation.
2. **"Truthful" captions contain false claims** (e.g. `000000321938` claims sandwiches "spread
   across multiple plates"; there is one plate).
3. **Hallucination type is uncontrolled** — almost exclusively object existence/relation, so
   per-dimension analysis (existence vs. attribute vs. counting vs. relation) is impossible.

Target output: **150 COCO train2014 images**, each with **one truthful caption and four
hallucinated variants**, where each variant differs from the truthful caption by **exactly one
edit region in exactly one designated sentence**, corresponding to one dimension:
`existence`, `attribute`, `counting`, `relation`. Every image must support all four dimensions;
images that cannot are dropped, not partially included.

Success gates:
- Deterministic diff validator (§8) passes on 100% of emitted rows (enforced, not sampled).
- Truthfulness grounded in COCO instance annotations wherever possible (counts, presence,
  bbox geometry); MLLM judgment used only where annotations cannot reach (attributes,
  absence confirmation, phrasing).
- Romanus manually reviews an HTML gallery after every stage before the next stage runs.

---

## 2. Placement, conventions, dependencies

- **Scripts:** `data_scripts/vti_demos_v2/` (new package). One script per stage + shared modules.
- **Artifacts:** `data/vti/v2/` (new dir) for all intermediate stage outputs, raw API call logs,
  and review HTML. Final deliverable: `data/vti/demos_v2.jsonl`.
- **`src/paths.py` additions (new):** `vti_demos_v2_dir() -> Path` (`data/vti/v2`) and
  `vti_demos_v2_path() -> Path` (`data/vti/demos_v2.jsonl`).
- **Provider client:** mirror the `evaluation/classifiers/judges.py` conventions — spec strings
  `anthropic[:model]` / `openai[:model]` / `mock`, keys from `ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY`, clear error when missing. **First check** whether the provider client in
  `data_scripts/generate_captions.py` already supports image inputs and can be reused; if not,
  write `data_scripts/vti_demos_v2/mllm_client.py` as a thin vision-capable client following the
  judges.py pattern. Do **not** add new packages to the conda env (pins are load-bearing); use
  whatever HTTP/SDK approach the existing provider code already uses.
- **`mock` provider is mandatory** for every stage: returns canned, schema-valid outputs so the
  full pipeline can be dry-run end-to-end with no keys (same philosophy as `--judge mock`).
- **Resume-by-id:** every stage skips input ids already present in its own output artifact
  (mirrors the repo's checkpoint/resume convention). Re-running a stage is always safe.
- **Runs on lambdab2, CPU-only**, inside the normal env. No GPU, no W&B. All data is public
  COCO — external API calls are fine.
- **No `pycocotools`:** `instances_train2014.json` is plain JSON; parse with stdlib.
- **Call logging:** every real API request/response is appended (JSONL) under
  `data/vti/v2/calls/stage{N}.jsonl` with image id, model, token counts. Each stage also writes
  `data/vti/v2/stage{N}_summary.json` (yield, rejection reasons histogram, total tokens in/out).

### Config

Single `data_scripts/vti_demos_v2/config.py` with defaults (all overridable by CLI flags):

```python
N_CANDIDATES = 300          # stage-0 output batch size
N_FINAL = 150               # final image target
COUNT_MIN, COUNT_MAX = 2, 9 # annotated instance count range for the counting anchor
REL_GAP_FRAC = 0.05         # min horizontal gap between relation bboxes, fraction of image width
REL_MIN_AREA_FRAC = 0.015   # min bbox area fraction for each relation object
DISTRACTOR_TOP_K = 5        # co-occurrence-ranked absent-category candidates passed to stage 1
FALSE_COUNT = lambda n: min(max(n + 2, math.ceil(1.5 * n)), 20)
SEED = 42
STAGE1_PROVIDER = "anthropic:claude-sonnet-4-6"
STAGE2_PROVIDER = "anthropic:claude-opus-4-8"   # strongest available; Romanus may swap
STAGE3_PROVIDER = "anthropic:claude-haiku-4-5"
STAGE4_PROVIDER = "anthropic:claude-sonnet-4-6" # must differ from stage-2 and stage-3 models
QUESTION = "Describe this image in detail."
BANNED_HEDGES = ["possibly", "likely", "appears", "appear", "seems", "seem", "perhaps",
                 "might", "may", "probably", "some kind of", "what looks like"]
```

Model choice rationale (Romanus decides finals): stage 2 is the hardest (compose an accurate,
constrained caption) → strongest model. Stage 1 needs real visual judgment (absence checks,
attribute grounding) → strong-mid. Stage 3 is a text-only one-phrase insertion → cheapest.
Stage 4 must be a **different model** from the stage-2/3 writers so verification is not
self-grading. All vision calls are **one image per call** — multi-image batching degrades
per-image attention and complicates output parsing, and total volume (~850 calls) makes cost a
non-issue (expect low tens of dollars with this tiering; exact token usage is in the stage
summaries).

### Data prerequisites (stage 0 checks these and fails loudly)

- `data/coco/annotations/instances_train2014.json` — may not be present (CHAIR only needs
  val2014). If missing, download the COCO 2014 trainval annotations zip and extract this file
  into `coco_annotations_dir()`.
- `data/coco/train2014/` may contain only the 100 original demo images. Stage 0 must not assume
  the full split is local: candidate images missing on disk are fetched individually from
  `http://images.cocodataset.org/train2014/COCO_train2014_{id:012d}.jpg` (config flag
  `ALLOW_IMAGE_DOWNLOAD = True`).
- Exclusion list: the 100 ids in `data/vti/demos.jsonl` are never candidates.

---

## 3. Pipeline overview

Romanus's three tasks map onto six stages. Stages 0 and 5 are deterministic (no API); stage 3 is
mostly deterministic. Manual review gates sit after every stage.

| Stage | Romanus's task | What it does | API? |
|---|---|---|---|
| 0 | task 1 (part) | Mine candidates from COCO annotations: anchors for counting/relation/existence | no |
| 1 | task 1 (part) | MLLM verifies anchors visually, picks distractor + attribute anchor | yes (vision) |
| 2 | task 2 | MLLM writes the rigid 4-sentence truthful caption + exact edit spans | yes (vision) |
| 3 | task 3 | Build 4 hallucinated variants: 3 by validated string edit, 1 (existence) by cheap LLM | small (text) |
| 4 | (added) | Independent MLLM verifies truthful=true, each variant=false-in-one-way | yes (vision) |
| 5 | (added) | Assemble final JSONL, exporter, galleries | no |

The caption template is fixed (Romanus's spec). Sentence roles:

- **S1 — scene + existence anchor.** Names the scene and enumerates ≥2 present anchor objects.
  The existence variant inserts the distractor into this enumeration.
- **S2 — attribute.** One object (or object group) with one unambiguous attribute value.
- **S3 — counting.** Exactly the phrase `at least {N} {category}` with N spelled out as a word.
- **S4 — relation.** `The {A} is to the left of the {B}` (or right), viewer perspective.

Global caption rules (enforced by prompt + deterministic post-check, §8): exactly 4 sentences;
**no digits anywhere** (numbers spelled out — also keeps naive sentence splitting on `". "` safe);
no words from `BANNED_HEDGES` (this is exactly the `uncertain_objects` hedging vocabulary the
author pipeline injected — we eliminate it); each sentence self-contained (no cross-sentence
pronouns whose referent breaks when one sentence is edited); no claims beyond the verified
anchors plus minimal scene glue; the distractor noun must not appear anywhere in the truthful
caption.

---

## 4. Stage 0 — `stage0_mine_candidates.py` (deterministic)

**Input:** `instances_train2014.json`, exclusion list.
**Output:** `data/vti/v2/stage0_candidates.jsonl` (top `N_CANDIDATES` by score) + summary.

Per image, from annotations only:

1. **Category inventory:** all annotated categories with instance counts, split by `iscrowd`.
2. **Counting anchor:** a category with exact instance count in `[COUNT_MIN, COUNT_MAX]`, zero
   `iscrowd` instances of that category in the image, and median instance bbox area ≥ 0.5% of
   image area (annotation completeness correlates with object size; tiny-object categories are
   where COCO under-annotates most). If several qualify, keep the one with largest median area.
3. **Relation anchor:** ordered pair of **distinct** categories (A, B), individual instances only
   (`iscrowd=0`), such that **every** instance of A is horizontally disjoint-left of **every**
   instance of B: `max(x2 of A instances) < min(x1 of B instances) − REL_GAP_FRAC · width`, and
   each participating bbox area ≥ `REL_MIN_AREA_FRAC` of image area. The all-instances
   quantifier removes referential ambiguity when a category has multiple instances. Image
   x increases to the viewer's right, so this is "A to the left of B" in viewer perspective —
   the same convention stages 2 and 4 must use. Keep the pair with the largest gap.
4. **Existence distractor candidates:** compute corpus-level co-occurrence over train2014
   annotations once (cache to `data/vti/v2/cooccurrence.json`): for each COCO category c absent
   from this image, score = mean over present categories p of P(c | p). Emit the top
   `DISTRACTOR_TOP_K` absent categories. (POPE-adversarial style: plausible-in-context but
   absent. Restricting distractors to the COCO-80 lets the absence claim be annotation-backed;
   the residual under-annotation risk is what stage 1 and stage 4 check.)
5. **Candidate iff** counting anchor ∧ relation anchor ∧ ≥1 distractor candidate ∧ ≥2 distinct
   present categories. Rank by a simple deterministic score (weighted: relation gap, count-anchor
   median area, number of distinct categories; ties broken by image id).

Record schema (one line per candidate):

```json
{
  "id": "000000321938",
  "image": "COCO_train2014_000000321938.jpg",
  "present_categories": {"sandwich": 8, "bowl": 4, "person": 1},
  "counting_anchor": {"category": "sandwich", "count": 8, "median_area_frac": 0.021},
  "relation_anchor": {"a": "sandwich", "b": "bowl", "relation": "left",
                       "gap_frac": 0.11, "a_n": 8, "b_n": 4},
  "distractor_candidates": ["fork", "knife", "cup", "spoon", "dining table"],
  "score": 0.83
}
```

CLI: `--n-candidates`, `--exclude-ids-file` (defaults to original demos), `--emit-next-batch`
(for top-up: excludes ids already present in any stage artifact, emits the next N by rank —
used if final yield < `N_FINAL`).

**Manual gate 0:** Romanus skims `stage0_candidates.jsonl` and the summary (yield, category
distribution — check it isn't 300 pictures of sheep).

---

## 5. Stage 1 — `stage1_verify_anchors.py` (vision API, 1 image/call)

**Input:** stage-0 candidates + the image.
**Output:** `stage1_verified.jsonl`, `stage1_rejected.jsonl` (with per-dimension reasons).

One call per image. The model receives the image **and** the stage-0 anchor metadata; its job is
**verification and selection, not open-ended description** — the annotations are the primary
ground truth, the model is a visual sanity filter and supplies only what annotations cannot.

**System prompt must specify all of the following** (Cursor writes the prose; these constraints
are the spec):

- Role: quality-control annotator for a contrastive captioning dataset; the reply is machine-
  parsed; output **only** a single JSON object matching the given schema — no prose, no
  markdown fences.
- Per-check semantics, each returning `pass` | `fail` + one-sentence reason:
  1. **counting_check:** the stated category and count are given. Confirm the objects are
     individually distinguishable in the image (not a blurred mass, not mostly occluded) and
     that the claim "at least {count} {category}" reads as clearly true. Do **not** recount from
     scratch as the primary signal; the count comes from human annotation — fail only if the
     image visibly contradicts it or the objects can't be distinguished.
  2. **relation_check:** confirm, from the **viewer's** perspective, that every {A} is clearly to
     the left of every {B}, with visible separation. Fail on ambiguity (overlap in depth,
     partial visibility).
  3. **distractor_selection:** from the provided candidate list (in order), pick the **first**
     category that is (a) plausible in this scene and (b) **definitely not visible anywhere** in
     the image, including background, partial views, and reflections. Scan carefully before
     ruling absent. If none qualify, fail. Output the chosen distractor.
  4. **attribute_selection:** choose one clearly visible object or homogeneous object group and
     one attribute with an unambiguous, nameable value — **color strongly preferred**; material
     or a coarse size term only if no clean color exists. Also output one **false** alternative
     value that is clearly not present on that object (e.g. plates are white → false value
     "black", and nothing black about the plates). Both values must be 1–2 words. If the image
     is black-and-white, do not use color. The chosen object should ideally be distinct from the
     counting-anchor category (avoid coupling two dimensions to the same noun); allow overlap
     only if unavoidable, and flag it.
- Prohibitions: no hedging language in any output field; no attributes requiring counting or
  spatial reasoning (those dimensions are taken); no subjective attributes ("delicious",
  "beautiful").

Response schema (per image):

```json
{
  "counting_check": {"verdict": "pass", "reason": "..."},
  "relation_check": {"verdict": "pass", "reason": "..."},
  "distractor": {"verdict": "pass", "choice": "fork", "reason": "..."},
  "attribute": {"verdict": "pass", "object": "plates", "true_value": "white",
                 "false_value": "black", "overlaps_counting_category": false, "reason": "..."}
}
```

An image passes stage 1 iff **all four** verdicts pass. Parsing failure → one retry with the
parse error appended to the user turn; second failure → rejected with reason `unparseable`.

**Manual gate 1:** `render_review.py --stage 1` (see §10) — image beside anchors and verdicts.
Romanus reviews before stage 2 runs.

---

## 6. Stage 2 — `stage2_write_truthful.py` (vision API, strongest model, 1 image/call)

**Input:** stage-1 verified records + image.
**Output:** `stage2_captions.jsonl`, `stage2_rejected.jsonl`.

One call per image; the model writes **one** truthful caption plus a span map. It is given the
image and every anchor (categories, count N spelled out, relation pair + direction, attribute
object/value, distractor — the distractor is provided **only** as a forbidden word).

**System prompt must specify:**

- Write exactly four sentences, in the fixed roles S1–S4 above, and nothing else.
- S1: name the scene and enumerate the anchor objects present (must include the counting
  category and both relation categories or their natural noun phrases); enumeration form
  ("…with X, Y, and Z") so a later insertion is grammatical.
- S2: exactly one attribute claim: the given object with the given true value; the value word(s)
  must appear **exactly once in the whole caption**.
- S3: must contain the exact phrase `at least {N_word} {category…}`; the number word must appear
  exactly once in the whole caption; no other quantity words (several, many, few, a couple).
- S4: exactly `The {A phrase} is to the left of the {B phrase}.` structure (or "right" if the
  anchor says so), viewer perspective; the words "left"/"right" appear exactly once in the
  caption.
- Every claim must be visibly true in the image; nothing beyond the anchors plus minimal scene
  glue; glue must be trivially true (locations like "on a counter" only if visible).
- Forbidden: digits; all `BANNED_HEDGES` words; the distractor word; pronouns referring across
  sentences; subjective language.
- Output JSON only:

```json
{
  "caption": "The image features a counter laden with food, including sandwiches, bowls of pasta, and bottles. The plates holding the food are white. There are at least eight sandwiches on one plate. The plate of sandwiches is to the left of a bowl of pasta.",
  "spans": {
    "existence_insertion_hint": "including sandwiches, bowls of pasta, and bottles",
    "attribute": "white",
    "counting": "eight",
    "relation": "left"
  }
}
```

`spans` values must be verbatim substrings of the caption (`existence_insertion_hint` = the S1
enumeration the distractor will be inserted into; the other three are the exact tokens stage 3
replaces).

**Deterministic post-validation in code (no API), before accepting a record:** exactly 4
sentences (split on `". "` / terminal `.`); `at least {N_word}` present with the correct number
word; each span present exactly once in the caption; span sentence-membership correct (attribute
span in S2, counting in S3, relation in S4, insertion hint in S1); no digits; no banned hedges;
no distractor token; relation word matches the anchor direction. Any failure → one retry with
the specific violations listed in the user turn; second failure → rejected.

**Manual gate 2:** gallery — image + caption with the four spans highlighted. This is the most
important human check (caption truthfulness); Romanus reviews all ~200 before stage 3.

---

## 7. Stage 3 — `stage3_make_variants.py` (mostly code; text-only LLM for existence)

**Input:** stage-2 records. **Output:** `stage3_variants.jsonl` (+ per-variant diff report).

- **counting / relation / attribute — pure string operations, no API:**
  - counting: replace the `counting` span (number word) with `number_to_word(FALSE_COUNT(N))`.
  - relation: replace `left` ↔ `right`.
  - attribute: replace the true value span with the stage-1 `false_value`.
- **existence — one cheap text-only call per image** (`STAGE3_PROVIDER`; no image attached).
  System prompt must specify: given a caption and a target noun `{distractor}`, insert **one**
  noun phrase naming it (with a natural article/plural, ≤4 words total) into the first
  sentence's enumeration (the provided `existence_insertion_hint` substring); change **nothing
  else** — no rewording, no punctuation changes outside the insertion; return the full caption
  as JSON `{"caption": "..."}`.
- **Every variant (all four) passes the diff validator (§8) before acceptance.** Existence
  failures retry once with the violation; deterministic edits should never fail — a failure
  there means a stage-2 span bug, so it is logged loudly and the record rejected.

**Manual gate 3:** gallery with the truthful caption and all four variants, edit regions
highlighted (word-level diff rendering). Fast to skim because the validator already guarantees
the shape.

---

## 8. Diff validator — `validators.py` (shared, unit-tested)

The mechanical guarantee the author demos lack. For a (truthful, variant, dimension) triple:

1. Whitespace-tokenize both captions; run `difflib.SequenceMatcher.get_opcodes()`.
2. Require **exactly one** non-`equal` opcode region.
3. Map the region to its sentence index (sentence split is safe: no digits, controlled prose)
   and require it to be the dimension's designated sentence (existence→S1, attribute→S2,
   counting→S3, relation→S4).
4. Per-dimension token constraints:
   - existence: pure insertion (or replace that only extends the enumeration), ≤4 inserted
     tokens, must contain the distractor token(s), remaining inserted tokens ∈ {"and", "a",
     "an", "some", commas}.
   - attribute: replace; removed tokens == true value; inserted == false value.
   - counting: replace; removed == true number word; inserted == expected false number word.
   - relation: replace; {left ↔ right} exactly.
5. Additionally, for the record as a whole: the truthful caption passes the stage-2 structural
   checks, and the distractor token appears in **no** variant except existence.

Unit tests in `tests/test_demos_v2_validators.py`: pass cases per dimension; fail cases —
multi-region edit, edit in wrong sentence, extra reworded token, distractor leaking into the
truthful caption, digit in caption, hedge word. Also test `number_to_word`, `FALSE_COUNT`
boundaries (N=2→4, N=8→12, cap at 20), relation geometry helper from stage 0, and sentence
splitting. All tests run without GPU or network (mock provider).

---

## 9. Stage 4 — `stage4_verify_faithfulness.py` (vision API, independent model, 1 image/call)

**Input:** stage-3 records + image. **Output:** `stage4_verdicts.jsonl`, `stage4_rejected.jsonl`.

One call per image, model ≠ stage-2 model and ≠ stage-3 model (config-asserted). Verification is
framed as **targeted binary judgments**, not open captioning — binary presence/truth judgments
are much more reliable than free generation, and the count question is anchored by annotation
ground truth anyway.

**System prompt must specify:** answer each numbered statement about the image with strictly
`true` / `false` / `unsure` + one short reason; judge only what is visible; viewer perspective
for left/right; output JSON only (`{"answers": [{"idx": 1, "verdict": "true", "reason": "..."}]}`).

The user turn lists, generated from the record: (a) each of the four truthful-caption sentences
as a separate statement — expected `true`; (b) the four hallucinated claims, phrased minimally —
"There is a {distractor} visible in the image" (expected `false`), "The {object} is
{false_value}" (expected `false`), "There are at least {false_N} {category}" (expected `false`),
"The {A} is to the {false_rel} of the {B}" (expected `false`).

Pass iff all eight expectations met. `unsure` counts as fail (conservative). Failures →
rejected with the failing statement recorded; **no automatic retry loop** — disagreement here is
signal for the human, not noise to be prompted away.

Honest caveat for the log: the verifier is itself a VLM with the same failure modes (counting
especially). It is a filter, not ground truth. The counting claims lean primarily on the COCO
annotation counts from stage 0; the final authority is manual gate 4.

**Manual gate 4:** gallery of all records (passes and rejects), verdicts inline. Romanus reviews
rejects to catch over-strict filtering and spot-checks passes.

---

## 10. Stage 5 — `stage5_assemble.py` + tooling (deterministic)

- Collect records that passed stages 1–4; if count < `N_FINAL`, print the shortfall and the
  command to mine the next stage-0 batch (`--emit-next-batch`); else take the first `N_FINAL`
  by stage-0 rank.
- Write `data/vti/demos_v2.jsonl`, one line per image:

```json
{
  "id": "000000321938",
  "image": "COCO_train2014_000000321938.jpg",
  "question": "Describe this image in detail.",
  "value": "<truthful caption>",
  "h_values": {
    "existence": "<variant>",
    "attribute": "<variant>",
    "counting": "<variant>",
    "relation": "<variant>"
  },
  "anchors": {
    "existence": {"distractor": "fork"},
    "attribute": {"object": "plates", "true_value": "white", "false_value": "black"},
    "counting": {"category": "sandwich", "annotated_count": 8,
                  "true_word": "eight", "false_word": "twelve"},
    "relation": {"a": "plate of sandwiches", "b": "bowl of pasta",
                  "true": "left", "false": "right"}
  },
  "provenance": {"stage1_model": "...", "stage2_model": "...", "stage3_model": "...",
                  "stage4_model": "...", "pipeline_version": "...", "date": "..."}
}
```

  Field names `value` / `question` / `image` / `id` deliberately match the original demo schema;
  `co_objects` / `uncertain_objects` are dropped (extraction never reads them per
  IMPLEMENTATION.md).

- **`export_vti_flat.py`** (deterministic): `--dimension {existence|attribute|counting|relation}
  --out data/vti/demos_v2_{dimension}.jsonl` → original VTI flat schema (`id`, `image`,
  `question`, `value`, `h_value`), directly consumable by
  `VTITextualIntervention(demos_path=...)` with no loader change. `num_demos=70` seed-42
  sampling then draws from the 150-pool per dimension as it does today.
- **`render_review.py --stage {0|1|2|3|4|final} [--open]`**: self-contained base64-embedded HTML
  galleries under `data/vti/v2/_review/`, following the `render_vti_demos_review.py` /
  `review_lib` patterns (reuse `review_lib` helpers where they fit). The stage-3/4/final views
  render word-level diffs of each variant against the truthful caption with the edit region
  highlighted.

---

## 11. Execution order (for Romanus)

```bash
# 0. deterministic mining (no keys needed)
python data_scripts/vti_demos_v2/stage0_mine_candidates.py
python data_scripts/vti_demos_v2/render_review.py --stage 0        # gate 0

# 1–4, each followed by its review gate
python data_scripts/vti_demos_v2/stage1_verify_anchors.py          # gate 1
python data_scripts/vti_demos_v2/stage2_write_truthful.py          # gate 2  (most important)
python data_scripts/vti_demos_v2/stage3_make_variants.py           # gate 3
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py     # gate 4

# 5. assemble + exports
python data_scripts/vti_demos_v2/stage5_assemble.py
python data_scripts/vti_demos_v2/export_vti_flat.py --dimension counting  # etc.
```

Full-pipeline dry run first: `--provider mock` on every API stage over ~5 stage-0 candidates,
verifying artifacts, resume, and galleries end-to-end before spending tokens. Then a paid pilot
of **~15 images through all stages** for Romanus to review quality and per-stage prompts before
the full 300-candidate run.

---

## 12. Required follow-up before any extraction uses this data (flagged, out of scope here)

1. **Textual direction cache is not keyed on demo identity.** IMPLEMENTATION.md places the
   textual cache at `experiment_artifacts/vti/{model_short}/` — swapping `demos_path` to a
   demos_v2 export would silently reuse directions computed from the old demos. Before any
   demos_v2 extraction run, the cache path/slug must incorporate the demos file (name + content
   hash), mirroring how the visual arm's `config_slug` works. Verify whether the visual slug
   includes demos identity as well; fix both if not.
2. `run_eval.py` has no CLI plumbing for `demos_path` — needed to run per-dimension steering
   through the standard eval path (`--demos_path` or per-dimension registry variants; decision
   with Romanus and the analyst).

---

## 13. After implementation

- Update **IMPLEMENTATION.md**: new package, artifact layout, final schema, exporter, paths.py
  helpers, provider/config conventions, and the two follow-ups above if addressed.
- Append a factual **RESEARCH_LOG.md** entry per pipeline run: date, commit, stage commands,
  models used, yield per stage, token totals, final file path + content hash.

## 14. Open questions (resolve with Romanus before/while starting)

1. Can the provider client in `data_scripts/generate_captions.py` be reused for vision +
   JSON-constrained calls, or does judges.py's pattern generalize better? (Cursor: check both,
   state the choice in IMPLEMENTATION.md.)
2. Is `instances_train2014.json` already under `data/coco/annotations/`? Is `data/coco/train2014/`
   the full split or just the 100 demo images? (Determines download volume; both handled either
   way per §2.)
3. Confirm final model choices per stage (defaults in §2 config) and that both API keys are
   exported on lambdab2.
4. Confirm the name `demos_v2` (file `data/vti/demos_v2.jsonl`) or propose an alternative before
   paths are baked in.
5. Stage-1 attribute preference is color-first. Acceptable, or should material/size be weighted
   in from the start for image variety?
6. When yield < 150 after the first 300 candidates: automatic top-up batch size (default:
   another 100 by stage-0 rank)?
