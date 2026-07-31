# Implementation Plan — `demos_v2.1`: diversity augmentation of the minimal-pair demo pipeline

**For:** Cursor (implementation agent)
**From:** analyst, via Romanus
**Date:** 2026-07-13
**Baseline:** the working v2.0 pipeline as documented in IMPLEMENTATION.md § `data/vti/demos_v2`
(300 → 131 funnel, content hash `ad44185346c48958`). This plan modifies that package in place;
the v2.0 finals remain a valid frozen artifact under `pipeline_version = demos_v2_2026-07-12`.
New version string: `demos_v2.1_<date>`.

---

## 1. Goal and motivation

Diagnosis on the v2.0 finals (131 rows):

- **relation:** 131/131 truthful direction is `left` — every pair is the identical `left→right`
  token swap. The extracted "relation" direction is a lexical perturbation direction, not a
  spatial-hallucination direction.
- **attribute:** ~95% color; `white` is the true value in 32/131; false values dominated by
  `black`/`red`/`white`.
- **existence:** top-5 distractor categories (bowl, cup, chair, knife, bottle) cover 71%.
- **counting:** 74% N=2 (`two→four`), 40% category `person`.
- **accuracy:** relation records can be referentially ambiguous (`000000304765`: "the dining
  table" with multiple tables visible; S1 says "dining tables" plural; the cup sits *on* a
  table) and still pass stage 4.

Root cause of every distributional collapse: **per-image argmax choices** (first plausible
distractor, color-preferred attribute, canonical left ordering, single counting anchor). The
central change is therefore a global, seeded **allocator** that assigns one option per dimension
per image against fractional quotas. Around it: relation-type expansion (4 geometric types),
attribute-type expansion (5 types), counting mode expansion (`at_least`/`at_most`), scene
stratification at stage 0, referential-uniqueness and depth checks, sentence-scoped span
replacement, and a fifth combined-hallucination caption `h_values.all`.

Usage context that shapes the design: per-dimension steering vectors will be extracted from the
**same** final pool (one flat export per dimension), and the pipeline will later run on
substantially more than 300 stage-0 candidates. Hence all quotas are **fractions of the pool**,
never absolute counts, and subtype balance must hold within each dimension.

---

## 2. Task 0 — rejection diagnostics on the existing 300-run (do this first; no behavior change)

New script `data_scripts/vti_demos_v2/analyze_rejections.py`:

- Reads `stage1_rejected.jsonl` → histogram by failed check
  (`counting_check` / `relation_check` / `distractor` / `attribute` / `unparseable`), with
  example ids per bucket.
- Reads `stage4_rejected.jsonl` → histogram by failing statement, mapped to
  {truthful-S1..S4} × {existence, attribute, counting, relation false claims}. If the current
  reject records do not store the failing statement index → dimension mapping explicitly, add
  that to stage 4's reject records as part of this task (and note the v2.0 rejects can only be
  bucketed as far as the stored reason allows).
- Writes `data/vti/v2/rejection_analysis.json` + prints a table. Keep it runnable after every
  future run — this is a permanent pipeline health tool.

Also three **code reconciliation checks** — report findings to Romanus before implementing §4:

1. **Stage-0 relation gap:** is the horizontal gap computed edge-to-edge or centroid-based, and
   per instance pair or per category aggregate? (Something admitted the 304765 cup/table pair;
   edge-disjointness between the *chosen instances* should have been violated if the cup sits on
   the chosen table, so either centroids were used or a different table instance was chosen.)
2. **Span replacement mechanics:** confirm whether stage-3 replacement is caption-global
   unique-token replace (as the v2.0 validator constraints suggest). §6 makes it
   sentence-scoped; flag anything that makes that refactor non-local.
3. **Artifact compatibility:** the stage-2 `spans` schema changes in §6; confirm nothing else
   consumes `spans` besides stage 3 and the validators. v2.1 requires a fresh run from stage 0
   (predicates change); no migration of v2.0 stage artifacts.

---

## 3. Config additions (`config.py`)

All new values live in config with these defaults; every quota is a fraction.

```python
PIPELINE_VERSION = "demos_v2.1_<date>"

# Relation types and geometry
REL_TYPES = ("horizontal", "vertical", "support", "proximity")
REL_GAP_FRAC = 0.05            # edge-to-edge, per chosen instance pair (reconcile per §2.1)
V_GAP_FRAC = 0.05              # vertical edge-to-edge gap
X_OVERLAP_MIN = 0.30           # vertical: x-range overlap ≥ 30% of the narrower box
SUPPORT_SURFACES = ("dining table", "bed", "couch", "bench", "chair")
SUPPORT_MAX_AREA_RATIO = 0.25  # area(A) / area(B) for "A on top of B"
NEAR_MAX_FRAC = 0.02           # proximity "right next to": normalized box-to-box distance
FAR_MIN_FRAC = 0.40            # proximity "far away from"
REL_TYPE_TARGET = {t: 0.25 for t in REL_TYPES}   # allocator targets, feasibility-constrained

# Attribute types
ATTR_TYPES = ("color", "material", "state", "action", "texture")
ATTR_TYPE_TARGET = {"color": 0.35, "material": 0.20, "state": 0.20,
                    "action": 0.15, "texture": 0.10}
ATTR_VALUE_CAP_FRAC = 0.08     # no true_value exceeds 8% of pool
K_ATTR = 3                     # stage-1 candidates per image, distinct types preferred

# Existence distractors
DISTRACTOR_TAU = 0.5           # sampling weight = cooccurrence_score ** TAU
DISTRACTOR_CAP_FRAC = 0.04     # no distractor category exceeds 4% of pool

# Counting
COUNT_MODE_AT_MOST_FRAC = 0.5  # of at_most-eligible images
AT_MOST_MIN_N = 3              # eligibility: N >= 3 AND stage-1 count_complete
FALSE_COUNT_AT_MOST = lambda n: max(1, min(n - 2, (2 * n) // 3))
N_GE3_TARGET_FRAC = 0.40       # allocator target share of finals with N >= 3
PERSON_COUNT_CAP_FRAC = 0.15   # counting category "person" cap

# Scene stratification (stage 0)
SUPERCAT_CAP_FRAC = 0.20       # per COCO supercategory of the counting-anchor category

ALLOC_SEED = 42
```

`number_words.py` gains `false_count_at_most(n)` implementing the formula above, with tests.
Note on `y = 1`: "at most one {singular noun}" forces a plurality change on the category noun
(`chairs → chair`, `people → person`). This is intended and handled by the validator (§6);
the counting edit region for `at_most` may span the number word **plus** the adjacent category
noun (reuse the `category_mentioned` plural map for the singular↔plural forms). If Romanus later
prefers to avoid singular morphology entirely, the toggle is `AT_MOST_MIN_Y = 2` +
`AT_MOST_MIN_N = 4`; default is the formula as agreed.

---

## 4. Stage 0 — `stage0_mine_candidates.py`

Stage 0 stops making choices; it emits **feasible option sets** plus geometry facts.

1. **Relation options.** For every ordered pair of categories (A, B) where **both have exactly
   one annotated instance** (`iscrowd=0`) and neither is the image's counting-anchor category,
   evaluate each predicate on the instance boxes (image coords; x→viewer right, y→down):
   - `horizontal`: x-ranges disjoint edge-to-edge with gap ≥ `REL_GAP_FRAC · width`, **and**
     neither box contains the other (belt-and-suspenders with disjointness), areas ≥
     `REL_MIN_AREA_FRAC`. Store which side A is on; the allocator decides subject order and
     hence the truthful word.
   - `vertical`: y-ranges disjoint with gap ≥ `V_GAP_FRAC · height`, x-range overlap ≥
     `X_OVERLAP_MIN` of the narrower box (so "above/below" reads naturally), areas ≥
     `REL_MIN_AREA_FRAC`.
   - `support`: B ∈ `SUPPORT_SURFACES`; A's box contained in B's box (small tolerance); A in
     the upper half of B's box; `area(A)/area(B) ≤ SUPPORT_MAX_AREA_RATIO`. Truthful phrase is
     always "on top of" (there is no truthful "underneath" pole from COCO boxes).
   - `proximity`: normalized min box-to-box distance d (by image diagonal): `near` if
     d ≤ `NEAR_MAX_FRAC`, `far` if d ≥ `FAR_MIN_FRAC`, plus the containment exclusion. Both
     poles are truthful options depending on geometry.
   Emit every satisfied (pair, type, geometry stats) as an option. An image is
   relation-feasible if ≥1 option exists.
2. **Counting options.** Emit **all** categories with count in `[COUNT_MIN, COUNT_MAX]`, zero
   `iscrowd`, median area ≥ `COUNT_MIN_AREA_FRAC` — each with its N (allocator picks).
3. **Distractor candidates:** unchanged mining (top-`DISTRACTOR_TOP_K` absent categories by
   co-occurrence), but they are now candidates for *sampling*, not an ordered pick-first list.
4. **Scene stratification:** during ranked emission, cap candidates whose available counting
   anchors all fall in one COCO supercategory at `SUPERCAT_CAP_FRAC` of the batch; rank
   preference for images offering larger-N counting options and multiple relation types
   (extend the existing `W_*` score weights; keep deterministic tie-breaking by id).

Candidate record: replace `counting_anchor` / `relation_anchor` / `distractor_candidates`
scalars with `counting_options[]`, `relation_options[]`, `distractor_candidates[]` (schema
bump; stage 1 and the allocator consume this).

---

## 5. Stage 1 — `stage1_verify_anchors.py`: verify options, don't choose

Same one-image-per-call structure; the response schema changes from single choices to verdicts
over option sets. **System prompt requirements** (rewrite `prompts.py` stage-1 templates):

- Role/framing unchanged (QC annotator; JSON only; annotations are ground truth, the model is a
  visual filter and supplies only what annotations cannot).
- **relation_options:** for EACH provided option, return `pass`/`fail` + reason, checking:
  (a) **visible uniqueness** — exactly one instance of each category visible anywhere in the
  image, including partial, background, and unannotated instances (this is the check COCO
  annotations cannot make); (b) the stated geometric claim holds from the **viewer's**
  perspective; (c) the two objects are at **comparable depth / on the same surface or plane** —
  fail foreground-vs-background pairings even when the projective claim is technically true;
  (d) for `support`: A visibly rests on B; for `proximity` `near`: visibly adjacent with no
  object of the same categories between them; for `far`: clearly separated.
- **counting:** existing visibility/distinguishability check, PLUS a new boolean
  `count_complete` per counting option: "true only if no additional instances of {category} are
  visible beyond the {N} identified — check edges, background, partial occlusions." This gates
  `at_most` eligibility; it must be judged conservatively (when unsure → false). Note in the
  prompt that `count_complete=false` does NOT fail the option; it only restricts its mode.
- **distractors:** verdict `absent`/`present`/`unsure` for EACH candidate (not pick-first),
  with the same careful-absence instruction; `unsure` is treated as ineligible downstream.
- **attributes:** return up to `K_ATTR` candidates, distinct `type` values preferred, each
  `{type ∈ ATTR_TYPES, object, true_value, false_value, confidence}`; same-type false value
  mandatory (color→different color clearly not present; state→the opposite state, e.g.
  open/closed, on/off, full/empty, lit/unlit; action→a different posture/activity for animate
  objects, e.g. standing/sitting/lying/walking; material/texture→a clearly different one);
  values 1–2 words; no counting- or position-flavored attributes (those dimensions are taken);
  no subjective terms; color candidates forbidden on black-and-white images.
- An image passes stage 1 iff: ≥1 relation option passed, ≥1 counting option passed, ≥1
  distractor `absent`, ≥1 attribute candidate returned. Reject reasons per dimension feed
  `stage1_rejected.jsonl` as today (and now `analyze_rejections.py`).

---

## 6. New component — `stage1b_allocate.py` (deterministic allocator)

**Input:** complete `stage1_verified.jsonl`. **Output:** `stage1b_allocation.jsonl` (one chosen
option per dimension per image) + `stage1b_summary.json` (achieved vs target distributions —
the diversity health-check artifact; manual gate 1b).

- Seeded (`ALLOC_SEED`), deterministic given input. Suggested algorithm: process images in
  most-constrained-first order (fewest feasible options across dimensions); per dimension pick
  the feasible option that most reduces the max quota deficit; weighted-random (seeded) among
  near-ties for distractors using `score ** DISTRACTOR_TAU`.
- Quotas (all fractional, from config): relation type ≈ `REL_TYPE_TARGET` subject to
  feasibility; relation direction 50/50 within `horizontal` (left/right) and `vertical`
  (above/below) via subject-order choice — this also kills the all-`left` bug; `proximity`
  balanced between near-truthful and far-truthful where geometry allows; `support` has one
  truthful pole by construction (report its share, no balancing possible). Attribute type ≈
  `ATTR_TYPE_TARGET`, `ATTR_VALUE_CAP_FRAC` on true values and on (true,false) mappings.
  Distractor `DISTRACTOR_CAP_FRAC` per category. Counting: mode split
  `COUNT_MODE_AT_MOST_FRAC` among eligible (N ≥ `AT_MOST_MIN_N` ∧ `count_complete`), category
  caps (`PERSON_COUNT_CAP_FRAC`), N-distribution target (`N_GE3_TARGET_FRAC`).
- **Top-up semantics (required for the planned scale-up):** the allocator reads any existing
  allocation file and assigns **only new ids**, counting existing assignments against quotas.
  Never reassign an already-allocated id (downstream stages depend on it). A full reallocation
  requires wiping stage 1b–5 artifacts (document this in the script header and
  IMPLEMENTATION.md).

---

## 7. Stage 2 — `stage2_write_truthful.py`

Consumes the allocation (one option per dimension). Prompt and validator changes:

- **S3 templates:** `at_least`: "There are at least {x_word} {category-plural} …" (x = N,
  unchanged). `at_most`: "There are at most {x_word} {category} …" with **x = N** (per
  Romanus's decision; the stage-1 `count_complete` gate is what makes x = N safe against
  under-annotation).
- **S4 templates per relation type** (fixed templates in `prompts.py`; the model fills noun
  phrases only): horizontal "The {A} is to the {left|right} of the {B}."; vertical "The {A} is
  {above|below} the {B}."; support "The {A} is on top of the {B}."; proximity "The {A} is right
  next to the {B}." / "The {A} is far away from the {B}."
- **Singular-form rule:** both relation categories must be referred to in the **singular**
  everywhere in the caption (they are visually unique by stage-1 check; a plural mention
  anywhere recreates the 304765 inconsistency). Add to prompt and to
  `structural_check_truthful`.
- **Spans become sentence-scoped:** `spans` entries become
  `{"sentence_idx": int, "text": str}`; each span must be unique **within its sentence**.
  Additional leakage guards (deterministic): the attribute value and the count number word must
  not appear in any *other* sentence; the relation direction word need only be sentence-unique,
  but the **false** direction word must not already appear in the relation sentence. (Real
  motivating case from v2.0: 304765's S2 contains "right side of the room" while S4 flips
  left→right — a caption-global replace would corrupt S2.)
- All other v2.0 truthful-caption constraints stand (4 sentences, no digits, banned hedges,
  distractor absent, spans verbatim in the right sentences).

## 8. Stage 3 — `stage3_make_variants.py`

- All replacements sentence-scoped per the new span schema.
- Counting: `at_least` → `FALSE_COUNT(N)` (existing); `at_most` → `false_count_at_most(N)`.
  Edit region may include the category-noun plurality change when y = 1 (validator below).
- Relation flips by type: `left↔right`, `above↔below`, `on top of ↔ underneath`,
  `right next to ↔ far away from` (both directions).
- Existence path unchanged (deterministic insert into the S1 hint + Haiku grammar polish; LLM
  fallback; `existence_source` recorded).
- **Combined variant `h_values.all` (new, fully deterministic, no API, no stage-4 pass):**
  start from the truthful caption; replace S1 with the already-validated existence variant's
  S1 verbatim; apply the three sentence-scoped swaps to S2–S4. Validator `validate_combined`:
  word-diff vs truthful decomposes into **exactly four edit regions, one per sentence**, each
  region individually satisfying its dimension's constraint. Revisit condition (agreed with
  Romanus): if manual review finds the combined captions unnatural, add an optional
  verification/polish pass later — not now.

**Validator updates (`validators.py`):** per-dimension edit-region rules updated —
attribute/counting/relation regions must lie in their designated sentence with the expected
token swaps; counting `at_most` allows number word + adjacent noun-morphology in one contiguous
region; proximity allows a ≤4-token contiguous region ("right next to" ↔ "far away from");
existence keeps its S1 multi-region allowance; singular-form rule; combined-variant rule.

## 9. Stage 4 — `stage4_verify_faithfulness.py`

- Statement list updated per type/mode: relation records get **two uniqueness statements
  first** ("Exactly one {A} is visible in the image." / same for {B}, expected `true`) before
  the direction statement — this closes the 304765 pass-through, where "the cup is left of the
  dining table" was judged true because *a* reading was true. False relation claims phrased per
  type ("The {A} is underneath the {B}", etc.). Counting `at_most`: truthful "There are at most
  {x_word} {category} visible" (expected true — note this makes the verifier a second
  completeness check on top of stage 1's, deliberately conservative) and false "There are at
  most {y_word} {category}" (expected false).
- `h_values.all` is **not** verified (deterministic composition of verified parts).
- Reject records must store the failing statement's dimension mapping explicitly (consumed by
  `analyze_rejections.py`).
- Expect the new uniqueness/depth checks to cost relation yield; the Task-0 histograms are the
  before/after baseline.

## 10. Stage 5, schema, exporter, review

- **Final schema additions:** `h_values.all`; `anchors.attribute.type`;
  `anchors.relation.type` + chosen geometry stats; `anchors.counting.mode`
  (`at_least`/`at_most`) with `true_word`/`false_word` interpreted per mode. Bump
  `provenance.pipeline_version`.
- **`export_vti_flat.py`:** `--dimension` gains `all`; new optional `--subtype` filter
  (`--dimension relation --subtype vertical`, `--dimension attribute --subtype state`,
  `--dimension counting --subtype at_most`). Flat schema unchanged (`value`/`h_value`).
- **`stage5_summary.json`:** full distribution report — relation types and directions,
  attribute types and top values, distractor categories, counting modes/N histogram, counting
  categories, supercategory mix. This plus `stage1b_summary.json` are the standing diversity
  health checks.
- **`render_review.py`:** show subtype/mode labels per record; render `all` with its four
  highlighted edit regions; stage-1b view = the allocation summary tables.

## 11. Tests (`tests/test_demos_v2_validators.py` + new `tests/test_demos_v2_allocator.py`)

- Relation predicates on synthetic boxes: containment (cup-on-table ⇒ horizontal infeasible,
  support feasible), disjoint-with-gap, vertical alignment threshold, proximity near/far bands,
  support area-ratio and surface whitelist.
- `false_count_at_most` boundaries (N=3→1, N=4→2, N=6→4, N=9→6) and the y=1 plurality edit
  (including `person/people`).
- Sentence-scoped replacement: the "right side of the room" collision case verbatim from
  000000304765 (global replace must be shown NOT to occur).
- `validate_combined`: exactly-four-regions pass case; failure cases (missing region, two
  regions in one sentence, region violating its dimension rule).
- Allocator: determinism (same input+seed ⇒ identical allocation), quota satisfaction on a
  synthetic feasibility matrix, top-up invariance (existing assignments untouched, quotas
  respected marginally).
- All CPU-only, no network (mock provider).

## 12. Execution order

```bash
# A. diagnostics on the existing run (no behavior change)
python data_scripts/vti_demos_v2/analyze_rejections.py        # + §2 code checks → Romanus

# B. implement §3–§11; then
python -m pytest tests/test_demos_v2_validators.py tests/test_demos_v2_allocator.py -q

# C. fresh mine + mock dry-run end-to-end (~5 ids), then paid pilot ~20 ids through all stages
python data_scripts/vti_demos_v2/stage0_mine_candidates.py --n-candidates 300
#    ... stages 1, 1b, 2, 3, 4, 5 ...; review galleries at every gate, esp. 1b distributions
#    and the h_values.all naturalness check (revisit condition in §8)

# D. full run at Romanus's chosen N_CANDIDATES (larger than 300); update run_full300.sh into a
#    parameterized run_full.sh (N_CANDIDATES env var) that wipes stage1–5 but keeps stage0,
#    runs stage1b between 1 and 2, and assembles with --n-final = pool size
```

Cost scaling note: calls ≈ N_candidates (stage 1) + survivors (stage 2) + existence polish +
survivors (stage 4); stage-1 outputs get longer (option verdicts). At N_candidates = 1000
expect roughly 2.5–3k calls — still modest; token totals in the stage summaries as before.

## 13. Bookkeeping

- Update IMPLEMENTATION.md § `data/vti/demos_v2`: new stage 1b, option-set schemas, quotas and
  config, sentence-scoped spans, `h_values.all`, exporter `--subtype`, allocator top-up
  semantics, new tests, version bump.
- RESEARCH_LOG.md: factual entry for the Task-0 diagnostics (histograms + the §2 reconciliation
  findings) and for each pilot/full v2.1 run (commands, models, yield funnel, distribution
  summaries, finals hash).

## 14. Open questions

1. §2 reconciliation findings may adjust §4's horizontal predicate spec (edge vs centroid) —
   Cursor reports before implementing.
2. Allocator quota defaults (§3) are analyst suggestions; Romanus may retune after seeing the
   pilot's `stage1b_summary.json` — they are config values, not code.
3. `run_full300.sh` → parameterized `run_full.sh`: confirm Romanus wants the rename or a new
   sibling script (resume compatibility with any in-flight v2.0 artifacts is not required —
   v2.1 is a fresh run).
