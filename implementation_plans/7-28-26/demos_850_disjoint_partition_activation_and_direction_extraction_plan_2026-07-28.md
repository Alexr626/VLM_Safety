# demos_850 disjoint-partition activations and steering directions

extraction_spec: extractions/steering_vector_diff_sample_size_07_28_26_extraction.md

Date: 2026-07-28
Status: to be implemented and run by Cursor.
Kind: extraction. Produces primitives only — no comparison, no number read as evidence.

## Handoff: who runs the mining, and what gates it

**Alex runs steps 1-5 (the mining and caption-generation stages) by hand. Cursor does not run them.**
They are CPU + API only, no GPU, and they are the long pole — so they should start as early as
possible and run concurrently with the rest of the implementation.

They cannot start yet. **Two prerequisites must land before Alex runs a single command**, both of
which are Cursor's to do:

**P1 — Add `--summary-out` to `data_scripts/vti_demos_v2/stage0_mine_candidates.py`.** Two lines.
After line 258 in `parse_args`:

```python
    p.add_argument("--summary-out", type=Path, default=None)
```

and line 324 becomes:

```python
    write_summary(args.summary_out or (v2 / "stage0_summary.json"), summary)
```

Default behaviour is unchanged when the flag is absent. **Without this change, step 1 overwrites
`data/vti/v2/stage0_summary.json` — the 555-era mining provenance — in place.** That file is not
regenerable; the 1000-candidate run that produced it will not be re-run.

**P2 — Execute check 0.3, the stage-artifact snapshot.** Full definition in the Checks section. Every
stage script overwrites its `*_summary.json` in place, so the snapshot and the
`pre_topup_checksums.txt` baseline must exist before the first API call, not after.

**When both are done, Cursor tells Alex explicitly: "P1 and P2 are complete, mining is unblocked."**
Cursor then continues with steps 6 onward while Alex mines. Do not hand Alex the commands before
both are confirmed; a step-1 run against an unpatched script is not recoverable by re-running it.

### What Alex must know before running anything

Read this before the first command, not after a failure.

1. **Never run `data_scripts/vti_demos_v2/run_full.sh`.** Lines 8-10 `rm -f` the stage 1-5 artifacts,
   the stage1b allocation, the call logs, and `data/vti/demos_v2*.jsonl` — including the 555-item
   pool this whole plan is built on top of. There is no confirmation prompt. Run the stage scripts
   individually, as listed in steps 1-5.
2. **`data/vti/demos_v2.jsonl` is read-only for the entire plan.** Its sha256[:16] `9a44f4afde0324b5`
   is a slug component in all 40 existing direction directories and an integrity guard in
   `demos_v2_order_s42.json`. Changing it by one byte — including appending the new rows to it —
   makes every existing textual_v2 direction unreachable and the existing selection unreproducible.
   The new pool is a separate file. If any command would write to `demos_v2.jsonl`, stop.
3. **Run step 2 (stage 1) promptly after step 1.** `--emit-next-batch` excludes ids already present
   in `demos.jsonl` and in every `data/vti/v2/stage*.jsonl`. New ids only become exclusion-visible
   once stage 1 writes them, so a second step-1 run before stage 1 can re-emit the same candidates.
4. **Mock runs write real rows.** If smoke-testing with `--provider mock --limit 5`, point `--input`
   at a scratch file. Never mock against `stage0_candidates_topup_2026-07-28.jsonl` — the mock rows
   land in the real stage outputs and are then skipped as already-processed by the live run.
5. **Stages 1b-4 take no `--input`.** They read the default v2 files and skip already-processed ids,
   so they touch only the new candidates. This is correct and intended; it also means a mistake in
   step 1 propagates silently through all of them.
6. **After stage 4, verify before continuing.** `sha256sum data/vti/demos_v2.jsonl` must still return
   `9a44f4afde0324b5...`, and every `data/vti/v2/stage*.jsonl` must be append-only against
   `pre_topup_checksums.txt`. Any mismatch: stop and report, do not proceed to step 6.
7. **If fewer than 295 admissible rows come out of two extra mining rounds, stop.** Per check 0.2,
   the 50/100/200/500 partition cannot then be built as specified, and changing the block sizes is a
   field-2 change only Alex makes.

## Question

Build an 850-item demo pool (the existing 555 `demos_v2` items plus 295 newly generated items of the
same construction), partition it once into disjoint blocks of 50, 100, 200, and 500, and extract, for
both LLaVA-1.5-7B and Qwen2.5-VL-7B, the per-layer last-token activations for every item under every
caption variant, plus one steering direction per (model x hallucination dimension x block).

Nothing here is compared. The blocks exist so that a later experiment — behind its own design spec —
can compare directions built from independent samples of different size.

## Extraction spec reference

`extractions/steering_vector_diff_sample_size_07_28_26_extraction.md`

- Field 1: activations at each layer, for 295 new demos-v2-style items (one truthful caption, four
  single-dimension hallucinated captions, one all-dimension hallucinated caption), for both models.
- Field 2: one partition of the 850 items into 50 / 100 / 200 / 500. Every pair of blocks disjoint.
  The same partition is used for all five hallucination dimensions — no image appears in two blocks
  regardless of dimension. Held constant: item pool, caption construction, models. Varies: sample
  size, and with it item identity.
- Field 3: steering vectors must be computable from these activations.

Everything below this line is plan-side resolution: namespacing, slugs, cache layout, output paths,
reuse, counts, feasibility, verification. None of it changes fields 1-3.

## Additivity trace (read before writing any file)

The 850-item pool must **not** be created by appending to `data/vti/demos_v2.jsonl`. That file's
sha256[:16] is `9a44f4afde0324b5` and the hash is a slug component and an integrity guard:

- `evaluation/interventions/vti/directions_v2.py:164` builds every existing direction slug as
  `demosv2_{hash[:8]}_{dim}_nd{N}_s42_r2_prefix`. Appending rows changes the hash, so
  `compute_or_load_textual_directions_v2` would stop resolving to the 40 existing direction
  directories under `experiment_artifacts/vti/{model_short}/textual_v2/` — bytes intact, artifacts
  unreachable.
- `load_or_build_master_order` (`directions_v2.py:83-88`) raises when
  `data/vti/demos_v2_order_s42.json` `_meta.content_hash_sha256_16` no longer matches the demos file,
  so the existing `shuffled_prefix` selection would become unreproducible.

Therefore the 850-row pool is a **new file**, `data/vti/demos_850.jsonl`, whose first 555 lines are
byte-identical copies of `data/vti/demos_v2.jsonl` and whose remaining 295 lines are the new items.
`demos_v2.jsonl`, `demos_v2_order_s42.json`, and every existing `textual_v2/demosv2_*` directory are
read-only for the whole of this plan.

Two further prohibitions, both load-bearing:

- **Do not run `data_scripts/vti_demos_v2/run_full.sh`.** Lines 8-10 delete `stage1..5` artifacts,
  the stage1b allocation, the call logs, and `data/vti/demos_v2*.jsonl`. The top-up runs the stage
  scripts individually; all of them append and skip already-processed ids
  (`stage1_verify_anchors.py:149-153`, `stage2_write_truthful.py:120-124`,
  `stage3_make_variants.py:240-244`, `stage4_verify_faithfulness.py:80-84`), and
  `stage1b_allocate.py` is explicitly built for top-ups (existing allocation rows are preserved
  verbatim, new ids appended, balancing histograms seeded from the existing rows).
- **Do not pass `data/vti/demos_850.jsonl` to `run_eval.py --demos_path` or to
  `evaluation/run_scripts/extract_demosv2_directions.py`.** Those paths route through
  `load_or_build_master_order` with `DEFAULT_ORDER_PATH`, which raises on the hash guard. That
  failure is correct behaviour; the new module below never touches the v2 order file.

The stage scripts overwrite their `*_summary.json` files in place. Snapshot them first (step 0.3) so
the 555-era provenance survives.

## Cells

Activation cells, per model: 850 items x 6 caption variants = 5100 cached last-token stacks.
Variants are `value` (truthful) plus `existence`, `attribute`, `counting`, `relation`, `all`.

Direction cells: 2 models x 5 dimensions x 4 block sizes = 40.

| model_short | dimension | block | slug |
|---|---|---|---|
| llava-1.5-7b-hf | existence \| attribute \| counting \| relation \| all | 50 \| 100 \| 200 \| 500 | `demos850_{h8}_{dim}_nd{N}_s42_r2_partition` |
| qwen2.5-vl-7b-instruct | existence \| attribute \| counting \| relation \| all | 50 \| 100 \| 200 \| 500 | `demos850_{h8}_{dim}_nd{N}_s42_r2_partition` |

`{h8}` is the first 8 hex chars of sha256(`data/vti/demos_850.jsonl`), known only after step 6.
Models: `llava-hf/llava-1.5-7b-hf` and `Qwen/Qwen2.5-VL-7B-Instruct`, normalised by
`src.model._normalize_model_name` to the two `model_short` values above.

Forward-pass count is invariant to the partition. Each activation is a property of
(model, item, caption variant) alone, so which block an item lands in changes nothing about what must
be forwarded: the cost is set by the 850-item pool, not by the block sizes, and a nested design would
have cost exactly the same.

## Data

### Existing, reused as-is

| Artifact | Fact |
|---|---|
| `data/vti/demos_v2.jsonl` | 555 rows, sha256[:16] `9a44f4afde0324b5`. Verified 2026-07-28: 555 unique ids, and all 555 rows carry all five `h_values` keys. |
| `data/vti/v2/stage0_candidates.jsonl` | 1000 mined candidates (`stage0_summary.json`, `n_excluded=200`). |
| `data/vti/v2/stage{1,1b,2,3,4}*.jsonl` | Stage records for those 1000. Stage-4 passes = 555, rejects = 258. |
| `data/coco/train2014/` | 82,783 images present locally; no COCO download needed at any stage. |
| `experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/_act_cache/` | 3000 files = 500 ids x 6 variants, 985 MB. |
| `experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/_act_cache/` | 3000 files = 500 ids x 6 variants, 622 MB. Built with `--max_pixels 1003520` (log `logs/demosv2_qual_qwen_2026-07-13.log`, `num_layers=28 hidden_dim=3584`). |

The 500 cached ids are exactly the first 500 entries of `data/vti/demos_v2_order_s42.json` (spot-checked
at the boundary: index 499 `000000054492` is cached for both models; index 500 `000000466519` is not).
So 500 of the 850 items already have all six variants cached for both models, and only 350 items x 6
variants x 2 models = 4200 forwards are new.

Cache reuse is only sound because `ActivationCache` keys on `sample_{id}_{suffix}` alone
(`src/extraction.py:65-66`) — the image path, caption text, and Qwen `max_pixels` are **not** in the
key. Steps 0.1, 0.2 and 0.4 close each of those holes before a single new file is written.

### New item generation (295 items)

Same package, same providers, same `PIPELINE_VERSION` (`demos_v2.1_2026-07-13`), same
`config.QUESTION` (`"Describe this image in detail."`). New rows are distinguishable from the 555 only
by `provenance.date = 2026-07-28`. The 555 finals were assembled from *all* stage-4 passes with no
manual exclusion list (`stage5_summary.json`: `n_final = n_available = 555`), so the new items pass
through exactly the same automated gates and nothing else — caption construction is held constant, as
field 2 requires.

Observed stage yields from the 1000-candidate run: stage1 835/1000, stage2 827/835, stage3 813/827,
stage4 555/813. End-to-end 55.5%. Mining **650** new candidates has an expected yield of ~361 stage-4
passes against a requirement of 295.

### Partition

`data/vti/demos_850_partition_s42.json`, built once and then immutable:

```json
{"_meta": {"demos_file": "data/vti/demos_850.jsonl",
           "content_hash_sha256_16": "<h16 of demos_850.jsonl>",
           "seed": 42, "selection_policy": "disjoint_partition",
           "sizes": [50, 100, 200, 500], "n_ids": 850, "created": "2026-07-28"},
 "blocks": {"50": ["..."], "100": ["..."], "200": ["..."], "500": ["..."]}}
```

Construction mirrors `load_or_build_master_order`: take the 850 ids, `sorted()`, shuffle once with
`random.Random(42)`, then consume the shuffled list in order into blocks of 50, 100, 200, 500. The
seed and the ordering convention are the repo's existing ones (`directions_v2.py:36`, `:90-95`); they
carry no experimental content. 50+100+200+500 = 850 exactly, so the partition consumes every id once
and there is zero slack — which is why step 0.2 refuses to assemble any row missing a dimension
caption rather than skipping it at selection time the way `select_prefix_demos` does.

## Metrics

None. This plan produces no metric and no number that is read as evidence. The quantities computed
are verification quantities about the artifacts themselves — file counts, tensor shapes, per-layer
norms of a single direction, and a reproduction check of one cached activation against a fresh
forward of the same input. No two arms are contrasted anywhere in this plan. Cosines, explained-
variance comparisons across blocks, and any behavioural evaluation of these directions belong to a
later plan behind a design spec.

`metadata.json` will contain `explained_variance_ratio`, `pc1_layer_norms`, and
`direction_layer_norms` because `obtain_textual_vti_v2_from_stacks` already emits them per fit; they
are descriptive fields of one artifact, written and not read here.

## Sanity checks that must pass first

These gate everything. 0.1-0.3 gate the paid API stages; 0.4 gates the GPU stage and may run
concurrently with the API stages since it depends only on artifacts that already exist.

**0.1 — Existing pool completeness and id space.** Programmatically confirm `demos_v2.jsonl` has 555
rows, 555 unique ids, every row with non-empty `value` and non-empty `h_values[d]` for all five
`d in (existence, attribute, counting, relation, all)`, and `question == "Describe this image in
detail."` on every row. Invalidates the design if false: a row missing a dimension cannot be in a
partition shared across all five dimensions, and the 850 total would not hold.
*(Pre-checked 2026-07-28 by line count and pattern match: 555/555 rows carry all five keys, 555 unique
ids. The implementer re-runs it as code.)*

**0.2 — New-row admission rule.** Every candidate new row must carry non-empty `value` and all five
`h_values` entries before it counts toward the 295. Rows failing this are dropped and the mining
target is raised, never backfilled and never allowed into the pool. If, after two additional
`--emit-next-batch` mining rounds, fewer than 295 admissible new rows exist, **stop and report** — the
850 pool, and therefore the 50/100/200/500 partition, cannot be built as specified, and that is a
field-2 change only Alex can make.

**0.3 — Stage-artifact snapshot and non-destruction.** Before the first API call, copy
`data/vti/v2/stage0_summary.json`, `stage1_summary.json`, `stage1b_summary.json`,
`stage2_summary.json`, `stage3_summary.json`, `stage4_summary.json`, `stage5_summary.json` into
`data/vti/v2/_summaries_snapshot_555_2026-07-28/`, and record `sha256sum` plus `wc -l` of every
`data/vti/v2/stage*.jsonl` and of `data/vti/demos_v2.jsonl` into
`data/vti/v2/_summaries_snapshot_555_2026-07-28/pre_topup_checksums.txt`. After every stage, re-check
that each stage file's first N recorded bytes are unchanged (append-only) and that
`sha256sum data/vti/demos_v2.jsonl` still returns `9a44f4afde0324b5...`. Any mismatch: stop.

**0.4 — Activation cache fidelity (per model, gates all GPU work).** The Qwen cache was built at
`max_pixels=1003520`, a parameter absent from the cache key and absent from the existing
`metadata.json`. Before extending either cache, re-forward 3 already-cached ids x 2 variants
(`value`, `all`) under the exact intended invocation and compare against the cached `.npz`:
per-layer cosine >= 0.9999 and `max|Δ| / ||cached||_2 <= 1e-3` at every layer.
- Pass: the shared `textual_v2/_act_cache/` may be extended with the 350 new ids.
- Fail: do not write into it. Create `experiment_artifacts/vti/{model_short}/textual_v2/_act_cache_demos850_2026-07-28/`
  and extract all 850 x 6 there (+3000 forwards/model), recording the deviation in the manifest.
  Either branch is additive; neither needs Alex.
The new runner enforces the pin: for any model id matching `qwen2`, it exits non-zero unless
`--max_pixels 1003520` is passed. A different Qwen visual budget would need its own cache namespace
and its own plan.

**0.5 — Partition integrity (after step 7, gates step 8).** Blocks pairwise disjoint; union equals the
850 ids of `demos_850.jsonl` exactly; sizes exactly (50, 100, 200, 500);
`_meta.content_hash_sha256_16` equals sha256[:16] of `demos_850.jsonl`; the 555 v2 ids and the 295 new
ids are disjoint. A shared id between the two groups would make a cache entry ambiguous, so this check
also protects step 8.

## Steps

### 0. Checks 0.1-0.3, then 0.4 on GPU 0.

### 1. Mine the top-up candidates

```bash
python data_scripts/vti_demos_v2/stage0_mine_candidates.py \
  --n-candidates 650 \
  --emit-next-batch \
  --out data/vti/v2/stage0_candidates_topup_2026-07-28.jsonl \
  --summary-out data/vti/v2/stage0_summary_topup_2026-07-28.json
```

`--summary-out` **does not exist yet** — add it to `stage0_mine_candidates.py` (`parse_args`, and
line 324 `write_summary(v2 / "stage0_summary.json", summary)` becomes
`write_summary(args.summary_out or (v2 / "stage0_summary.json"), summary)`). Default behaviour is
unchanged. Without it this command overwrites the 555-era `stage0_summary.json`.

`--emit-next-batch` excludes ids found in `demos.jsonl` and in every `data/vti/v2/stage*.jsonl`
(`stage0_mine_candidates.py:276-289`), so the 1000 already-seen candidates and the v1 demos cannot be
re-emitted. Run step 2 immediately after this so the new ids enter `stage1_verified/rejected` and
become exclusion-visible.

### 2-5. Stages 1 through 4 on the top-up candidates only

```bash
python data_scripts/vti_demos_v2/stage1_verify_anchors.py \
  --input data/vti/v2/stage0_candidates_topup_2026-07-28.jsonl
python data_scripts/vti_demos_v2/stage1b_allocate.py
python data_scripts/vti_demos_v2/stage2_write_truthful.py
python data_scripts/vti_demos_v2/stage3_make_variants.py
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py
```

Providers are the `config.py` defaults (stage1 `anthropic:claude-sonnet-5`, stage2
`anthropic:claude-opus-4-8`, stage3 `anthropic:claude-haiku-4-5`, stage4 `anthropic:claude-sonnet-5`);
stage 4 asserts provider independence from stages 2 and 3. Stages 1b-4 take their input from the
default v2 files and skip everything already processed, so each one only touches the new ids.
`ANTHROPIC_API_KEY` comes from the repo-root `.env` (`mllm_client.py:24`). Smoke first with
`--provider mock --limit 5` on stage 1 if desired; mock calls write real rows, so run mock only
against a scratch `--input` file, never against the real top-up candidates.

After stage 4: if `stage4_verdicts.jsonl` gained fewer than 295 rows, repeat step 1 with
`--n-candidates 300` and a new dated filename, then re-run stages 1-4. Two rounds maximum before check
0.2's stop condition applies.

### 6. Assemble the 850-item pool

New script `data_scripts/vti_demos_v2/assemble_demos_850.py`:

```
--base        data/vti/demos_v2.jsonl                                   (default)
--stage4      data/vti/v2/stage4_verdicts.jsonl                         (default)
--stage0-topup data/vti/v2/stage0_candidates_topup_2026-07-28.jsonl     (rank order for new ids)
--n-new       295                                                       (default)
--out         data/vti/demos_850.jsonl
--summary-out data/vti/v2/demos_850_assembly_summary.json
```

Behaviour:
1. Write `--base` bytes verbatim to `--out` (the base file ends every record with `\n`).
2. `base_ids` = ids in base. Candidates = stage-4 rows whose id is not in `base_ids` and which satisfy
   check 0.2's admission rule.
3. Sort candidates by `(topup_stage0_rank, id)` — the same deterministic convention as
   `stage5_assemble.py:45-47` — take the first `--n-new`, error out if fewer.
4. Emit each as the stage-5 final schema (`id`, `image`, `question` = `config.QUESTION`, `value`,
   `h_values`, `anchors`, `provenance` with the four stage models, `pipeline_version`, and
   `date = today`) using `json.dumps(..., ensure_ascii=False)` one per line, appended to `--out`.
5. Summary: row counts, the 295 new ids, sha256[:16] of `--out`, per-dimension caption-presence counts,
   and a re-assertion that lines 1-555 of `--out` are byte-identical to `--base`.

Then: `sha256sum data/vti/demos_v2.jsonl` must still be `9a44f4af…`, and
`diff <(head -555 data/vti/demos_850.jsonl) data/vti/demos_v2.jsonl` must be empty.

Add to `src/paths.py`: `vti_demos_850_path() -> Path` returning `vti_data_dir() / "demos_850.jsonl"`,
and `vti_demos_850_partition_path() -> Path` returning
`vti_data_dir() / "demos_850_partition_s42.json"`. New functions only; nothing existing changes.

Add to `.gitignore`, below the existing `data/vti/*` block (lines 118-123):

```
!data/vti/demos_850.jsonl
!data/vti/demos_850_partition_s42.json
```

Without these two negations the pool and the partition are untracked and the extraction is not
reproducible from a clone.

### 7. Build the partition, then run check 0.5.

### 8. Activations and directions, per model

New module `evaluation/interventions/vti/directions_partition.py`, importing the extractor math and
cache helpers from `directions_v2` (`act_cache_dir`, `ensure_variant_activation`,
`obtain_textual_vti_v2_from_stacks`, `save_textual_v2_directions`, `load_textual_v2_directions`,
`demos_content_hash`, `variant_suffix`, `_git_commit`, `DIFF_POLARITY`, `SIGN_CONVENTION`,
`TOKEN_POLICY`, `STEER_COMPONENT`). `directions_v2.py` itself is **not edited** — the existing
`shuffled_prefix` path stays byte-identical, which is the cheapest possible guarantee that the 40
existing direction directories remain reproducible.

```python
PARTITION_SIZES: Tuple[int, ...] = (50, 100, 200, 500)
SELECTION_POLICY = "disjoint_partition"
SLUG_STEM = "demos850"
PARTITION_SEED = 42
QWEN_ACT_CACHE_MAX_PIXELS = 1003520

def build_or_load_partition(demos_path: Path, partition_path: Path,
                            seed: int = PARTITION_SEED,
                            sizes: Sequence[int] = PARTITION_SIZES) -> dict
def select_block_demos(rows_by_id: Dict[str, dict], block_ids: Sequence[str],
                       dimension: str) -> List[dict]     # raises on a missing caption
def partition_slug(demos_hash: str, dimension: str, num_demos: int,
                   seed: int = 42, rank: int = 2) -> str
def partition_cache_dir(model_short: str, slug: str) -> Path
def compute_or_load_partition_directions(wrapper, model_short: str, *, dimension: str,
                                         num_demos: int, rank: int = 2, seed: int = 42,
                                         demos_path: Optional[Path] = None,
                                         partition_path: Optional[Path] = None,
                                         act_cache_override: Optional[Path] = None,
                                         max_pixels: Optional[int] = None,
                                         force_recompute: bool = False) -> np.ndarray
def extract_partition_grid(wrapper, model_short: str, dimension: str,
                           sizes: Sequence[int] = PARTITION_SIZES, **kw) -> Dict[int, Path]
```

- `build_or_load_partition` reproduces the `load_or_build_master_order` guard shape: if the file
  exists and its `_meta.content_hash_sha256_16` differs from `demos_content_hash(demos_path)`, raise.
- `partition_slug` returns `demos850_{hash[:8]}_{dimension}_nd{N}_s{seed}_r{rank}_partition`. Distinct
  stem and distinct policy token, so no slug can collide with `demosv2_9a44f4af_*_prefix`.
- `partition_cache_dir` writes into
  `experiment_artifacts/vti/{model_short}/textual_v2/{slug}/` — sibling directories to the existing
  ones, sharing `_act_cache` deliberately.
- `extract_partition_grid` prefetches (`value`, `dimension`) across all four blocks' ids first, then
  fits each block. Fits reuse the cache, so the five dimensions cost 850 x 6 forwards in total, not
  per dimension.
- Metadata written per cell: everything `compute_or_load_textual_directions_v2` writes, with
  `selection_policy = "disjoint_partition"`, `demos_file = "demos_850.jsonl"`, the demos hash,
  `partition_file`, `partition_content_hash`, `block_size`, `ids_used` (the block, in partition-file
  order), `max_pixels`, `act_cache_dir`, and `git_commit`. `skipped_ids` must be empty by construction.

New runner `evaluation/run_scripts/extract_demos850_partition_directions.py`, modelled on
`extract_demosv2_directions.py`:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_scripts/extract_demos850_partition_directions.py \
  --model llava-hf/llava-1.5-7b-hf \
  --demos_path data/vti/demos_850.jsonl \
  --partition_path data/vti/demos_850_partition_s42.json \
  --dimensions all existence attribute counting relation \
  --num_demos 50 100 200 500 \
  --rank 2 --seed 42 \
  --check_cache_fidelity \
  > logs/demos850_partition_llava_2026-07-28.log 2>&1

CUDA_VISIBLE_DEVICES=0 python evaluation/run_scripts/extract_demos850_partition_directions.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --demos_path data/vti/demos_850.jsonl \
  --partition_path data/vti/demos_850_partition_s42.json \
  --dimensions all existence attribute counting relation \
  --num_demos 50 100 200 500 \
  --rank 2 --seed 42 --max_pixels 1003520 \
  --check_cache_fidelity \
  > logs/demos850_partition_qwen25_2026-07-28.log 2>&1
```

`--check_cache_fidelity` runs check 0.4 for that model and aborts before any write on failure.
The runner exits non-zero if the model id matches `qwen2` and `--max_pixels != 1003520`.

Placement: `CUDA_VISIBLE_DEVICES=0`. GPU 0 was free at 48,673 MiB on 2026-07-28; GPUs 1-3 held ~48 GB
each from other users. Re-check `nvidia-smi` immediately before launching and pin whichever single
device is free. A single visible device means `device_map="auto"` resolves to that one device, which
is how the existing 500-item cache was produced. This path uses no hooks, but keep it unsharded
anyway — sharding is the documented failure mode for every hook-based path in this repo, and mixing
placements across one cache is not worth the risk.

Peak GPU memory: one 7B model in fp16 with batch-1 forwards — ~15 GB (LLaVA, 576 image tokens) and
~17 GB (Qwen at the 1003520-px cap). Both fit the free device with >30 GB headroom. The PCA fits run
on CPU (`_live_pca_fit` receives CPU tensors); the largest is 500 x 135,168 fp32 = 270 MB of fit data
plus SVD workspace, against 480 GB available RAM, and the identical fit already ran for `nd500` on
this machine.

Order: run LLaVA first, verify, then Qwen. Do not run both concurrently on one device.

### 9. Verification and manifest

New script `helper_scripts/verify_demos850_partition_extraction.py --model <hf id>`, writing
`experiment_artifacts/vti/{model_short}/textual_v2/demos850_partition_verification_report_{model_short}_2026-07-28.md`
and the machine-readable manifest
`experiment_artifacts/vti/{model_short}/textual_v2/demos850_partition_extraction_manifest_2026-07-28.json`.

Checks, all pass/fail, none of them a comparison between cells:

1. `demos_850.jsonl` hash matches every cell's `metadata.json` `content_hash_sha256_16`, and lines
   1-555 still match `demos_v2.jsonl` byte for byte.
2. Partition integrity re-checked from disk (check 0.5).
3. Activation cache: for each of the 850 ids, all six `sample_{id}_vl_v2_{variant}.npz` exist and load;
   every stack has shape `(num_layers+1, hidden_dim)` = (33, 4096) for LLaVA, (29, 3584) for Qwen; all
   finite; no all-zero rows. Expected file count 5100 per model (3000 pre-existing + 2100 new).
4. All 20 cells for this model exist with `directions.npz`, `components.npz`, `metadata.json`;
   `directions` shape `(32, 4096)` / `(28, 3584)`; all finite; every per-layer norm > 0;
   `n_pairs == block_size`; `ids_used` equals the partition block exactly, in order;
   `skipped_ids == []`; `selection_policy == "disjoint_partition"`; `max_pixels` recorded.
5. No pre-existing artifact changed: `demos_v2.jsonl` hash, `demos_v2_order_s42.json` hash, and the
   `directions.npz` sha256 of all 20 `demosv2_9a44f4af_*` directories for this model, compared against
   values recorded in step 0.3.

Manifest contents: one record per cell (slug, model, dimension, block size, n_pairs, output path,
demos hash, partition hash, max_pixels, act cache dir, git commit, timestamp), plus run-level totals
(new forwards executed, cache hits, wall clock, cache-fidelity result).

### 10. Documentation

Append to `IMPLEMENTATION.md`, next to the existing "demos_v2 textual directions" section: a
"demos_850 disjoint-partition directions" subsection with the same table shape — module, extract CLI,
source demos + hash, partition file, slug example, act-cache reuse note including the Qwen
`max_pixels=1003520` pin, and the explicit statement that `demos_850.jsonl` must not be passed to
`run_eval.py`. Append a factual run record to `RESEARCH_LOG.md`: dates, counts, hashes, paths, wall
clock, and the cache-fidelity outcome. No interpretation in either file.

## Artifacts

| Path | Content |
|---|---|
| `data/vti/v2/stage0_candidates_topup_2026-07-28.jsonl` | 650 newly mined candidates |
| `data/vti/v2/stage0_summary_topup_2026-07-28.json` | mining summary for the top-up |
| `data/vti/v2/_summaries_snapshot_555_2026-07-28/` | pre-top-up stage summaries + checksums |
| `data/vti/demos_850.jsonl` | 850 rows: 555 verbatim + 295 new |
| `data/vti/demos_850_partition_s42.json` | the one partition, blocks 50/100/200/500 |
| `data/vti/v2/demos_850_assembly_summary.json` | assembly counts, new ids, output hash |
| `experiment_artifacts/vti/{model_short}/textual_v2/_act_cache/` | +2100 files per model (5100 total per model) |
| `experiment_artifacts/vti/{model_short}/textual_v2/demos850_{h8}_{dim}_nd{N}_s42_r2_partition/` | 20 cells per model: `directions.npz`, `components.npz`, `metadata.json` |
| `experiment_artifacts/vti/{model_short}/textual_v2/demos850_partition_extraction_manifest_2026-07-28.json` | per-cell manifest |
| `experiment_artifacts/vti/{model_short}/textual_v2/demos850_partition_verification_report_{model_short}_2026-07-28.md` | verification report |
| `logs/demos850_partition_{llava,qwen25}_2026-07-28.log` | run logs |

New code: `data_scripts/vti_demos_v2/assemble_demos_850.py`,
`evaluation/interventions/vti/directions_partition.py`,
`evaluation/run_scripts/extract_demos850_partition_directions.py`,
`helper_scripts/verify_demos850_partition_extraction.py`;
edits: `--summary-out` on `stage0_mine_candidates.py`, two path helpers in `src/paths.py`, two
`.gitignore` negations, `IMPLEMENTATION.md`, `RESEARCH_LOG.md`.

### Cost

- API: ~2,260 calls over four stages for a 650-candidate batch. Scaling the recorded per-item token
  averages (stage1 523 in / 258 out per candidate; stage2 1550/239; stage3 375/78; stage4 782/379):
  roughly 1.80 M input and 0.54 M output tokens across Sonnet 5, Opus 4.8, and Haiku 4.5. Sequential
  single-item calls; at 3-8 s per call this is an estimated 2-5 hours wall clock.
- GPU: 2100 new forwards per model, 4200 total. Estimated 15-45 minutes per model including load and
  npz compression; no measured per-forward timing exists in the 2026-07-13 logs, so treat the range as
  an estimate and record the actual in the manifest.
- Disk: ~690 MB (LLaVA) + ~435 MB (Qwen) of new activations, from the measured per-file averages of
  336 KB and 212 KB, plus ~80 MB of direction files. Total ~1.2 GB against 744 GB free.

## What confirms or falsifies

The extraction is usable when, for both models: the 850-row pool exists with its first 555 lines
byte-identical to `demos_v2.jsonl`; the partition file passes check 0.5; all 5100 activation files per
model load with the right shape and finite values; all 20 direction cells per model carry `n_pairs`
equal to their block size and `ids_used` equal to their block; and every pre-existing hash recorded in
step 0.3 is unchanged.

It is failed, and nothing downstream may use it, if any of these hold: the cache-fidelity check fails
and the run nonetheless wrote into the shared `_act_cache`; any cell's `ids_used` deviates from its
partition block or overlaps another block; a new id collides with one of the 555; `demos_v2.jsonl`,
`demos_v2_order_s42.json`, or any `demosv2_9a44f4af_*` artifact changed; or fewer than 295 admissible
new rows were produced and the shortfall was papered over by reducing a block size.

## Open questions

1. If two further mining rounds still leave fewer than 295 admissible new items — the stop condition
   in check 0.2 — the 850 pool and its 50/100/200/500 partition cannot be built, and the block sizes
   are field 2's to change, not the plan's. The run stops and reports the achieved count.
2. The Qwen activation cache is pinned to `max_pixels=1003520` by the existing 500-item cache and by
   the guard added in step 8. Any future Qwen extraction at a different visual budget needs its own
   cache namespace and its own extraction spec; it cannot share this one.
3. Consuming these directions in `run_eval.py` needs plumbing that does not exist —
   `VTITextualIntervention` resolves directions through `compute_or_load_textual_directions_v2`, whose
   order-file hash guard rejects `demos_850.jsonl` by design. Deliberately out of scope: this plan
   produces the primitives; wiring them into an evaluation belongs to the experiment that compares
   them, behind a design spec.
