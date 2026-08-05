# Textual mean-difference steering: does it improve visual reasoning? (v2)
design_spec: designs/07_30_26/steering_vector_visual_reasoning_validation.md

Date: 2026-07-30
Supersedes: `implementation_plans/7-30-26/steering_vector_visual_reasoning_validation_plan_2026-07-30.md`
(kept in place; lineage is load-bearing).
Status: ready to implement. The 72-arm grid is gated on the single sanity check below.
Kind: design. The plan produces the one primitive the design spec whitelists (raw mean-difference
textual directions) under this spec; no extraction spec is written.

**What changed from v1.** All five of v1's open questions were answered by Alex on 2026-07-30 and
are now settled conditions, not questions. The second sanity check v1 carried, on baseline parse
rate and the gold-label split, is removed. The
compute target changed from RunAI to lambdab2 GPU 0 with both models running concurrently on the
one card, which changes the memory arithmetic and the wall-clock estimate, and the plan now
carries an execution order. No cell, metric, item set, or condition changed.

**Execution order, settled by Alex 2026-07-30 and applied in place.** Benchmark is the
**outermost** loop — every configuration that can be run over AMBER runs before CHAIR starts, and
every configuration over CHAIR before POPE starts. Within a benchmark: baseline, then the
all-layers arm, then the two layer windows; within a layer set, sample sizes 50, 500, 200, 100;
betas ascending. The wall-clock consequences of that ordering were put to him and he chose it
anyway — the run goes as far as it goes. This is a sequence change only.

---

## Question

Does textual steering with a direction built solely from the mean difference of paired truthful
versus hallucinated captions improve the visual reasoning capability of a VLM, or does it move a
discriminative benchmark without moving a generative one?

---

## Design spec reference

`designs/07_30_26/steering_vector_visual_reasoning_validation.md`

- Factors, levels, and the eight cell rows come from that file's **Cells** section verbatim.
- Item sets come from its **Item sets** section: the pinned AMBER-450, POPE-600, and CHAIR-500
  subsets already on disk, and the demos_850 disjoint blocks of 50 / 100 / 200 / 500.
- The one primitive produced here is its **Primitives this design requires**: raw mean-difference
  textual directions, no PCA, dimension slice `all`.
- Code gaps 1a, 1b, 2, 3, 4, 5 in that file are implementation steps below, not questions.
- Primary measurement, the gold-label split reporting, and the h+/h- definitions are carried
  through unchanged.

### Decisions recorded 2026-07-30, applied throughout

1. **CHAIR generation length is 256 new tokens, as a standing default.** 64 was frozen at the
   time and was known to truncate captions and bias CHAIR; that is why the spec raises it. Alex is
   changing the code defaults (`run_eval.py`, `eval_runner.py`, `run_exp1_repro_grid.sh`,
   `readme.md`) outside this plan, so 256 is the default rather than a per-invocation override.
   The driver still passes `--chair_max_new_tokens 256` explicitly so each cell is
   self-documenting. Historical logs and historical cells generated at 64 are not rewritten and
   remain valid records of what was run.
2. **The 84 unparsed Qwen AMBER answers in the 2026-06-22 tree were caused by out-of-memory
   errors on that uncapped run.** With `--max_pixels 1003520` enforced they do not recur. Alex's
   instruction: do not run checks to establish why, and do not read into it. The baseline
   parse-rate check v1 carried is therefore removed and not replaced. `n_unparsed` is still reported next to every
   accuracy figure, as an ordinary metric-with-counts obligation, but it gates nothing.
3. **`--max_pixels 1003520` is enforced for Qwen going forward.** Settled condition, not an open
   question. Factual note retained: it is a departure from Policy A native resolution, it is
   identical in the Qwen baseline and in every Qwen steered cell so it cannot differentiate cells,
   and Qwen numbers from this run are not directly comparable to any uncapped Qwen run.
4. **CHAIR prompt is the verbatim VTI `"Please Describe this image in detail."`** Confirmed.
5. **h+ = tn→fp, h- = fp→tn**, exactly as Primary measurement defines them. Confirmed.

---

## Cells

Frozen across every cell: greedy decoding (`do_sample=False`, all wrapper `generate_*`);
`--max_new_tokens 256`; `--chair_max_new_tokens 256`; CHAIR prompt
`"Please Describe this image in detail."`; `--amber_task discriminative`;
Qwen `--max_pixels 1003520`; direction dimension `all`; intervention `vti_textual_additive_mlp`;
`--run_date 2026-07-30`.

Layer indices are absolute decoder indices, verified against the cached activation stacks:
LLaVA-1.5-7B has 32 decoder layers (33 rows × 4096 in
`experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/_act_cache/sample_*_vl_v2_value.npz`);
Qwen2.5-VL-7B has 28 (29 rows × 3584). Both windows exclude the model's last layer; the
all-layers arm includes it. The Qwen late window 15–24 is the same band
`layer_windows(28, width=10, stride=5)` yields (`src/prompt_spans.py:293-307`).

| Cell | Model (HF id) | Benchmarks / subsets | Intervention | `--layer_set` | Beta | Direction sample sizes |
|---|---|---|---|---|---|---|
| S1 | `llava-hf/llava-1.5-7b-hf` | AMBER 450 / POPE 3×200 / CHAIR 500 | `vti_textual_additive_mlp` | `all` (0–31) | 0.2, 0.5, 0.9 | 50, 100, 200, 500 |
| S2 | `llava-hf/llava-1.5-7b-hf` | same | `vti_textual_additive_mlp` | `5-14` | 0.2, 0.5, 0.9 | 50, 100, 200, 500 |
| S3 | `llava-hf/llava-1.5-7b-hf` | same | `vti_textual_additive_mlp` | `20-29` | 0.2, 0.5, 0.9 | 50, 100, 200, 500 |
| S4 | `Qwen/Qwen2.5-VL-7B-Instruct` | same | `vti_textual_additive_mlp` | `all` (0–27) | 0.2, 0.5, 0.9 | 50, 100, 200, 500 |
| S5 | `Qwen/Qwen2.5-VL-7B-Instruct` | same | `vti_textual_additive_mlp` | `5-14` | 0.2, 0.5, 0.9 | 50, 100, 200, 500 |
| S6 | `Qwen/Qwen2.5-VL-7B-Instruct` | same | `vti_textual_additive_mlp` | `15-24` | 0.2, 0.5, 0.9 | 50, 100, 200, 500 |
| B1 | `llava-hf/llava-1.5-7b-hf` | same | `no_intervention` | — | — | — |
| B2 | `Qwen/Qwen2.5-VL-7B-Instruct` | same | `no_intervention` | — | — | — |

72 steered arms × 5 benchmark invocations + 10 baseline invocations = **370 invocations**
(185 per model), 1550 scored items each arm. POPE is three invocations because `run_eval.py:33`
takes one `--pope_split` and `eval_runner.py:284` keys the tree `pope_{split}`.

Result directory per steered cell (new suffix components, see Code changes):

```
evaluation/results/2026-07-30/{model_short}/{amber|chair|pope_random|pope_popular|pope_adversarial}/
    vti_textual_additive_mlp__b{beta}__dall__nd{N}__meandiff__layers_{all|5_14|20_29|15_24}/
        responses.json
        metric_summary.json
```

Baseline cells write to the bare `no_intervention/` directory in the same trees
(`no_intervention.config` is `{}`, so `eval_runner.py:304-305` takes the bare-`{iv}` branch).
The two concurrent processes write under different `{model_short}` subtrees and cannot collide.

---

## Data

### Benchmark item sets — all already on disk, no new construction

| Set | Path | n | How it is passed |
|---|---|---|---|
| AMBER discriminative 450 | `data/amber/pinned_amber_disc_450.json` | 450 (150 each existence/attribute/relation; `_meta.gold_counts` = 276 no / 174 yes) | `--subset_ids_file` + `--amber_task discriminative` |
| CHAIR 500 | `data/chair/pinned_chair_500.json` | 500 COCO val2014 ids | `--subset_ids_file` + `--chair_prompt` |
| POPE 600 | `data/pope/pinned_eval_ids.json` | 200 per split, 100 gold-yes / 100 gold-no in each | `--limit 200 --pope_split {random,popular,adversarial}` |

**Resolved fact — POPE must be pinned with `--limit 200`, not `--subset_ids_file`.**
`pinned_eval_ids.json` is shaped `{benchmark, limit_per_split, generated, selection, splits:{...}}`,
with no top-level `pope` key, so `eval_runner.py:210-213` would build an empty `subset_map` from
it; and `load_pope` (`src/dataset.py:146-218`) has no `subset_ids` parameter, so a `pope` key would
raise. Its own `selection` field records "deterministic first-N of `data/pope/combined.json`
filtered by category/task==split; no RNG", and I verified that the first 200 entries of each split
in `combined.json` are byte-equal to the three pinned id lists. `--limit 200` therefore reproduces
the pin exactly. Consequence: POPE invocations are POPE-only, because `_benchmark_kwargs`
(`eval_runner.py:63`) applies `limit` to every benchmark in the invocation.

**Frozen CHAIR prompt.** `"Please Describe this image in detail."` (capital D — the verbatim VTI
prompt, not the lowercase prompt stored in `combined.json`), per Alex's confirmation. This is the
repo's frozen prompt for the CHAIR-500 pin (`run_exp1_repro_grid.sh:41`,
`step0_chair_token_cap.py:42`) and it is identical in baseline and steered arms.

**Do not reuse the 2026-06-22 baselines.** `evaluation/results/2026-06-22/{model}/chair/no_intervention/`
ran at `--chair_max_new_tokens 64`, and the Qwen cells in that tree ran without a `--max_pixels`
cap — which is what produced the out-of-memory losses recorded there. Both differ from this
design's frozen settings, and the h+/h- join requires a baseline matched to the steered arms item
for item. Both no-intervention baselines are run fresh under `--run_date 2026-07-30`.

### Direction demos and activation cache — all already on disk, zero forward passes

- Pool: `data/vti/demos_850.jsonl`, content hash `ba05bd960cad0c18`
  (`src.paths.vti_demos_850_path()`).
- Blocks: `data/vti/demos_850_partition_s42.json`, `selection_policy: disjoint_partition`,
  seed 42, sizes 50/100/200/500, verified 850 unique ids with no overlap between blocks. The
  four nd arms differ in demo identity as well as in n.
- Cached last-token stacks: `experiment_artifacts/vti/{model_short}/textual_v2/_act_cache/`,
  file name `sample_{demo_id}_vl_v2_{value|all}.npz`. **Verified present for all 850 ids × both
  variants × both models** (0 missing). Every stack is `(num_layers+1, hidden_dim)` float32.
  The mean-difference extraction reads these and executes **0 forward passes**; it must raise
  on a cache miss rather than forwarding.

---

## Metrics

Named in plain English; every rate is written out in the counts it is formed from, per
`CLAUDE.md`'s reporting convention and the spec's Primary measurement.

### Per-item primitive

The parsed answer per discriminative item (`_normalize_yes_no`, `evaluation/classifiers/metrics.py:10-27`)
and the generated caption per CHAIR item. Both are retained per item in `responses.json`
(`eval_runner.py:143-154`), which is what makes the aggregates re-attributable by reanalysis.

### Discriminative cells (AMBER 450, and each POPE split separately)

Reported for every steered arm and its matched baseline:

- **tp, fp, tn, fn** — the four parsed yes/no outcome counts, positive class `yes`.
- **n_total, n_unparsed** — always reported next to everything below.
- **accuracy** = (tp + tn) / n_total. Unparsed answers are counted as incorrect, matching the
  existing scorers.
- **precision** = tp / (tp + fp); **recall** = tp / (tp + fn);
  **F1** = 2·precision·recall / (precision + recall).
- **yes rate** = parsed-yes answers / parsed answers = (tp + fp) / (n_total − n_unparsed).
  *This differs from the `yes_ratio` field already written by the scorers*, which divides by
  `n_total` including unparsed (`metrics.py:85`, `metrics.py:132`). Both are written to the
  result table under distinct column names; the spec's definition is the one plotted.
- **accuracy on the gold-no subset**, with `n_gold_no`; **accuracy on the gold-yes subset**,
  with `n_gold_yes`. AMBER already emits these as `neg_item_accuracy` / `pos_item_accuracy` /
  `n_neg_total` / `n_pos_total` (`metrics.py:124-142`); POPE does not, and they are computed
  offline from `responses.json` (Code gap 5).
- **h+ = hallucinations induced** = number of items that were tn under the matched baseline and
  fp under the steered arm.
  **h- = hallucinations removed** = number of items that were fp under the baseline and tn under
  the steered arm.
  Both are counts, not rates. They match `flip_tn_to_fp` and `flip_fp_to_tn` in
  `evaluation/vti_rotation_strength/rotation_strength.py:120-161` and the console legend at
  line 245. They require a per-item join by sample `id` against the no-intervention baseline for
  the same model and the same benchmark subset, which is why the baselines must be run and
  retained per item.

The spec's reason for the gold-label split, carried through: AMBER 450 is 61.3% gold-no, and
accuracy on an unbalanced yes/no set moves both when the model discriminates better and when its
answer rate shifts toward the majority label; the aggregate alone does not say which happened.

### Generative cell (CHAIR 500)

- **chair_i** = (objects mentioned that are not in the image, summed over captions) /
  (objects mentioned, summed over captions).
- **chair_s** = (captions containing at least one hallucinated object) / (non-empty captions).
  Both are hallucination rates, lower is better, both computed over non-empty captions only
  (`metrics.py:278-351`). Object grounding uses the vendored Rohrbach synonym list at
  `evaluation/classifiers/chair_synonyms.txt`.
- Reported jointly and never dropped, because caption length confounds CHAIR in both directions
  (`metrics.py:288-304`): **n_total, n_nonempty, n_empty, empty_fraction,
  avg_objects_mentioned, avg_caption_len_chars**.

### Baseline bound

The spec's Sample size section leaves a confidence-interval bound on the baseline as the check
performed at analysis time. Implement it as a **Wilson score 95% interval** on:
- baseline accuracy for each discriminative cell, from `n_correct / n_total`;
- baseline `chair_s`, from `n_caps_hallucinated / n_nonempty` (recoverable as
  `chair_s × n_nonempty`).

`chair_i` is a ratio of two sums, not a binomial proportion; no interval is computed for it, and
it is reported with `avg_objects_mentioned` and `n_nonempty` instead.

`p_yes_norm` and `answer_mass` appear nowhere.

---

## Sanity checks that must pass first

**One.** v1 carried two; the second (baseline parse rate and gold-label split on the discriminative
benchmarks) was removed on Alex's instruction of 2026-07-30, because the 84 unparsed Qwen AMBER
answers it was built around are explained by out-of-memory errors on an uncapped run and will not
recur under the enforced `--max_pixels 1003520`. It is not replaced by a smaller check. Everything
else the plan needed was settled by reading the repo during planning and is recorded as a resolved
fact elsewhere in this file.

### The CHAIR caption-length check — captions must terminate within 256 new tokens, on both models

**What it verifies.** That a 256-token cap is above the natural stopping length of both models
under the frozen CHAIR prompt, so that `chair_i` and `chair_s` move through hallucination
behaviour rather than through a truncated denominator.

**Why it is not already settled.** The existing CHAIR token-cap provenance probe — the script
`evaluation/chair_amber_diagnostics/step0_chair_token_cap.py` and its output
`evaluation/results/2026-06-22/_diagnostics/step0_chair_token_cap.json`, both already named that
way on disk — compared caps 64 and
512 on LLaVA only, and CHAIR moved a long way between them: `chair_s` 0.30 → 0.65,
`chair_i` 0.167 → 0.240, `avg_objects_mentioned` 2.4 → 3.75. The cap is therefore not a free
parameter. 256 has never been probed on either model, and Qwen2.5-VL has never been probed at any
cap.

**How.** Reuse `evaluation/chair_amber_diagnostics/step0_chair_token_cap.py` on its existing
20-image draw (`--num_images 20 --seed 1234`), at `--caps 256 512`, once per model, run
sequentially with exclusive use of GPU 0 before the two grid processes start. Two small additions
to that script (see Code changes): a `--max_pixels` pass-through for Qwen, and a model-suffixed
output filename so the two models do not overwrite each other. Report per model per cap:
`chair_s`, `chair_i`, `avg_objects_mentioned`, `avg_caption_len_chars`, plus two new truncation
readouts — the number of captions whose response re-encodes to ≥ cap − 2 tokens with that model's
tokenizer, and the number not ending in `.`, `!`, or `?`.

**Pass.** Zero captions hit the cap at 256 on both models. Then the 256 and 512 arms generated
identical text and CHAIR cannot move with the cap.

**What invalidates the design.** Any caption hitting the cap at 256 on either model. Report how
many, and by how much `chair_s` / `chair_i` / `avg_objects_mentioned` differ between 256 and 512,
and stop — per the spec, the cap is then not safe and rows 3 and 4 need a length-matched
comparison instead of a fixed cap. Because benchmark is the outermost loop, a failure here blocks
only the CHAIR phase; the AMBER and POPE phases are untouched by it. See Open questions.

**Cost.** 20 images × 2 caps × 2 models = 80 generations, roughly 20 minutes sequential on an
otherwise idle GPU 0.

---

## Launch prerequisites

Environment and scheduling facts, not sanity checks.

### Target: lambdab2, GPU 0, both models concurrent on the one card

Alex verified on 2026-07-30: `nvidia-smi` reports GPU 0 = NVIDIA RTX A6000, 49140 MiB total,
**48673 MiB free, zero compute processes**; GPUs 1, 2 and 3 are each occupied at ~48.3 GB by other
users. GPU 0 is the target. This overrides the standing routing note in `IMPLEMENTATION.md`
("do not schedule multi-day full-benchmark sweeps on lambdab2"); it is his decision and is
recorded here as such, in the same way RESEARCH_LOG records the 2026-07-21 and 2026-07-29
concurrency overrides.

- **Two concurrent processes, one per model**, both launched with `CUDA_VISIBLE_DEVICES=0`.
- **No `device_map` sharding and no CPU offload.** This constraint does not relax because there is
  now one GPU: mixed-device tensors corrupt the forward hooks the entire design depends on
  (`IMPLEMENTATION.md`, Device policy). With exactly one device visible there is nothing to shard
  across, but confirm it anyway — `wrapper.load()` prints the model device
  (`src/model.py:131-144`), and it must be a single `cuda:0` for both processes, with no
  "Model is on CPU" warning.
- **Launch under `nohup` or tmux.** On 2026-06-22 a full Exp1/Exp2 launch was lost mid-run when
  the SSH tunnel dropped and the foreground processes were SIGHUP-killed (RESEARCH_LOG 2026-06-22).
  The runner is resume-safe (`--skip_if_exists`, plus `responses.checkpoint.json`), but a lost
  session costs the in-flight cell.

### Peak memory against the 48673 MiB headroom

What the records establish, at these settings:

| Source | Figure |
|---|---|
| RESEARCH_LOG 2026-07-29, **both models concurrent on GPU 0**, Qwen at `--max_pixels 1003520` | "GPU 0 ~30792 MiB used (LLaVA ~14110 + Qwen ~16670)" |
| RESEARCH_LOG 2026-06-19 / 2026-06-18, LLaVA-1.5 single-device cuda:0 | ~13.2 GiB `memory_allocated` |
| `IMPLEMENTATION.md` device-placement note | `Qwen2VLWrapper` fits a single A6000 at native resolution, ~15 GiB allocated |

**Summed resident footprint of the pairing this run uses: 30,792 MiB measured, against 48,673 MiB
free — 17,881 MiB (~17.5 GiB) of headroom.** That is the closest available measurement: same two
models, same card, same `--max_pixels` cap, both processes live at once.

**What is unestablished.** That 30,792 MiB figure was taken during *direction extraction* — single
forward passes, batch 1, no generation. No record in this repo states the peak footprint of either
model during *generation* at a 256-token cap with hooks on every decoder layer. I am not inventing
one. What can be bounded from the model configs rather than measured:

- KV cache, LLaVA-1.5-7B (Vicuna-7B: 32 layers, 32 heads × 128, MHA, fp16): 32 × 4096 × 2 × 2 B =
  0.5 MiB per token. At 576 vision tokens + prompt + 256 generated ≈ 880 tokens, ~430 MiB.
- KV cache, Qwen2.5-VL-7B (`config.json` text side: 28 layers, `num_key_value_heads: 4`, head dim
  128, bf16): 28 × 512 × 2 × 2 B = 0.055 MiB per token. Under 100 MiB at any sequence length this
  run reaches.
- The ViT forward is bounded by construction on both: LLaVA's tower is fixed at 336×336 / 577
  tokens, and Qwen's is capped by `max_pixels 1003520` — which is precisely the cap that prevents
  the ~251 GiB ViT-attention blow-up documented in `IMPLEMENTATION.md`.

So the generation-time delta over the measured 30,792 MiB is sub-GiB from the KV cache, plus
allocator slack of unknown size, against ~17.5 GiB of headroom. That is comfortable but it is an
inference, not a measurement. Two operational consequences, both cheap:

- `eval_runner.py:147` calls `cleanup_gpu()` after **every** sample, which runs
  `torch.cuda.empty_cache()` and returns unused cached blocks to the driver. Under two-process
  sharing this actively works in our favour and should not be removed.
- **Watch `nvidia-smi` through the first CHAIR cell of each process** (CHAIR is the longest
  generation and therefore the peak). Because benchmark is the outermost loop, neither process
  reaches a CHAIR cell until its AMBER phase completes — roughly 15 h in — so the CHAIR
  caption-length check, which
  generates 20 CHAIR captions per model at caps 256 and 512 before either process launches, is the
  earliest sighting of the generation-time peak and its `nvidia-smi` reading should be recorded.
  Stagger the two launches by ~5 minutes so the two model loads do not coincide. If combined usage
  approaches the card, run the two models sequentially instead of concurrently — that changes no
  cell, only wall clock.

### Wall clock under this routing

Per-invocation cost, exclusive use of one A6000. Basis: 0.65–0.75 samples/s measured for steered
POPE cells (`evaluation/results/_logs/beta_grid_20260619_110137.log`), i.e. ~1.4 s per short
yes/no item, which also covers AMBER; ~4 s per caption for LLaVA and ~7 s for Qwen at a 256-token
cap; ~1.5 min model load per invocation; ~1.5 min to decode a POPE split's 3000 images before
slicing to 200.

| Invocation | items | LLaVA | Qwen |
|---|---|---|---|
| AMBER 450 | 450 | 12 min | 12 min |
| CHAIR 500 @ 256 tokens | 500 | 35 min | 60 min |
| POPE, one split | 200 | 8 min | 8 min |

Per model the grid is 37 configurations per benchmark (1 baseline + 3 layer sets × 4 sample sizes
× 3 betas), with POPE costing three invocations per configuration: 37 + 37 + 111 = 185
invocations. Aggregated by benchmark, which is how the run is traversed:

| Benchmark phase | invocations/model | LLaVA exclusive | Qwen exclusive |
|---|---|---|---|
| AMBER | 37 | 7.4 h | 7.4 h |
| CHAIR | 37 | 21.6 h | 37.0 h |
| POPE | 111 | 14.8 h | 14.8 h |
| **Total** | **185** | **~44 h** | **~59 h** |

**~103 GPU-hours in total**, unchanged by the ordering — the same 370 invocations run either way.

**Two processes sharing one card do not each run at full-card speed.** The estimates below assume
even time-slicing, i.e. each stream runs at roughly half its exclusive rate while both are live,
so the aggregate throughput is about the same as running them back to back. It will be somewhat
better than that in practice because a meaningful fraction of each stream is CPU-bound and
overlaps the other's GPU work — the per-sample `_save_json(records, checkpoint_path)` write
(`eval_runner.py:148`), the POPE image decode, and tokenization. It will be worse if the two
streams contend on host RAM or disk. On that assumption, cumulative from launch:

| Phase complete | LLaVA | Qwen |
|---|---|---|
| AMBER | ~15 h | ~15 h |
| AMBER + CHAIR | ~58 h | ~89 h |
| AMBER + CHAIR + POPE (all 185) | ~88 h | ~118 h |

**Combined wall clock ≈ 100–110 hours, i.e. 4 to 4.5 days**, with the Qwen tail accelerating once
the LLaVA stream exits and frees the card. The run goes as far as it goes; nothing in this plan is
scheduled against a deadline.

### Disk

~200 MB of `responses.json` / `metric_summary.json` across 370 cells; 8 new direction directories
at 0.5–4 MB each. 742 GB free on `/`. Host RAM: each POPE invocation holds ~2.7 GB of decoded
images; two concurrent processes is ~5.4 GB against 480 GB available.

---

## Execution order

Settled by Alex on 2026-07-30. This changes no cell, no metric, no item set and no condition —
only sequence.

### The two rules it follows

1. **Benchmark is the outermost loop: AMBER → CHAIR → POPE.** Every configuration that can be run
   over AMBER is run before CHAIR starts; every configuration over CHAIR before POPE starts.
2. **Within a benchmark, the baseline first** (`no_intervention` — fewest configurations and no
   intervention involved), then the all-layers arm before either window (it is the paper's method
   and the arm this design exists to reproduce), then `layers=5-14`, then the late window
   (`layers=20-29` LLaVA / `layers=15-24` Qwen). Within a layer set the sample sizes run in the
   order 50, 500, 200, 100. Betas ascend 0.2 → 0.5 → 0.9 within each block.

A cell is only complete when its `metric_summary.json` exists; `_run_one` writes it last
(`eval_runner.py:154-163`) and leaves `responses.checkpoint.json` behind on a mid-cell stop. The
analysis scripts skip any cell without `metric_summary.json`, so a stop mid-block never produces a
half-filled row.

### The sequence, identical in shape for both processes

Every block is named by what it is; there are no phase codes anywhere in this plan, in the driver,
or in anything written to disk.

| Block | Content | Invocations per model |
|---|---|---|
| Mean-difference direction extraction | CPU, no GPU, 0 forward passes, plus verification | — |
| The CHAIR caption-length check | the sanity check above, sequential, exclusive GPU | — |
| The AMBER baseline | AMBER, `no_intervention` | 1 |
| The AMBER all-layers block | AMBER, `layers=all`, nd 50 → 500 → 200 → 100, betas 0.2 → 0.5 → 0.9 | 12 |
| The AMBER early-middle-window block | AMBER, `layers=5-14`, same nd and beta order | 12 |
| The AMBER late-window block | AMBER, `layers=20-29` LLaVA / `15-24` Qwen, same nd and beta order | 12 |
| The CHAIR baseline | CHAIR, `no_intervention` | 1 |
| The CHAIR all-layers block | CHAIR, `layers=all`, same nd and beta order | 12 |
| The CHAIR early-middle-window block | CHAIR, `layers=5-14`, same nd and beta order | 12 |
| The CHAIR late-window block | CHAIR, late window, same nd and beta order | 12 |
| The POPE baseline | POPE, `no_intervention`, ×3 splits | 3 |
| The POPE all-layers block | POPE, `layers=all`, same nd and beta order, ×3 splits | 36 |
| The POPE early-middle-window block | POPE, `layers=5-14`, same nd and beta order, ×3 splits | 36 |
| The POPE late-window block | POPE, late window, same nd and beta order, ×3 splits | 36 |
| | | **185** |

Both processes run this same sequence concurrently; they differ only in speed.

### Coverage milestones

What is complete and what is outstanding at each natural boundary. Cumulative estimates are on the
even-time-slicing assumption in the wall-clock section and are positions in the sequence, not
commitments.

**Both baselines complete on all three benchmarks.** The no-intervention cells for both models,
which is what every steered cell's h+/h- join and every Wilson interval is measured against. Under
this ordering the three baseline invocations are not contiguous: the AMBER baseline opens the
AMBER phase, the CHAIR baseline opens the CHAIR phase, and the POPE baseline opens the POPE phase,
so the CHAIR and POPE baselines only land once their phases begin.

**End of the AMBER late-window block — the AMBER axis complete (LLaVA ~15 h, Qwen ~15 h).**
All 37 AMBER configurations: baseline plus all three layer sets × four sample sizes × three betas,
both models. This gives rows 1 and 2 of the prediction table across the entire grid, with the four
counts, both per-gold-label accuracies, the yes rate over parsed answers, and h+/h- on every AMBER
cell.
*Outstanding:* CHAIR and POPE entirely — no generative reading yet, and rows 3 and 4 unfilled.

**End of the CHAIR late-window block — AMBER and CHAIR both complete (LLaVA ~58 h, Qwen ~89 h).**
All four rows of the prediction table across the full grid, and the AMBER-change-against-
CHAIR-change figure the spec names as the comparison that separates the explanations.
*Outstanding:* POPE entirely — the second discriminative benchmark, which the abandonment
criterion reads jointly with AMBER.

**End of the POPE late-window block — the full grid (LLaVA ~88 h, Qwen ~118 h).** All 185
invocations per model.

Within any phase there are three further boundaries, at the end of each layer-set block: each
closes one layer set at all four sample sizes and all three betas for that benchmark, which is the
smallest unit that fills a complete row of the layer-set axis. Within a block, each (layer set,
sample size) triple of betas is complete after three invocations (nine for POPE), and the analysis
scripts read whatever subset of those exists.

---

## Artifacts

### Mean-difference directions (CPU only, 0 forward passes)

Eight new directories, one per (model, nd), sibling to the existing PCA sets:

```
experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/demos850_ba05bd96_all_nd{50,100,200,500}_s42_meandiff_partition/
experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/demos850_ba05bd96_all_nd{50,100,200,500}_s42_meandiff_partition/
    directions.npz     # keys layer_0..layer_{L-1} plus num_layers, hidden_dim
    metadata.json
```

No `components.npz` — there are no components.

**Additivity, traced.** The slug stem is `..._s42_meandiff_partition`, which cannot collide with
the existing `..._s42_r2_partition` (demos_850 PCA) or `demosv2_9a44f4af_..._r2_prefix` (demos_v2
PCA) directories. Nothing globs or iterates `textual_v2/*` anywhere in the tree (grepped: only
direct `load_textual_v2_directions(cache_dir)` calls with an explicit directory), so an extra
sibling strands nothing. The demos file, its content hash, the partition file, and the shared
`_act_cache/` are all read-only in this stage; the extractor never calls
`ensure_variant_activation` and never constructs a model wrapper. Verification records sha256 of
every pre-existing `_r2_partition` `directions.npz` before and after, and the `_act_cache` file
count and mtimes before and after.

Manifest and report:

```
experiment_artifacts/vti/demos850_meandiff_extraction_manifest_2026-07-30.json
experiment_artifacts/vti/{model_short}/textual_v2/demos850_meandiff_verification_report_{model_short}_2026-07-30.md
```

### Evaluation cells

Up to 370 cell directories under `evaluation/results/2026-07-30/`, laid out as in the Cells
section, each with `responses.json` and `metric_summary.json`. Plus a run manifest written by each
driver process before its first invocation:

```
evaluation/results/2026-07-30/_analysis_steering_visual_reasoning_validation/run_manifest_{model_short}.json
```

recording git commit, model, `max_pixels`, CHAIR cap and prompt, subset files and their hashes,
beta / nd / layer-set grids, the four direction slugs, and the launch timestamp — because
`no_intervention.config` is `{}` and the baseline cells otherwise carry no record of the frozen
settings.

### Analysis tables and plots

All under `evaluation/results/2026-07-30/_analysis_steering_visual_reasoning_validation/`. Both
scripts must run correctly against a **partial** grid, skipping any cell without a
`metric_summary.json` and recording which grid points are absent in a `coverage.json` alongside
the tables. Under benchmark-outermost traversal a partial grid is normally *whole benchmarks
missing* rather than scattered gaps, so the scripts must tolerate a benchmark with no cells at all
— including the case where a steered AMBER cell exists but its CHAIR and POPE counterparts do not,
and the case where a benchmark's baseline has not yet run because its phase has not started.

**Naming rule for everything written to disk.** No filename, directory component, JSON key, CSV
column, plot title, axis label, legend entry or log line may contain a code for a phase, block,
stage or cell. Each is identified by the values that define it — model, benchmark, layer set,
direction sample size, beta — spelled out. The cell-directory suffix already does this
(`…__b0.5__dall__nd200__meandiff__layers_5_14`), and the table columns and plot names below follow
the same rule. The short cell labels in the Cells table above stay in that table and are not to
appear in code or on disk.

`coverage.json` schema, specified here so it cannot acquire block codes:

```json
{
  "run_date": "2026-07-30",
  "generated_at": "<ISO timestamp>",
  "cells_complete": [
    {"model": "llava-1.5-7b-hf", "benchmark": "amber", "intervention": "no_intervention",
     "layer_set": null, "direction_sample_size": null, "beta": null},
    {"model": "llava-1.5-7b-hf", "benchmark": "amber", "intervention": "vti_textual_additive_mlp",
     "layer_set": "5-14", "direction_sample_size": 500, "beta": 0.5}
  ],
  "cells_missing": [ "<same shape>" ],
  "benchmarks_not_started": ["chair", "pope_random", "pope_popular", "pope_adversarial"],
  "n_complete": 0,
  "n_expected": 370
}
```

`layer_set` takes exactly `"all"`, `"5-14"`, `"20-29"` or `"15-24"`, and is `null` on a baseline
row; `benchmark` takes the runner's own tree keys `amber`, `chair`, `pope_random`, `pope_popular`,
`pope_adversarial`.

Tables:

- `discriminative_counts_and_accuracy_by_gold_label.csv` — one row per
  (model, benchmark cell, arm). Columns: `model, benchmark, layer_set, direction_sample_size,
  beta, n_total, n_unparsed, tp, fp, tn, fn, accuracy, precision, recall, f1,
  yes_rate_over_parsed_answers, yes_ratio_over_all_items, accuracy_gold_no, n_gold_no,
  accuracy_gold_yes, n_gold_yes`. Baseline rows carry `layer_set=baseline`.
- `hallucinations_induced_and_removed_vs_baseline.csv` — one row per (model, discriminative cell,
  steered arm). Columns: `model, benchmark, layer_set, direction_sample_size, beta,
  hallucinations_induced_baseline_tn_to_steered_fp, hallucinations_removed_baseline_fp_to_steered_tn,
  n_decision_flips, n_flip_correct_to_wrong, n_flip_wrong_to_correct, n_items_joined`.
- `chair_scores_with_length_controls.csv` — one row per (model, arm). Columns: `model, layer_set,
  direction_sample_size, beta, chair_s, chair_i, n_total, n_nonempty, n_empty, empty_fraction,
  avg_objects_mentioned, avg_caption_len_chars`.
- `baseline_accuracy_wilson_intervals.csv` — one row per (model, discriminative cell): `n_correct,
  n_total, accuracy, wilson_lower_95, wilson_upper_95`; plus the CHAIR_s rows over `n_nonempty`.

Plots (self-describing names, titles, and axis labels; x axis is beta throughout, one line per
layer set, one facet per direction sample size, baseline drawn as a dashed horizontal line with
its Wilson band shaded; facets with no data are drawn empty and labelled "not yet run"):

- `amber_accuracy_by_beta_layer_set_and_direction_sample_size_{model_short}.png`
- `pope_accuracy_by_beta_layer_set_and_direction_sample_size_{model_short}.png` (facet row per split)
- `chair_hallucination_rate_by_beta_layer_set_and_direction_sample_size_{model_short}.png`
  (two panels: CHAIR_s and CHAIR_i, both labelled "lower is better")
- `amber_accuracy_on_gold_no_and_gold_yes_items_by_beta_{model_short}.png`
- `yes_rate_over_parsed_answers_by_beta_{model_short}_{amber|pope}.png`
- `hallucinations_induced_and_removed_by_beta_{model_short}_{amber|pope}.png` (grouped bars,
  counts on the y axis)
- `amber_accuracy_change_and_chair_hallucination_change_from_baseline_by_arm_{model_short}.png` —
  one point per steered arm, x = AMBER accuracy minus baseline accuracy in percentage points,
  y = CHAIR_i minus baseline CHAIR_i in percentage points. This figure exists because the spec
  names that pairing as the comparison that separates the explanations: it is the relative change
  in row 1 read against the relative change in row 3.

Summary of facts only (no interpretation):
`steering_visual_reasoning_validation_result_summary.md`.

---

## Code changes

Every item below closes a numbered Code gap in the design spec. Nothing here changes a cell,
metric, item set, or condition.

### New — `evaluation/interventions/vti/directions_meandiff.py` (Code gap 1a)

CPU-only. Imports helpers from `directions_v2` / `directions_partition` without editing them.

```python
SELECTION_POLICY     = "disjoint_partition"
STEER_RECONSTRUCTION = "raw_mean_difference"
SIGN_CONVENTION      = "mean over demos of (value - h_value); no PCA, no component selection, no sign flip"

def meandiff_partition_slug(demos_hash: str, dimension: str, num_demos: int, seed: int = 42) -> str
    # f"demos850_{demos_hash[:8]}_{dimension}_nd{num_demos}_s{seed}_meandiff_partition"

def meandiff_cache_dir(model_short: str, slug: str) -> Path
    # experiment_artifacts_dir("vti", model_short) / "textual_v2" / slug

def load_cached_stack(cache: ActivationCache, demo_id: str, variant: str) -> np.ndarray
    # cache.load_or_none(demo_id, suffix=variant_suffix(variant)); raise FileNotFoundError on miss.
    # Returns (num_layers+1, hidden_dim) float32.

def obtain_textual_meandiff_from_stacks(
    h_stacks: Sequence[np.ndarray], value_stacks: Sequence[np.ndarray],
) -> Tuple[np.ndarray, dict]
    # full = mean over demos of (value_stack - h_stack), shape (num_layers+1, hidden_dim).
    # diag: n_pairs, direction_layer_norms, mean_diff_layer_norms_mean_over_demos, flat_dim.

def compute_or_load_meandiff_directions(
    model_short: str, *, dimension: str = "all", num_demos: int, seed: int = 42,
    demos_path: Optional[Path] = None, partition_path: Optional[Path] = None,
    act_cache_override: Optional[Path] = None, force_recompute: bool = False,
) -> np.ndarray
```

Behaviour of `compute_or_load_meandiff_directions`:

1. Resolve `demos_path` / `partition_path` from `src.paths.vti_demos_850_path()` /
   `vti_demos_850_partition_path()`; compute `demos_content_hash`.
2. `build_or_load_partition` then `check_partition_integrity`
   (`directions_partition.py:53,175`) — both are read-only when the file exists.
3. `select_block_demos(rows_by_id, part["blocks"][str(num_demos)], dimension)`
   (`directions_partition.py:112`) so the demo order matches the PCA sets exactly.
4. Load stacks from `act_cache_dir(model_short)` via `load_cached_stack`; raise on any miss.
5. Mean-difference, drop the embedding row (`full[1:]`), cast float32.
6. Save with `save_textual_v2_directions(directions, meta, cache_dir)` and `components=None`
   (`directions_v2.py:326-355`), so only `directions.npz` and `metadata.json` are written.

`metadata.json` fields: `demos_path, demos_file, content_hash_sha256_16, partition_file,
partition_content_hash, block_size, dimension, num_demos_requested, n_pairs, ids_used,
skipped_ids: [], question, token_policy, diff_polarity ("value_minus_h_value"), sign_convention,
selection_policy, seed, steer_reconstruction ("raw_mean_difference"), rank: null,
steer_component: null, direction_layer_norms, mean_diff_layer_norms_mean_over_demos, model_short,
slug, act_cache_dir, forwards_executed: 0, act_cache_read_only: true, date, git_commit`.

Diff polarity is inherited unchanged (`DIFF_POLARITY = "value_minus_h_value"`,
`directions_v2.py:41`), so the direction points from hallucinated toward truthful with no
sign-alignment step. `steer` L2-normalises each layer slice before applying it
(`steer.py:43`), so only per-layer orientation reaches the model and beta is the sole magnitude
knob — identical to how the PCA directions are consumed.

### New — `evaluation/run_scripts/extract_demos850_meandiff_directions.py`

```bash
python evaluation/run_scripts/extract_demos850_meandiff_directions.py \
    --models llava-hf/llava-1.5-7b-hf Qwen/Qwen2.5-VL-7B-Instruct \
    --dimension all --sizes 50 100 200 500
```

CPU only; no `create_wrapper`, no CUDA — it can run while the GPU is busy. `model_short` comes
from `src.model._normalize_model_name(model_id)` (a pure string function); `num_layers` and
`hidden_dim` are read off the cached stacks. Writes the eight direction directories and
`experiment_artifacts/vti/demos850_meandiff_extraction_manifest_2026-07-30.json` with per-cell
sha256 of `directions.npz` and `metadata.json`, `forwards_executed: 0`, and the `_act_cache` file
count and mtime digest before and after.

### New — `helper_scripts/verify_demos850_meandiff_extraction.py`

Gating checks, all must pass: eight directories exist; shapes are `(32, 4096)` for LLaVA and
`(28, 3584)` for Qwen; `n_pairs` equals the block size; `ids_used` equals the partition block
list element-for-element and in order; no `components.npz`; every pre-existing `_r2_partition`
`directions.npz` is byte-identical before and after; `_act_cache` file count and mtimes unchanged.
Non-gating: prints per-layer L2 norms of the mean-difference direction next to
`direction_layer_norms` from the matching `_r2_partition` metadata. Writes the per-model
verification report named under Artifacts.

### Modified — `evaluation/interventions/vti/intervention.py` (Code gaps 1b, 2)

Add three constructor parameters to `VTITextualIntervention`:

```python
directions_dir: Optional[Path] = None,     # a textual_v2-format direction directory
layer_indices: Optional[Sequence[int]] = None,
layer_set_label: Optional[str] = None,
```

- `ensure_directions`: when `directions_dir` is set it takes precedence over the demos_v2 and
  legacy branches — `directions, meta = load_textual_v2_directions(Path(self._directions_dir))`
  (`directions_v2.py:358`), then assert
  `directions.shape == (wrapper.num_layers, wrapper.hidden_dim)` with a message naming the
  directory. Cache `meta` on the instance.
- `config`: when `directions_dir` is set, add `directions_dir`, `direction_slug` (meta `slug`),
  `steer_reconstruction` (meta), `demos_content_hash_sha256_16` (meta), `dimension` (meta),
  `num_demos` (meta `n_pairs`), `selection_policy` (meta), `diff_polarity` (meta),
  `token_policy` (meta). Always add `layer_indices` (sorted list or `null`) and
  `layer_set_label`. The existing `uses_demos_v2` branch is left untouched.
- `generate`: forward `layer_indices=self._layer_indices` to `vti_hook_ctx`
  (`hooks.py:62`, which validates the range against `wrapper.num_layers` and registers hooks only
  on those layers while indexing the full-length `directions` array by absolute layer).

### Modified — `evaluation/interventions/__init__.py`

`_make_visual_factory` currently drops only `beta`; extend the drop set to
`{"beta", "directions_dir", "layer_indices", "layer_set_label"}` so a visual registry key
constructed with the new kwargs cannot raise. The textual factory's existing `alpha` drop is
unchanged.

### Modified — `evaluation/runners/eval_runner.py`

- `run_evaluation` gains `directions_dir: Optional[str] = None`,
  `layer_indices: Optional[Sequence[int]] = None`, `layer_set_label: Optional[str] = None`,
  forwarded into `iv_kwargs` exactly as `beta` / `num_demos` are today (`eval_runner.py:249-261`).
- After `wrapper.load()`, if `layer_indices` is set, raise a clear error unless
  `0 <= min(layer_indices)` and `max(layer_indices) < wrapper.num_layers`.
- Result-directory composition (`eval_runner.py:293-305`): inside the existing
  `elif beta is not None and "beta" in cfg:` branch, add a **new** sub-branch taken only when
  `directions_dir is not None`:

  ```python
  iv_dir = (f"{iv_name}__b{beta}__d{cfg['dimension']}__nd{cfg['num_demos']}"
            f"__meandiff__layers_{cfg['layer_set_label']}")
  ```

  When `directions_dir is None` the existing demos_v2 composition is executed unchanged, so every
  path already on disk under `evaluation/results/` keeps its exact name. This is the component
  that stops the three layer sets from colliding on one directory at equal beta and nd.
- `chair_max_new_tokens` default becomes **256** as part of Alex's standing-default change.

### Modified — `evaluation/run_eval.py` (Code gaps 2, 4)

Two new flags, forwarded to `run_evaluation`:

- `--directions_dir PATH` — a `textual_v2`-format direction directory.
- `--layer_set STR` — `all`, or an inclusive `A-B` range. Parsed before the model loads:
  `all` → `layer_indices=None`, `layer_set_label="all"`; `A-B` →
  `layer_indices=list(range(A, B+1))`, `layer_set_label=f"{A}_{B}"`.

Code gap 4(a) is closed by the **default change**, not by a per-invocation override:
`--chair_max_new_tokens` now defaults to 256 (Alex is making that edit here and in
`eval_runner.py`, `run_exp1_repro_grid.sh`, `readme.md`). The driver still passes it explicitly so
each cell is self-documenting. Code gap 4(b) needs no code change, only invocation discipline: the
additive variant is selected by naming `vti_textual_additive_mlp` in `--interventions` rather than
relying on `VTITextualIntervention`'s `variant="uniform_rotation"` default (`intervention.py:38`).

### Modified — `evaluation/chair_amber_diagnostics/step0_chair_token_cap.py` (for the CHAIR caption-length check)

- `--max_pixels` forwarded to `create_wrapper` only when `"qwen2" in model_id.lower()`, matching
  `eval_runner.py:266-270` (the base wrapper `__init__` rejects unknown kwargs).
- Output path becomes `…/_diagnostics/step0_chair_token_cap_{model_short}.json` so two models
  under one run_date do not overwrite each other. The existing 2026-06-22 file is untouched.
- Per cap, add `n_captions_at_token_cap` (response re-encoded with the model's tokenizer,
  length ≥ cap − 2) and `n_captions_without_terminal_punctuation` to the emitted row.

### New — `evaluation/steering_visual_reasoning_validation/` (Code gaps 3, 5)

`build_result_tables.py`

```bash
python evaluation/steering_visual_reasoning_validation/build_result_tables.py \
    --run_date 2026-07-30 --output_dir evaluation/results
```

Walks `evaluation/results/{run_date}/{model_short}/{bench_key}/{iv_dir}/`, **skips any cell
lacking `metric_summary.json`** (so it is safe to run against a partial grid at any stopping
point), parses `iv_dir` into `(beta, dimension, direction_sample_size, layer_set)`, and for each
completed cell:

- Recomputes the four counts tp/fp/tn/fn, `n_unparsed`, per-gold-label accuracy, and the yes rate
  over parsed answers directly from `responses.json`, using `_normalize_yes_no` imported from
  `evaluation.classifiers.metrics`. This closes Code gap 5 offline for POPE (`score_pope_records`
  returns no `tn` and no per-gold-label accuracy, `metrics.py:102-113`) and is applied to AMBER
  as well so both benchmarks are computed by one code path. Cross-check: the recomputed AMBER
  values must equal `neg_item_accuracy` / `pos_item_accuracy` / `n_neg_total` / `n_pos_total` in
  the cell's `metric_summary.json`; a mismatch is a hard error.
- Joins each steered cell against the matched baseline cell by sample `id` and computes
  `flip_counts_from_records(steered_records, baseline_records)` — a new function reproducing the
  branching of `_flip_counts` (`rotation_strength.py:132-161`) but taking gold from each record's
  `ground_truth` field rather than from an `EvalSample`. It returns the same four confusion
  directions; the two the spec names are written out as
  `hallucinations_induced_baseline_tn_to_steered_fp` (= `flip_tn_to_fp`) and
  `hallucinations_removed_baseline_fp_to_steered_tn` (= `flip_fp_to_tn`). This closes Code gap 3.
  The join must assert that the steered and baseline id sets are identical and record
  `n_items_joined`. A steered cell whose baseline is not yet complete is written with the flip
  columns empty rather than dropped.
- Copies CHAIR fields straight from `metric_summary.json`.
- Computes Wilson 95% intervals on baseline accuracy and baseline CHAIR_s.

Writes the four CSVs, `coverage.json` (which grid points are present, which absent), and
`result_tables.json` (same content, machine-readable, `schema_version: 1`).

`make_plots.py --run_date 2026-07-30` reads `result_tables.json` and writes the PNGs listed under
Artifacts, drawing absent facets empty and labelled. Matplotlib only; no seaborn dependency is in
`environment.yml`.

### New — `evaluation/run_scripts/run_steering_visual_reasoning_validation.sh`

Driver, one model per invocation of the script so the two processes are independent. Env knobs
`BENCHMARKS` (default `amber chair pope` — the outermost loop), `BETAS` (default `0.2 0.5 0.9`),
`NUM_DEMOS` (default `50 500 200 100`), `LAYER_SETS` (default `all 5-14 20-29` for LLaVA,
`all 5-14 15-24` for Qwen), `RUN_DATE` (default `2026-07-30`), `OUTPUT_DIR`,
`CUDA_VISIBLE_DEVICES`, `MAX_PIXELS`.

Launch:

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup bash \
  evaluation/run_scripts/run_steering_visual_reasoning_validation.sh llava-hf/llava-1.5-7b-hf \
  > logs/steering_visual_reasoning_validation_llava_2026-07-30.log 2>&1 &

sleep 300   # stagger so the two model loads do not coincide

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MAX_PIXELS=1003520 nohup bash \
  evaluation/run_scripts/run_steering_visual_reasoning_validation.sh Qwen/Qwen2.5-VL-7B-Instruct \
  > logs/steering_visual_reasoning_validation_qwen25_2026-07-30.log 2>&1 &
```

The three per-benchmark invocation forms. `$MAXPX` is `--max_pixels 1003520` for Qwen and empty
for LLaVA; `$IV` is `no_intervention` for a baseline cell and
`vti_textual_additive_mlp --beta "$BETA" --layer_set "$LAYER_SET" --directions_dir "$DIR"` for a
steered cell:

```bash
# AMBER — one invocation per configuration
python evaluation/run_eval.py --model "$MODEL" --benchmarks amber --amber_task discriminative \
  --interventions $IV --subset_ids_file data/amber/pinned_amber_disc_450.json \
  --max_new_tokens 256 --run_date 2026-07-30 --output_dir "$OUTPUT_DIR" --skip_if_exists $MAXPX

# CHAIR — one invocation per configuration
python evaluation/run_eval.py --model "$MODEL" --benchmarks chair \
  --interventions $IV --subset_ids_file data/chair/pinned_chair_500.json \
  --chair_prompt "Please Describe this image in detail." --chair_max_new_tokens 256 \
  --run_date 2026-07-30 --output_dir "$OUTPUT_DIR" --skip_if_exists $MAXPX

# POPE — three invocations per configuration
for split in random popular adversarial; do
  python evaluation/run_eval.py --model "$MODEL" --benchmarks pope --pope_split "$split" \
    --limit 200 --interventions $IV --max_new_tokens 256 \
    --run_date 2026-07-30 --output_dir "$OUTPUT_DIR" --skip_if_exists $MAXPX
done
```

Loop order `benchmark` → baseline → `layer_set` → `num_demos` → `beta`:

```bash
for BENCH in $BENCHMARKS; do                  # amber, then chair, then pope
  run_cell "$BENCH" no_intervention           # this benchmark's baseline, first
  for LAYER_SET in $LAYER_SETS; do            # all, then 5-14, then the late window
    for ND in $NUM_DEMOS; do                  # 50, 500, 200, 100
      DIR="experiment_artifacts/vti/${MODEL_SHORT}/textual_v2/demos850_ba05bd96_all_nd${ND}_s42_meandiff_partition"
      for BETA in $BETAS; do                  # 0.2, 0.5, 0.9
        run_cell "$BENCH" steered "$LAYER_SET" "$ND" "$BETA"
      done
    done
    print_comparison_table                    # after each completed layer-set block
  done
done
```

`run_cell` dispatches to the matching invocation form above, so POPE expands to three invocations
per configuration and AMBER and CHAIR to one. The driver writes `run_manifest_{model_short}.json`
before its first invocation and prints `print_comparison_table(model_short, OUTPUT_DIR, run_date)`
after each completed (benchmark, layer set) block, so the log itself carries a running readout.
Console banners name the block in words — for example
`=== AMBER | early-middle-window block (layers 5-14) | nd=500 | beta=0.5 ===` — and never a code.

---

## What confirms or falsifies

Read against the prediction table in the design spec. AMBER rows are accuracy; CHAIR rows are
CHAIR score, lower is better.

| Condition | Baseline | Explanation A | Explanation B | Explanation C |
|---|---|---|---|---|
| LLaVA, discriminative (AMBER) | 70 | 77 | 77 | 70 |
| Qwen, discriminative (AMBER) | 74 | 78 | 78 | 74 |
| LLaVA, generative (CHAIR) | 20 | 20 | 16 | 20 |
| Qwen, generative (CHAIR) | 17 | 17 | 15 | 17 |

- Rows 3 and 4 do the separating work; row 3 (LLaVA on CHAIR) is the single most informative cell
  per the spec. The pairing that carries it is the relative change from baseline in row 1 against
  the relative change from baseline in row 3, which is the joint change figure listed under
  Artifacts. Under benchmark-outermost traversal all four rows are filled once the CHAIR phase
  completes; before that, rows 1 and 2 are complete across the whole grid from the end of the
  AMBER phase, and rows 3 and 4 fill configuration by configuration through the CHAIR phase.
- **Abandonment criterion, in the spec's clarified wording:** if steering moves neither AMBER nor
  POPE off their baselines, the truthfulness hypothesis is abandoned. CHAIR being unchanged is a
  prediction of Explanation A, not a trigger for abandonment.
- The baseline Wilson intervals are what "off their baselines" is read against; a steered accuracy
  inside the baseline interval is not distinguishable from the baseline at this n.
- Every discriminative row must be read with `n_unparsed`, the yes rate over parsed answers, and
  the two per-gold-label accuracies in view. An accuracy move accompanied by a yes-rate shift
  toward the majority label, or by a change in parse rate, is not evidence of discrimination.
- h+ and h- are counts of items that changed confusion cell against the matched baseline, and are
  reported next to the accuracy they decompose.

---

## Open questions

1. The CHAIR caption-length check is the only gate, and the spec's invalidation condition covers
   rows 3 and 4 only. Benchmark-outermost traversal makes this cheap to act on: a failure blocks
   the CHAIR phase while the AMBER and POPE phases stay runnable, so the question is narrow — on a
   failure, does the run continue through the AMBER and POPE phases while the cap question is
   resolved, or halt entirely?
2. Generation-time peak GPU memory for the two concurrent processes is not established anywhere in
   the records; the closest measurement is 30,792 MiB for the same pairing during extraction, with
   17,881 MiB of headroom, and the KV-cache arithmetic above puts the generation delta under a GiB.
   This is flagged as a thing to watch during the first CHAIR cell of each process, not a check —
   but if the pair does approach the card, running the two models sequentially instead of
   concurrently roughly doubles the wall clock and changes nothing else. Note that under this
   traversal neither process reaches a CHAIR cell until its AMBER phase completes, so the peak is
   not exercised in the first ~15 hours.
