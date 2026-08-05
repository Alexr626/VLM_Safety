# Implementation Reference

Ground-truth description of the VLM hallucination mitigation codebase as it exists today. The external research analyst uses this file (without reading source) to plan experiments. **Keep it in sync with code changes.**

Last updated: 2026-07-31 (AMBER plot qualitative HTML galleries via `build_amber_plot_qualitative_html.py`)

---

## Repository layout

```
src/                          # Core library
data/                         # Benchmark manifests, images, per-model caches
data_scripts/                 # Download, caption, activation extraction
diagnostic_experiments/       # Mechanistic diagnostic pipelines
experiment_artifacts/         # Cross-run artifacts (matrices, indices, etc.)
evaluation/                   # Benchmark × intervention runner + metrics
environment.yml               # Conda env (authoritative dependency pin)
```

**Out of scope / ignore:** `VTI/` subdirectory (external reference, not part of this pipeline).

**Not implemented:** counting benchmarks (FSC-147-style), MAE/RMSE metrics.

**Implemented 2026-06-22:** CHAIR-s/i object-inventory scoring (rule-based vs.
COCO instance annotations), MMHal-Bench judge-based scoring (provider-agnostic
LLM judge), and AMBER **discriminative** POPE-style scoring (yes-bias /
negative-item grounding breakdown, with gold + qtype joined from the AMBER
annotation file at load time). All wired into the standard `run_eval.py` path;
see Evaluation §. (AMBER **generative** scoring remains a placeholder.)

---

## Environment

| Fact | Value |
|------|-------|
| Conda env | `vlm_hallucination_mitigation` |
| Create | `conda env create -f environment.yml` |
| Python | 3.12.7 |
| PyTorch | 2.10.0, CUDA wheels `cu121` (Ampere A6000 on lambdab2) |
| Transformers | 4.50.1 (pinned — v5 output changes affect activation extraction) |
| flash-attn | **Not installed** — use SDPA/eager |
| HF cache | `HF_HOME=/data/romanus/huggingface` (recommended) |
| W&B | `WANDB_DIR=/data/romanus/wandb` (recommended) |
| GPUs (lambdab2) | 4× RTX A6000 48GB; set `CUDA_VISIBLE_DEVICES`; check `nvidia-smi` — see **Compute resources** |
| GLIBCXX | Conda env has `activate.d` hook prepending `$CONDA_PREFIX/lib` on Ubuntu 20.04 |

**MiniGPT-4:** requires external MiniGPT-4 repo + `MINIGPT4_CKPT` env var pointing to `.pth` checkpoint.

**InternVL2 / 2.5:** remote model code may default to flash attention. This environment has no flash-attn; if load fails, ensure attention runs via SDPA/eager (no `use_flash_attn=True` in current wrapper — watch for upstream defaults).

**Confidentiality:** current benchmarks are public. Do not log internal/Nokia data to W&B or external services without explicit instruction.

---

## Compute resources

Romanus has **two compute targets**. The analyst must tag each planned run with
**target** (`lambdab2` | `runai`), **sample scale**, and **purpose**
(qualitative gate vs production benchmark). Do not schedule multi-day full-benchmark
sweeps on lambdab2; do not use RunAI for first-pass ~20-sample sanity checks.

### Routing summary

| | **lambdab2** | **RunAI** |
|--|--------------|-----------|
| **Purpose** | Dev, debug, **small exploratory runs** | **Large benchmark / reproduction runs** |
| **Typical N** | ~10–20 (qualitative), ≤~200 (single local cell) | Pinned subsets (CHAIR-500, AMBER-450), paper scale (e.g. POPE 3000/split) |
| **When** | New intervention/diagnostic; inspect raw outputs before scaling | Method already looks promising locally; confirm at benchmark scale |
| **Repo** | Live git clone (`~/dev/vlm_hallucination_mitigation_summer_2026`) | NFS extracted tree from `vti_repo.tar.gz` |
| **Python env** | Conda `vlm_hallucination_mitigation` | Micromamba `envs/vlm_hal` on NFS |
| **GPU** | 1× A6000 48GB (Romanus should **keep one slot free** long-term) | 1× H100 80GB HBM3 per job (`h100-pool`) |
| **Submit / run** | SSH on lambdab2; `CUDA_VISIBLE_DEVICES=N python …` | `runai training submit` from lambdab2 |

### lambdab2 (local server)

| Fact | Value |
|------|-------|
| Host | `lambdab2` (Tailscale `100.71.123.214`) |
| GPUs | 4× RTX A6000 48GB Ampere (sm_86); **shared** — check `nvidia-smi` before launch |
| Policy | **Reserve Romanus's A6000 for quick iteration** — avoid multi-day jobs that block the shared GPU |
| Conda env | `vlm_hallucination_mitigation` (`environment.yml`) |
| Caches | `HF_HOME=/data/romanus/huggingface`, `WANDB_DIR=/data/romanus/wandb` |
| Cursor agent | Runs here; canonical codebase |

**Appropriate jobs:** `LIMIT=10`–`20` `run_eval.py` cells; CHAIR/AMBER diagnostic smoke; reading `responses.json` / sample dumps under `evaluation/results/`; hook and activation-extraction debugging.

**Example drivers (local):** `run_beta_grid_local.sh`, `run_exp1_repro_grid.sh` with default pinned subsets only when a **single** model/cell is needed — not full multi-model grids at scale.

### RunAI (Bell Labs cluster)

| Fact | Value |
|------|-------|
| CLI | `~/.runai/bin/runai` on lambdab2; `runai login`; `runai project set nlm-mh` |
| Project quota | `runai project list` → typically **2 GPUs** for `nlm-mh` |
| Node pool | `h100-pool` (always `--gpu-devices-request 1 --node-pools h100-pool`) |
| GPU (probed) | NVIDIA **H100 80GB HBM3**, ~81559 MiB, compute cap 9.0 |
| Container image | `blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1` — **do not use** `dspy_image2:0.1` |
| Job commands | Use `bash` to invoke `helper_scripts/runai/*.sh` — do not rely on image `python3` |
| In-pod GPU | `CUDA_VISIBLE_DEVICES=0` (RunAI exposes the allocated GPU as device 0) |

**NFS (persistent):** SFTP `air-datalake@gpustorage-1.cloud.bell-labs.com`; Romanus path `/airl-datalake/romanus/`; mounted in pods at `/home/datalake/romanus/`.

```
/home/datalake/romanus/
├── vti_repo.tar.gz           # git archive from lambdab2 (upload via WinSCP)
├── vlm_hallucination/        # extracted repo (code + data/coco from setup)
├── bin/micromamba
├── envs/vlm_hal/             # micromamba env (torch/transformers for RunAI)
├── mamba/                    # micromamba package cache
└── hf_cache/                 # HF_HOME for RunAI weights
```

**One-time setup (done 2026-06-18):** `setup-vlm4` — micromamba env, COCO val2014, LLaVA-1.5-7B weights on NFS. Re-run setup only if `environment.yml` changes (or wipe `envs/vlm_hal`).

**Code sync workflow:** commit on lambdab2 → `git archive -o vti_repo.tar.gz HEAD` → upload tarball to NFS → eval job extracts with `tar -xzf … -C "$REPO"` (does not wipe `data/coco/` or `envs/`).

**RunAI helpers** (`helper_scripts/runai/`):

| Script | Purpose |
|--------|---------|
| `setup_vlm.sh` | One-time NFS bootstrap (env + COCO val2014 + weights) |
| `run_vti.sh` | VTI POPE eval (six variants, default LIMIT=200) |
| `run_smoke_pope.sh` | Minimal e2e smoke (LIMIT=5, `no_intervention`) |
| `run_steering_triple_one_h100.sh` | **One** job / **one** H100: concurrent LLaVA CHAIR→POPE + Qwen CHAIR + Qwen POPE (`TRIPLE_OK`) |
| `run_steering_visual_reasoning_llava.sh` | LLaVA grid helper (also used as a worker under the triple launcher) |
| `run_steering_visual_reasoning_qwen.sh` | Qwen grid helper (`BENCHMARKS=chair` or `pope`) |
| `run_steering_llava_smoke.sh` / `run_steering_qwen_smoke.sh` | Steered POPE limit=5 smokes |
| `sync_steering_llava.sh` | Extract repo + LLaVA/Qwen meandiff (+ optional LLaVA CHAIR partial); `VERIFY_SYNC_OK` / `SYNC_OK` |
| `verify_steering_nfs_layout.sh` | Required-path checks (`MODEL_SHORTS`) |
| `runai_job_logging.sh` | Tee to `$BASE/logs/runai/<job>_<ts>.log` |
| `SUBMIT_STEERING_LLAVA.md` | Sync → Qwen smoke → `hal-steer-triple` (1×H100, 3 processes) |
| `remap_lambdab2_paths.py` | Rewrite baked lambdab2 absolute paths (`/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/…`) to the local `project_root()` (NFS: `/home/datalake/romanus/vlm_hallucination`). Dry-run by default; `--apply` writes. Scans `data/` (incl. amber/pope augmented JSONLs + dump metadata), `experiment_artifacts/` |
| `bootstrap_micromamba.py` | Bootstrap micromamba if not pre-uploaded to NFS |
| `run_bash_lf.py` | Strip CRLF then `bash` a helper `.sh` (WinSCP-safe) |

**Env note (2026-07-31):** `evaluation/run_scripts/run_steering_visual_reasoning_validation.sh` supports `SKIP_CONDA_ACTIVATE=1` so RunAI helpers can run under `micromamba run` without calling `conda activate`. Pack must overlay `data/chair/pinned_chair_500.json` (gitignored under `data/*`); verify fails closed if that pin is missing on NFS.

**Example drivers (RunAI):** `evaluation/run_scripts/run_beta_grid_runai.sh` (N=3000/split); submit via `run_vti.sh` or experiment-specific wrappers. Full submit templates in `readme.md` § RunAI.

**Monitor:** `runai workload list -p nlm-mh`; `runai training logs <job> -p nlm-mh`; `runai training describe <job> -p nlm-mh` (Pods → Node for scheduling issues).

**Env note (2026-07-17):** RunAI helpers export `LD_LIBRARY_PATH=$ENV_PREFIX/lib:…` so Pillow/`libLerc` resolves conda’s newer `libstdc++` (image `/lib/.../libstdc++.so.6` lacks `GLIBCXX_3.4.29`).

### ThinkPad

WinSCP only: lambdab2 ↔ NFS file transfer. Not a dev or submit environment.

### Device policy (both targets)

VTI direction extraction and rotation steering **must** run with the **full model on a single GPU** — no CPU offload or multi-GPU sharding (corrupts hooks / activations). One 7B VLM fits comfortably on A6000 48GB or H100 80GB.

---

## `src/model.py` — VLM wrappers

### Factory

```python
create_wrapper(model_id: str = "llava-hf/llava-1.5-7b-hf", **kwargs) -> VLMWrapperBase
_normalize_model_name(model_id: str) -> str   # HF id → directory-safe short name
VLMWrapper(model_id, **kwargs)                # alias for create_wrapper
```

**Routing** (first match wins):

| Substring in `model_id` | Wrapper class |
|-------------------------|---------------|
| `sharegpt4v`, `share4v` | `ShareGPT4VWrapper` |
| `llava` | `LLaVAWrapper` |
| `minigpt` | `MiniGPT4Wrapper` |
| `internvl` | `InternVL2Wrapper` |
| `qwen2` | `Qwen2VLWrapper` |
| `qwen` | `QwenVLWrapper` |

### `VLMWrapperBase` interface

All wrappers implement:

```python
wrapper = create_wrapper(model_id).load()

# Forward (returns hidden_states tuple, attentions tuple|None, input_ids)
wrapper.forward_vl(image: PIL.Image, text: str, output_attentions: bool = False)
wrapper.forward_text(text: str, output_attentions: bool = False)

# Generation
wrapper.generate_vl(image, text, max_new_tokens=256) -> str
wrapper.generate_text(text, max_new_tokens=256) -> str
wrapper.generate_caption(image, max_new_tokens=200) -> str

# Properties after load
wrapper.model_name   # normalized short name
wrapper.num_layers
wrapper.hidden_dim
wrapper.device
wrapper.cleanup()    # gc + cuda empty_cache
```

Constructor kwargs: `torch_dtype` (default `float16`; InternVL and Qwen families default `bfloat16`), `device_map` (default `"auto"`).

### Vision resolution (Policy A)

Each model runs at its **native/default** vision resolution; frozen per model across baseline and intervention runs. Record vision-token count in `RESEARCH_LOG.md` entries.

| Model family | Wrapper | Vision tokens / resolution |
|--------------|---------|----------------------------|
| LLaVA-1.5 | `LLaVAWrapper` | Fixed **576** tokens (336×336, patch 14) |
| Qwen-VL-Chat | `QwenVLWrapper` | Fixed **448×448** |
| Qwen2-VL / 2.5 | `Qwen2VLWrapper` | **Dynamic** — processor native default (`max_pixels=None` → ~12.8M pixels); token count varies per image |
| InternVL2 / 2.5 | `InternVL2Wrapper` | 448×448 tile (`max_num=1`) |

**`Qwen2VLWrapper` extra kwarg:** `max_pixels: Optional[int] = None`. Default `None` uses the processor's built-in cap (Policy A). Pass an `int` (e.g. `1280 * 28 * 28 = 1003520`) to limit vision tokens for VRAM-constrained runs.

> **Qwen2-VL `max_pixels` OOM gotcha (transformers ≥ 4.49, observed on 4.50.1).** The Qwen2-VL image processor drives `smart_resize` from `image_processor.size["longest_edge"]` (default **12 845 056**), **not** from the `max_pixels` attribute — passing only `max_pixels=...` to `AutoProcessor.from_pretrained` sets the attribute but leaves resizing uncapped, so it is silently inert. A high-res image (e.g. a 16 MP AMBER/CHAIR photo → ~65k patches) then blows up the **ViT self-attention** (`F.scaled_dot_product_attention`, O(patches²)) to a ~251 GiB allocation and OOMs even on a 48 GB A6000 — independent of `max_new_tokens` or the LLM. `Qwen2VLWrapper.load()` therefore also pins `image_processor.size["longest_edge"] = max_pixels` (and `shortest_edge`/`min_pixels`) so the cap actually downsizes; `load()` prints both `max_pixels` and `size.longest_edge`. Native (`max_pixels=None`) is the unbounded default and **will OOM** on the large images in the CHAIR/AMBER subsets.

**Device placement notes (accuracy runs):** `QwenVLWrapper` with `device_map="auto"` pins the vision encoder on GPU and may offload LLM layers to CPU when free VRAM is tight (~18 GiB). Acceptable for accuracy/scoring; **not** for activation extraction (mixed-device tensors corrupt hooks). `Qwen2VLWrapper` fits on a single A6000 at native resolution (~15 GiB allocated).

Generation is **greedy** everywhere: `do_sample=False` in all wrapper `generate_*` methods.

---

## `src/dataset.py` — Benchmark loaders

### Registry

```python
BENCHMARK_REGISTRY: Dict[str, dict]   # key → {data_dir, loader, ...}
DATASET_DATA_DIRS: Dict[str, str]     # registry key → data/ subdirectory name
ALL_BENCHMARKS: List[str]
```

| Registry key | `data/` dir | Loader | Notes |
|--------------|-------------|--------|-------|
| `pope` | `pope` | `load_pope` | Splits: `random`, `popular`, `adversarial` via `split=`. When `combined.json` exists, entries are **filtered** by `category`/`task` matching `split` (ids are split-prefixed, e.g. `pope_random_00000`). Question template verbatim from official POPE JSON: `"Is there a [object] in the image?"` |
| `amber` | `amber` | `load_amber` | `task=` filter: `discriminative` \| `generative`. **Discriminative gold + qtype are joined at load time** from `data/amber/data/annotations.json` by `raw.id` (the query files / `combined.json` carry only id/image/query — `label` is empty there). For discriminative items the loader sets `label` = gold `"yes"`/`"no"` and `category` = the AMBER dimension `existence` \| `attribute` \| `relation` (existence ← annotation type `discriminative-hallucination`). `limit` (or `subset_ids=`, a pinned id set that takes precedence) is applied before image loading. |
| `chair` | `chair` | `load_chair` | COCO caption prompt (`text` = `"Please describe this image in detail."`; override per-sample with `prompt_override=`). `raw.coco_id` (int) is the COCO val2014 image id the CHAIR scorer keys on. `subset_ids=` pins a fixed sample-id subset (filtered before image load; takes precedence over `limit`). |
| `hallusionbench` | `hallusionbench` | `load_hallusionbench` | Some samples text-only (no image) |
| `mmhal_bench` | `mmhal-bench` | `load_mmhal_bench` | `text` = question, `label` = gt answer, `task` = the **8 MMHal question types** (attribute/adversarial/comparison/counting/relation/environment/holistic/other), `category` = question *topic* (outdoor/indoor/…). `raw` carries `image_content` (list, the human gt description the judge consumes), `gt_answer`, `question_type`. |

### Public API

```python
load_benchmark(name: str, limit: Optional[int] = None, **kwargs) -> List[dict]
load_pope(data_dir=None, split="random", limit=None) -> List[dict]
load_combined(benchmark: str) -> List[dict]
load_image_for_sample(sample: dict) -> Optional[PIL.Image]
benchmark_data_dir(benchmark: str) -> Path
combined_json_path(benchmark: str) -> Path
model_data_root(benchmark, model_short) -> Path
model_activations_dir(benchmark, model_short) -> Path
model_responses_dir(benchmark, model_short, intervention="no_intervention") -> Path
```

### Sample dict (uniform)

```python
{
    "id": str,
    "image_path": str | None,
    "image_pil": PIL.Image | None,
    "text": str,
    "label": str,
    "label_idx": int | None,
    "benchmark": str,
    "task": str | None,
    "category": str | None,
    "raw": dict,
}
```

### On-disk data layout

```
data/{benchmark_dir}/combined.json
data/{benchmark_dir}/images/...          # benchmark-specific image trees
data/coco/val2014/                       # shared (POPE, CHAIR)
data/coco/train2014/                     # VTI direction demos (textual + visual arms)
data/vti/demos.jsonl                     # 100 author-released paired captions (see below)
data/vti/demos_v2.jsonl                  # demos_v2 final (multi-dimension; see pipeline)
data/vti/demos_v2_order_s42.json         # master shuffled_prefix order (seed 42) + demos hash
data/vti/demos_850.jsonl                 # 850-row pool: 555 demos_v2 prefix + 295 top-up (hash ba05bd960cad0c18)
data/vti/demos_850_partition_s42.json    # disjoint 50/100/200/500 blocks over demos_850 (seed 42)
data/vti/qual_subset_chair5_amber25.json # CHAIR-5 + AMBER-25 qualitative pin (2026-06-22 LLaVA bundle)
data/amber/pinned_amber_disc_100.json    # AMBER-100 discriminative (20/stratum; keeps AMBER-25)
data/amber/augmented_amber{25,100}.jsonl # leading-clause (+ filler) prompt JSONLs
data/amber/dumps/{model_short}/{run_tag}/ # steered-capture dumps (AMBER subsets)
data/pope/pinned_pope_existence_yes_30.json # POPE-30-yes: 30 unique gold=yes from random (rebuilt 2026-07-21)
data/pope/pinned_pope_existence_no_30.json  # POPE-30-no: 10 gold=no/split, image-matched to yes-30
data/pope/pinned_pope_existence_{yes,no}_120.json # POPE-yes/no-120 (40/split)
data/pope/augmented_pope30_yes.jsonl       # leading+filler for POPE-30-yes
data/pope/augmented_pope30_no.jsonl        # leading+filler for POPE-30-no
data/pope/augmented_pope_{yes,no}_120.jsonl
data/pope/dumps/{model_short}/{run_tag}/ # steered-capture dumps (POPE subsets)
data/vti/qual_subset_chair1_smoke.json   # 1-image smoke gate pin (first CHAIR-5 id)
data/vti/demos_v2_{dimension}.jsonl      # flat exports (value/h_value) per dimension (optional)
data/vti/v2/                             # demos_v2 stage artifacts, call logs, review HTML
data/vti/_review/vti_demos_review.html   # optional HTML gallery (render_vti_demos_review.py)
data/captions/{benchmark}.json           # text-only inputs for TT extraction
data/{benchmark_dir}/{model_short}/activations/
    sample_{id}_vl.npz
    sample_{id}_tt.npz
    sample_metadata.json
data/{benchmark_dir}/{model_short}/responses/{intervention}/responses.json
```

**Download scripts:** `data_scripts/download_*.py`; all-in-one: `data_scripts/download_all_benchmarks.sh`.

| Benchmark | Source |
|-----------|--------|
| POPE | GitHub AoiDragon/POPE JSON + local COCO paths |
| CHAIR | COCO val2014 + annotations |
| AMBER | GitHub `junyangwang0410/AMBER` (`master`) + Google Drive images |
| HallusionBench | GitHub `HallusionBench.json` + Google Drive `hallusion_bench.zip` |
| MMHal-Bench | HuggingFace `Shengcao1006/MMHal-Bench` (`test_data.zip` + `response_template.json`) |

Google Drive downloads use `data_scripts/gdrive_utils.py` (`download_gdrive`, `is_valid_zip`).

### `data/vti/demos.jsonl` — VTI paired-caption demos

Author-released bundle (same schema as vendored `VTI/experiments/data/hallucination_vti_demos.jsonl`). **100** JSONL rows. Both textual and visual direction extractors take a **seed-42 sample of 70** by default (`num_demos=70`).

| Field | Role |
|-------|------|
| `id` | COCO train2014 image id (zero-padded string) |
| `image` | Filename under `data/coco/train2014/` |
| `question` | Usually `"Describe this image in detail."` |
| `value` | Clean / truthful caption |
| `h_value` | Hallucinated caption (paired with `value`) |
| `co_objects` | GPT-proposed co-occurring objects (LURE-style generation scaffolding) |
| `uncertain_objects` | Uncertainty / hedging tokens used at caption synthesis time |

**Direction extraction uses only** `image` / `question` / `value` / `h_value` (textual arm) or the image alone (visual arm). `co_objects` / `uncertain_objects` are **not** read at extraction or steering time — they document how `h_value` was built (paper appendix: GPT-3.5 co-object lists + LLaVA rewrite injecting sampled co/uncertain words). Useful for qualitative review only (`render_vti_demos_review.py`).

### `data/vti/demos_v2` — controlled minimal-pair VTI caption generation (v2.1)

**Status:** **v2.1** diversity pipeline live (`pipeline_version = demos_v2.1_2026-07-13`).  
**Live artifacts:** `data/vti/v2/stage0_candidates.jsonl` is the **v2.1 option-set** schema (currently **1000** candidates). Assembled finals: `data/vti/demos_v2.jsonl` (**555** rows after the 2026-07-13 full-1000 run; `content_hash_sha256_16=9a44f4afde0324b5`).  
**Frozen v2.0 POC:** `data/vti/v2_poc_2026-07-12/` (131 finals, hash `ad44185346c48958`) — do not delete.

**Package:** `data_scripts/vti_demos_v2/`  
**Plans:** `implementation_plans/vti_demos_v2_pipeline_plan.md` (v2.0), `implementation_plans/vti_demos_v2_1_diversity_plan.md` (v2.1)  
**Pipeline version string:** `demos_v2.1_2026-07-13` (`config.PIPELINE_VERSION`).

**Goal:** For each selected COCO train2014 image, produce one truthful caption (`value`) and five variants (`h_values.{existence,attribute,counting,relation,all}`). Stage 0 emits feasible **option sets**; seeded Stage 1b globally selects balanced assignments. Flat exports retain `value`/`h_value`.

| Path helper | Path |
|-------------|------|
| `vti_demos_v2_dir()` | `data/vti/v2/` (stage artifacts, call logs, summaries, review HTML under `_review/`) |
| `vti_demos_v2_path()` | `data/vti/demos_v2.jsonl` (stage-5 assembled finals) |

#### Design (what each dimension edits)

Truthful captions are constrained to **exactly four sentences** (S1–S4):

| Sentence | Content | Hallucination edit |
|----------|---------|-------------------|
| S1 | Scene + object enumeration | **existence** — insert an absent distractor noun into the enumeration |
| S2 | One visual attribute claim | **attribute** — swap true value → false value (types: color/material/state/action/texture) |
| S3 | `There are {at least\|at most} {N_word} {category}…` | **counting** — swap true count word → false count word (`FALSE_COUNT` or `at_most` morph) |
| S4 | Fixed relation template (viewer perspective) | **relation** — flip within type: horizontal left↔right; vertical above↔below; support on top of↔underneath; proximity right next to↔far away from |

Counting / relation / attribute variants are **sentence-scoped string replacements** of Stage-2 spans (`{"sentence_idx", "text"}`). Existence starts from a deterministic insert into `spans.existence_insertion_hint`, then a **text-only Haiku grammar polish**; a full Haiku rewrite is fallback only. Combined `h_values.all` applies existence + attribute + counting + relation edits.

#### Package layout

| Module | Role |
|--------|------|
| `config.py` | Defaults: `N_CANDIDATES=300`, `N_FINAL=150`, gates, providers, `FALSE_COUNT`, score/allocator weights, relation/attr type targets |
| `geometry.py` | COCO-box predicates for horizontal / vertical / support / proximity options |
| `mllm_client.py` | `anthropic[:model]` / `openai[:model]` / `mock`; vision JPEG encode; `.env` keys; Anthropic omits `temperature` unless set; for `claude-sonnet-5` sends `extra_body={"thinking": {"type": "disabled"}}` (adaptive thinking is on by default on that model) |
| `validators.py` | Sentence split, structural checks, plural-aware `category_mentioned`, `deterministic_existence_insert`, sentence-scoped attribute checks, minimal-pair + record validators (`all` skipped for distractor-leak) |
| `number_words.py` | `number_to_word` / `false_count` / `false_count_at_most` (must **not** be named `numbers.py` — shadows stdlib) |
| `prompts.py` | Stage 1–4 system/user templates; `stage4_statement_specs` builds dynamic claims; relation false phrase via `false_phrase` (legacy `false` accepted) |
| `io_utils.py` | JSONL I/O (tolerates IDE-beautified multi-line objects), resume-by-id, call logs, summaries |
| `images.py` | Resolve / optionally download COCO train2014 JPGs |
| `stage{0,1,1b,2–5}_*.py` | Pipeline stages |
| `export_vti_flat.py` | One dimension → flat `{value,h_value}` JSONL; `--subtype` / `all` |
| `analyze_rejections.py` | Stage-1/4 rejection histograms |
| `render_review.py` | HTML galleries (`--stage 0\|1\|1b\|2\|3\|4\|final`); Stage 3 includes rejects + `errors` |
| `run_full.sh` | Wipe stage1–5 (+ call logs / flat exports), keep stage0/cooccurrence, run 1→5; `N_CANDIDATES` sets Stage-5 `--n-final` |
| `run_full300.sh` | Thin wrapper: `N_CANDIDATES=300 exec run_full.sh` |

**Tests:** `tests/test_demos_v2_validators.py`, `tests/test_demos_v2_allocator.py` (no GPU / network).

#### Stages (CLI, resume-by-id on pass+reject JSONLs)

| Stage | Script | Compute | Writes |
|-------|--------|---------|--------|
| 0 | `stage0_mine_candidates.py` | CPU — COCO `instances_train2014.json` | `stage0_candidates.jsonl`, `cooccurrence.json`, `stage0_summary.json` |
| 1 | `stage1_verify_anchors.py` | Vision MLLM (`STAGE1_PROVIDER`) | `stage1_verified.jsonl`, `stage1_rejected.jsonl`, `calls/stage1.jsonl` |
| 1b | `stage1b_allocate.py` | CPU seeded allocator | `stage1b_allocation.jsonl`, `stage1b_summary.json`; top-ups never alter existing ids |
| 2 | `stage2_write_truthful.py` | Vision MLLM (`STAGE2_PROVIDER`) | `stage2_captions.jsonl`, `stage2_rejected.jsonl`, `calls/stage2.jsonl` |
| 3 | `stage3_make_variants.py` | Text-only Haiku for existence polish/fallback (`STAGE3_PROVIDER`); other dims string edits | `stage3_variants.jsonl`, `stage3_rejected.jsonl`, `calls/stage3.jsonl` |
| 4 | `stage4_verify_faithfulness.py` | Vision MLLM (`STAGE4_PROVIDER`; model **must ≠** stage-2 and stage-3 models) | `stage4_verdicts.jsonl`, `stage4_rejected.jsonl`, `calls/stage4.jsonl` |
| 5 | `stage5_assemble.py` | none | `data/vti/demos_v2.jsonl`, `stage5_summary.json` (type/mode/distractor histograms) |

**Default providers** (`config.py`): stage1 `anthropic:claude-sonnet-5`, stage2 `anthropic:claude-opus-4-8`, stage3 `anthropic:claude-haiku-4-5`, stage4 `anthropic:claude-sonnet-5`. Keys: repo-root `.env`. Use `--provider mock` for dry runs (no API). Bare `anthropic` defaults to `claude-sonnet-5` in `mllm_client._DEFAULT_MODELS`.

##### Stage 0 — candidate mine (no MLLM)

From COCO train2014 instances (excludes ids already in author `demos.jsonl` by default):

- **`counting_options`:** categories with count in `[COUNT_MIN, COUNT_MAX]` (2–9), median bbox area ≥ `COUNT_MIN_AREA_FRAC`.
- **`relation_options`:** singleton non-crowd ordered pairs evaluated by `geometry.evaluate_all_relation_types` (horizontal / vertical / support / proximity), each with geometry fields (e.g. `a_side`, `gap_frac`).
- **`distractor_candidates`:** top-`DISTRACTOR_TOP_K` absent categories ranked by co-occurrence `P(c|present)` from `cooccurrence.json`.
- Ranked by weighted score (`W_REL_GAP`, `W_COUNT_AREA`, `W_N_CATS`, `W_N_GE3`, `W_REL_TYPE_DIV`); emit top `--n-candidates` (config default `N_CANDIDATES=300`).

Candidate fields: `id`, `image`, `present_categories`, `counting_options`, `relation_options`, `distractor_candidates`, `score` (plus meta). **Not** the v2.0 scalar `counting_anchor` / `relation_anchor` schema.

##### Stage 1 — visual option verify

Vision model verifies **each** counting / relation / distractor option and proposes up to `K_ATTR` attribute candidates. Emits `verified_*_options` lists (geometry from Stage 0 is merged back when the model strips it). Rejects feed `stage1_rejected.jsonl` with reason tags (`counting_check`, `relation_check`, …).

##### Stage 1b — seeded allocation

CPU allocator (`ALLOC_SEED`) picks one counting / relation / distractor / attribute assignment per verified image under fractional type/category caps in `config.py`. Relation options overlapping the chosen counting category are rejected. Writes `true_phrase` on the relation assignment for Stage 2 templates.

##### Stage 2 — truthful caption

Vision model writes the 4-sentence caption + verbatim spans (`existence_insertion_hint`, `attribute`, `counting`, `relation`) with `sentence_idx`. `spans.relation.text` must be the **short** phrase only (`left`, `on top of`, …), not the full S4. `_coerce_edit_spans` repairs common full-sentence relation span mistakes. `structural_check_truthful` enforces: 4 sentences, no digits, no banned hedges, count/relation/attribute tokens appear once, counting category (or natural plural) present, distractor absent, spans are substrings of the right sentences.

##### Stage 3 — hallucinated variants

1. Counting / attribute / relation: sentence-scoped replace; false count via `FALSE_COUNT` or `false_count_at_most`; relation false via `_false_relation_phrase`. Attribute validation is **sentence-scoped** (S2-only; handles shared prefixes e.g. `turned off` → `turned on`).
2. Existence: `deterministic_existence_insert` → Haiku grammar polish → optional LLM fallback. Existence validator allows multi-region S1 edits. Records `existence_source`.
3. `h_values.all`: existence base + attribute/counting/relation edits. `validate_record_variants` checks per-dim minimal pairs and distractor leak on attribute/counting/relation only (**not** on `existence` or `all`).
4. Anchors include `anchors.relation.false_phrase` (and `false`) for Stage 4 claims.

##### Stage 4 — faithfulness filter

Independent vision model answers **dynamic** statements from `stage4_statement_specs`: truthful S1–S4 (`true`); existence / attribute / counting false claims (`false`); relation uniqueness for A and B (`true`); false relation claim (`false`). Failures store dimension tags. Summary notes this is a **VLM filter, not ground truth**; manual HTML review remains authoritative.

##### Stage 5 — assemble

Sort stage-4 passes by stage-0 rank; take up to `--n-final` (config default 150; `run_full.sh` uses `$N_CANDIDATES`). Writes final schema below + distribution histograms in `stage5_summary.json`. Shortfall prints a top-up hint (`stage0_mine_candidates.py --emit-next-batch`).

#### Final schema (`data/vti/demos_v2.jsonl`)

One object per line (avoid IDE “Format Document” — `read_jsonl` can recover beautified files, but writers emit canonical JSONL):

| Field | Type | Notes |
|-------|------|-------|
| `id` | str | COCO train2014 id, zero-padded |
| `image` | str | Filename under `data/coco/train2014/` |
| `question` | str | `config.QUESTION` = `"Describe this image in detail."` |
| `value` | str | Truthful 4-sentence caption |
| `h_values` | object | Keys `existence`, `attribute`, `counting`, `relation`, `all` |
| `anchors` | object | Per-dimension metadata: attribute `type`/`object`/`true_value`/`false_value`; counting `mode`/`true_word`/`false_word`; relation `type`/`a`/`b`/`true`/`false`/`false_phrase`/`geometry`; existence `distractor` |
| `provenance` | object | `stage{1–4}_model`, `pipeline_version`, `date` |

**Flat export** (for existing VTI loaders):

```bash
python data_scripts/vti_demos_v2/export_vti_flat.py --dimension counting
# → data/vti/demos_v2_counting.jsonl with value / h_value
```

`--dimension all` exports the deterministic composition. `--subtype` filters relation/attribute/counting allocation subtypes (for example `--dimension relation --subtype vertical`).

#### v2.1 option-set and allocation rules

Stage 0 stores `counting_options`, `relation_options`, and scored `distractor_candidates`; relation candidates require singleton non-crowd category pairs and use `geometry.evaluate_all_relation_types` (horizontal, vertical, support, proximity). Stage 1 verifies every option and emits verified lists (geometry backfilled from Stage 0 when stripped). Stage 1b uses `ALLOC_SEED`, fractional type/category caps from `config.py`, and writes achieved distributions. Relation candidates that overlap the chosen counting category are rejected by the allocator. For a complete reallocation, wipe stage1b–5; normal top-up allocation preserves existing rows.

Stage-2 spans have schema `{"sentence_idx": int, "text": str}`. Replacements are sentence-scoped. Counting supports `at_least` and completeness-gated `at_most`; relation wording includes horizontal, vertical, support, and proximity templates. Stage 4 checks relation uniqueness and tags failure dimensions. `stage5_summary.json` reports type/mode/distractor distributions.

#### Validator notes

- Stage-2 counting category match: `category_mentioned` accepts natural plurals (`person`/`people`, `dining table`/`dining tables`, …).
- Stage-3 existence: deterministic insert + Haiku grammar polish; multi-region S1 allowed; full LLM fallback only.
- Stage-3 attribute: S2-only phrase check (not raw token opcodes) so shared-prefix attributes validate.
- Distractor leak: forbidden in attribute/counting/relation variants; **allowed** in existence and `all`.
- `io_utils.read_jsonl`: line-oriented first; falls back to streaming `JSONDecoder.raw_decode` if an editor pretty-printed the file.

#### Historical / live funnels (factual)

**v2.0 POC** (frozen `data/vti/v2_poc_2026-07-12/`, 2026-07-12) — `run_full300.sh`, hash `ad44185346c48958`:

| Stage | Pass | Reject | Notes |
|-------|------|--------|-------|
| 0 | 300 | — | mined (scalar anchors) |
| 1 | 240 | 60 | vision anchor fails |
| 2 | 232 | 8 | `structural_validation_failed` |
| 3 | 208 | 24 | `deterministic_diff_failed`; existence mostly deterministic+grammar |
| 4 | 131 | 77 | `faithfulness_mismatch` |
| 5 | **131** | — | all stage-4 passes kept |

**v2.1 full-1000** (live `data/vti/v2/` + `data/vti/demos_v2.jsonl`, 2026-07-13) — Stage 0 mined at 1000; Stages 1/4 mostly Sonnet 5 (early Stage 1 used Sonnet 4.6 before the mid-run switch); hash `9a44f4afde0324b5`:

| Stage | Pass | Reject | Notes |
|-------|------|--------|-------|
| 0 | 1000 | — | v2.1 option-set mine |
| 1 | 835 | 165 | `relation_check` / `counting_check` dominate |
| 1b | 835 | — | allocated |
| 2 | 827 | 8 | `structural_validation_failed` |
| 3 | 813 | 14 | `deterministic_diff_failed`; existence: 771 deterministic+grammar, 41 llm |
| 4 | 555 | 258 | `faithfulness_mismatch` |
| 5 | **555** | — | `--n-final 1000`; all stage-4 passes kept |

`stage5_summary.json` also records achieved relation-type / attribute-type / counting-mode histograms for this assemble.

#### How to run

```bash
# Unit tests (no API)
python -m pytest tests/test_demos_v2_validators.py tests/test_demos_v2_allocator.py -q

# Deterministic mine only
python data_scripts/vti_demos_v2/stage0_mine_candidates.py --n-candidates 1000

# Mock dry-run
python data_scripts/vti_demos_v2/stage1_verify_anchors.py --provider mock --limit 5
# … stages 1b–4 with --provider mock …
python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 5

# Fresh paid downstream (WIPES stage1–5 + call logs + demos_v2*.jsonl; keeps stage0)
N_CANDIDATES=1000 nohup bash data_scripts/vti_demos_v2/run_full.sh \
  > logs/demos_v2_1_full1000.log 2>&1 &

# Resume after interrupt (do NOT use run_full.sh — it wipes):
#   python data_scripts/vti_demos_v2/stage1_verify_anchors.py
#   python data_scripts/vti_demos_v2/stage1b_allocate.py
#   python data_scripts/vti_demos_v2/stage2_write_truthful.py
#   … stage3, stage4, stage5_assemble.py --n-final 1000

# Review (download HTML if SSH; omit --open)
python data_scripts/vti_demos_v2/render_review.py --stage final
```

#### demos_v2 textual directions (live extractor math + shuffled_prefix)

**Status (2026-07-13):** wired for the qualitative grid plan
(`implementation_plans/vti_demos_v2_qual_grid_plan.md`). Uses the **live**
author-demo textual PCA path (not per-layer pure PC1) so `+β` semantics match
prior runs.

| Piece | Location / fact |
|-------|-----------------|
| Module | `evaluation/interventions/vti/directions_v2.py` |
| Extract CLI | `evaluation/run_scripts/extract_demosv2_directions.py` |
| Qual driver | `evaluation/run_scripts/run_demosv2_qual_grid.sh` |
| Gallery | `helper_scripts/render_demosv2_qual_review.py` (one HTML per dim); **preferred:** `helper_scripts/render_demosv2_qual_split_review.py` (split by intervention / dim / nd) |
| Source demos | `data/vti/demos_v2.jsonl` (555 rows; all five `h_values` present on every row; hash `9a44f4afde0324b5`) |
| Master order | `data/vti/demos_v2_order_s42.json` — sorted ids shuffled once with seed **42** |
| Qual subsets | `data/vti/qual_subset_chair5_amber25.json` (CHAIR-5 + AMBER-25); smoke pin `qual_subset_chair1_smoke.json` |

**Token-position policy (unchanged from author-demo extractor):**
`forward_vl(image, question + " " + caption)`; take the **last token** of the
full sequence at every hidden-state row (embedding + all decoder layers), i.e.
`hidden_states[layer][0, -1, :]`.

**Diff polarity (live path):** `act(value) − act(h_value)` where `h_value` is
the dimension caption from `h_values[d]` (truthful caption minus hallucinated
caption). Field names: `value` = verified truthful caption; `h_values[d]` =
dimension-specific hallucinated caption (`d ∈ {existence, attribute, counting,
relation, all}`).

**PCA / steering (live path, extended to rank 2 for caching):** stack each
demo’s last-token stack into one flattened vector of length
`(num_layers+1)*hidden_dim`; fit global PCA with `rank=2`; sign handling is
`PCA.svd_flip` only (no mean-diff sign-align). **Steering direction** =
reshape(`PC1 + mean`) — identical to calling the legacy
`obtain_textual_vti(..., rank=1)`. PC2 (+ flat mean) is stored in
`components.npz` for later projection diagnostics; never steered.

**Selection policy `shuffled_prefix`:** for `num_demos=N`, take the first N
master-order ids that have a non-empty `h_values[d]`. Nested prefixes
`{50,100,200,500}` share the same order. Activations are cached once under
`experiment_artifacts/vti/{model_short}/textual_v2/_act_cache/` with suffixes
`vl_v2_value` / `vl_v2_{dimension}` so PCA reruns are free.

**Direction cache (identity-bearing slug):**
`experiment_artifacts/vti/{model_short}/textual_v2/{slug}/directions.npz` +
`metadata.json` (+ `components.npz`). Slug example:
`demosv2_9a44f4af_all_nd500_s42_r2_prefix`. Legacy author-demo caches under
`experiment_artifacts/vti/{model_short}/textual_directions_*.npz` are
**never** read for demos_v2 runs.

**Eval plumbing:** `run_eval.py` / `run_evaluation()` accept `--demos_path`,
`--vector_dimension`, `--num_demos`, `--rank`, `--max_pixels` (Qwen2 only).
Result dirs for demos_v2 cells: `{iv}__b{beta}__d{dim}__nd{N}`.
`intervention.config` logs demos path/hash, dimension, selection policy,
steer_component, diff_polarity, token_policy.

**Qual grid frozen facts:** `--run_date 2026-07-13`; CHAIR prompt
`"Please Describe this image in detail."`; `--chair_max_new_tokens 512`
(deliberate vs 6/22 cap 64); AMBER discriminative; β∈{0.5,0.2,0.9}; Qwen
`max_pixels=1003520` (Exp1 Qwen cells had **no** cap — tonight is a VRAM-sharing
delta). Local only (no W&B).

#### demos_850 disjoint-partition directions

**Status (2026-07-30):** 850-row pool and seed-42 partition are **pinned on disk**.
Extraction primitives for sample-size work produce activations + directions only —
no evaluation wiring, no cross-block comparison in `run_eval.py`.

| Piece | Location / fact |
|-------|-----------------|
| Module | `evaluation/interventions/vti/directions_partition.py` — `build_or_load_partition`, `select_block_demos`, `partition_slug`, `compute_or_load_partition_directions`, `extract_partition_grid`. `partition_slug` / `textual_v2_slug` accept defaulted `fit_locus` (`"global"` \| `"perlayer"`) — omitted → existing slug strings unchanged |
| Assemble CLI | `data_scripts/vti_demos_v2/assemble_demos_850.py` (`--write-partition`) — first 555 lines byte-identical to `demos_v2.jsonl`; appends new stage-4 passes; does **not** rewrite `demos_v2.jsonl` |
| Top-up drivers | `helper_scripts/run_demos850_topup_stages1to4.sh` (stages 1→4 resume-safe); `helper_scripts/run_demos850_continue_after_topup.sh` (wait → assemble+partition → GPU extract → verify) |
| Extract CLI | `evaluation/run_scripts/extract_demos850_partition_directions.py` |
| Verify CLI | `helper_scripts/verify_demos850_partition_extraction.py --model <hf id>` |
| Source demos | `data/vti/demos_850.jsonl` — 555 byte-identical prefix of `demos_v2.jsonl` + 295 new rows; **`content_hash_sha256_16 = ba05bd960cad0c18`** (full sha256 `ba05bd960cad0c18a49abbd609c563267f55a12245ec9481e1c8161c299627a5`) |
| Partition | `data/vti/demos_850_partition_s42.json` — `_meta.content_hash_sha256_16` must match demos file; sorted ids shuffled once with seed **42**, consumed into disjoint blocks **50 / 100 / 200 / 500** (`created` 2026-07-29) |
| Path helpers | `src.paths.vti_demos_850_path()`, `vti_demos_850_partition_path()` |
| Stage0 top-up | `stage0_mine_candidates.py --summary-out` (required so top-up does not overwrite `stage0_summary.json`); snapshot under `data/vti/v2/_summaries_snapshot_555_2026-07-28/` |

**Selection policy `disjoint_partition`:** every id appears in exactly one block;
the same partition is used for all five dimensions. Missing captions raise (no
skip list). Slug stem `demos850_…_partition` — cannot collide with
`demosv2_9a44f4af_*_prefix`. Example: `demos850_ba05bd96_all_nd500_s42_r2_partition`.

**Act-cache reuse:** shares `experiment_artifacts/vti/{model_short}/textual_v2/_act_cache/`
with demos_v2 when check 0.4 (cache fidelity) passes. Qwen2 **must** pass
`--max_pixels 1003520` (runner exits otherwise). On fidelity failure the runner
isolates under `_act_cache_demos850_2026-07-28/` and does not write into the
shared cache.

**Do not** pass `demos_850.jsonl` to `run_eval.py --demos_path` or to
`extract_demosv2_directions.py` — those paths use `load_or_build_master_order`
/`DEFAULT_ORDER_PATH` and correctly reject the 850-row file on the hash guard.
Consuming partition directions in eval is out of scope for this extraction.

**Direction cache:** `experiment_artifacts/vti/{model_short}/textual_v2/{slug}/`
with `directions.npz` + `metadata.json` + `components.npz`.

#### Mean-difference textual directions (demos_850, CPU, 0 forwards)

Raw mean over demos of `(value_stack − h_stack)` — no PCA, no component
selection, no sign flip. Reads shared `textual_v2/_act_cache/` only; raises on
cache miss; never constructs a model wrapper.

| Piece | Location / fact |
|-------|-----------------|
| Module | `evaluation/interventions/vti/directions_meandiff.py` |
| Extract CLI | `evaluation/run_scripts/extract_demos850_meandiff_directions.py` |
| Verify | `helper_scripts/verify_demos850_meandiff_extraction.py` |
| Slug | `demos850_{hash8}_{dimension}_nd{N}_s42_meandiff_partition` |
| Out | `experiment_artifacts/vti/{model_short}/textual_v2/{slug}/` — `directions.npz` + `metadata.json` only (no `components.npz`) |
| Manifest | `experiment_artifacts/vti/demos850_meandiff_extraction_manifest_2026-07-30.json` |
| `steer_reconstruction` | `raw_mean_difference` |
| `diff_polarity` | `value_minus_h_value` (same as PCA live path) |
| Plan | `implementation_plans/7-30-26/steering_vector_visual_reasoning_validation_plan_2026-07-30.md` |

Consumed at eval via `--directions_dir <slug dir>` (not `--demos_path`).

**Eval driver:** `evaluation/run_scripts/run_steering_visual_reasoning_validation.sh`
(one model per process; AMBER → CHAIR → POPE; baseline then layer sets
`all` / `5-14` / late window; nd order 50→500→200→100; betas 0.2→0.5→0.9).

**Overnight orchestrator:**
`evaluation/run_scripts/launch_steering_visual_reasoning_overnight.sh` —
extract → verify → CHAIR caption-length probe @ 256/512 → if any caption hits
256 on either model, set `CHAIR_CAP=512` (never halt; never skip CHAIR) → launch
both model drivers staggered 5 min on `CUDA_VISIBLE_DEVICES=0` → start
`babysit_steering_visual_reasoning_grids.py` (crash restart with
`--skip_if_exists`; OOM crashes wait for sibling model to finish before
relaunch).

**Offline analysis:** `evaluation/steering_visual_reasoning_validation/`
(`build_result_tables.py`, `make_plots.py`) under
`evaluation/results/{run_date}/_analysis_steering_visual_reasoning_validation/`.

#### Out of scope / remaining follow-ups

1. ~~Textual cache path/slug must include demos identity~~ — **done** for demos_v2 (`textual_v2/` namespace). Author-demo legacy path still omits demos hash in the filename.
2. ~~`run_eval.py` demos_path CLI~~ — **done**.
3. ~~Top up beyond 555 Stage-4 passes and assemble an 850-row pool~~ — **done**
   (2026-07-29/30): `demos_850.jsonl` + `demos_850_partition_s42.json` pinned;
   hash `ba05bd960cad0c18`.
4. Per-layer PCA textual extraction (one PCA per activation row, PC1+mean recon) —
   **implemented** for demos850 geometric comparison (2026-07-29); see
   **Per-layer PCA vs global PCA** below. Not wired into `run_eval.py`.
5. Wiring demos850 directions into `run_eval.py` / `VTITextualIntervention` —
   **done for raw mean-difference** (2026-07-30): see **Mean-difference
   textual directions** and `--directions_dir` / `--layer_set` below. PCA
   `*_r2_partition` directories are still loadable the same way (same
   `textual_v2` on-disk format) but the validation grid uses meandiff slugs.

### Shuffled-control direction (image derangement)

Construct-and-verify control for the deployed `all`/nd200 direction
(`demosv2_9a44f4af_all_nd200_s42_r2_prefix`). Same 200 demo ids, order, and
caption pairs; each demo’s **image path** is overridden via a derangement so no
demo keeps its own image. Caption contrast preserved; image–caption binding
destroyed. Not wired into `run_eval.py` / interventions — extract-only.

| Piece | Location / fact |
|-------|-----------------|
| Module | `evaluation/interventions/vti/shuffled_control.py` |
| Extract CLI | `evaluation/run_scripts/extract_shuffled_control_directions.py` |
| Plan | `implementation_plans/7-22-26/shuffled_control_direction_sanity_checks_plan_2026-07-22.md` |
| Derangement | `experiment_artifacts/vti/shuffled_control_image_derangement_nd200_s1234.json` (seed **1234**, model-independent; 0 fixed points) |
| Direction out | `experiment_artifacts/vti/{model_short}/shuffled_control/all_nd200/{directions.npz,metadata.json,components.npz}` |
| Act cache | `experiment_artifacts/vti/{model_short}/shuffled_control/_act_cache/` — **must not** reuse `textual_v2/_act_cache` (cache key is demo id + variant; image is not part of the key) |
| Sanity report | `experiment_artifacts/vti/{model_short}/shuffled_control/shuffled_control_sanity_report_{model_short}.md` |
| Models verified | `llava-1.5-7b-hf` (`llava-hf/llava-1.5-7b-hf`); `qwen2.5-vl-7b-instruct` (`Qwen/Qwen2.5-VL-7B-Instruct`, `--max_pixels 1003520`) |

**Construction:** load deployed `ids_used` → build/reuse derangement map →
`select_prefix_demos(..., "all", 200)` → override each flat demo’s `image` from
the source-image demo id → `ensure_variant_activation` into the shuffled cache
namespace → `obtain_textual_vti_v2_from_stacks` (same live PCA path as demos_v2).

**Gating checks in the sanity report:** derangement fixed points = 0; captions
unchanged 200/200; positional id mismatches vs deployed `ids_used` = 0;
direction shape `(num_layers, hidden_dim)`; forward passes executed = 400
(200 demos × `value` + `all`). Non-gating: per-layer L2 norms printed next to
deployed `direction_layer_norms`.

#### demos_850 shuffled-control (per partition block, dimension=`all`)

**Status (2026-07-29/30):** construct-and-verify controls for each demos_850
disjoint block N∈{50,100,200,500}. Derangement maps for all four N are tracked
under `experiment_artifacts/vti/`. Direction / act-cache trees live locally and
are gitignored (`experiment_artifacts/vti/**/shuffled_control_demos850/` and
`…/shuffled_control_demos850_perlayer/`). Per-layer vs global geometric
comparison of these controls is **done** (see next section); a demos_v2-style
single-N cosine/magnitude plot suite for every demos850 N is not a separate
driver.

| Piece | Location / fact |
|-------|-----------------|
| Module | `evaluation/interventions/vti/shuffled_control_partition.py` — `load_or_write_block_derangement`, `extract_shuffled_control_partition_direction`, sanity report writers |
| Extract CLI | `evaluation/run_scripts/extract_demos850_shuffled_control_directions.py` |
| Plan | `implementation_plans/7-29-26/demos_850_shuffled_control_per_partition_block_plan_2026-07-29.md` |
| Spec | `extractions/shuffle_control_vector_diff_sample_size_07_29_26_extraction.md` |
| Control-for | `demos850_ba05bd96_all_nd{N}_s42_r2_partition` |
| Derangement | `experiment_artifacts/vti/shuffled_control_demos850_image_derangement_nd{N}_s1234.json` (seed **1234**, per block; tracked) |
| Direction out | `experiment_artifacts/vti/{model_short}/shuffled_control_demos850/all_nd{N}/` (local / gitignored) |
| Act cache | `…/shuffled_control_demos850/_act_cache/` — must not reuse `textual_v2/_act_cache` or legacy `shuffled_control/_act_cache` |
| Sanity | `…/shuffled_control_sanity_report_{model_short}_all_nd{N}.md` + rollup |
| Models | LLaVA-1.5-7B; Qwen2.5-VL-7B (`--max_pixels 1003520` required) |

**Construction:** load deployed partition `ids_used` (= partition block) →
per-block derangement → image override → `ensure_variant_activation` into the
demos850 shuffled cache → live PCA. Leaves demos_v2 `shuffled_control/all_nd200`
untouched. Across four N: 850×2 = 1700 forwards/model (disjoint blocks, shared
cache).

### Geometric comparison (deployed vs shuffled-control)

CPU-only analysis of per-layer cosine and L2 magnitude between the deployed
`all`/nd200 direction and the shuffled-image control. No model load, no
steering, no W&B. Within-model only.

| Piece | Location / fact |
|-------|-----------------|
| Script | `diagnostic_experiments/perception_diag/control/geometric_comparison/compare_deployed_vs_shuffled_control.py` |
| Plan | `implementation_plans/7-22-26/geometric_comparison_deployed_vs_shuffled_control_plan_2026-07-22.md` |
| Out dir | `diagnostic_experiments/perception_diag/control/geometric_comparison/` |
| Loader | `load_textual_v2_directions(cache_dir)` → decoder directions `(num_layers, hidden_dim)` |
| PC1 source | `components.npz` key `pc0` (PC1), shape `(num_layers+1, hidden_dim)`; decoder rows `[1:]` aligned with `directions.npz` |
| CSV cols | `layer, cosine, deployed_norm, shuffled_norm, deployed_pc1_norm, shuffled_pc1_norm` |
| Plots | `cosine_deployed_vs_shuffled_control_by_layer_{model_short}.png`; `magnitude_deployed_vs_shuffled_control_by_layer_{model_short}.png` |
| Summary | `geometric_comparison_summary.md` (facts only) |

**Preconditions:** shape match vs plan expectations (LLaVA `(32, 4096)`, Qwen
`(28, 3584)`); PC1/direction fraction emitted per layer (no hard cutoff);
cosine-plot open markers at/above the 90th percentile of
`max(PC1/direction)` across decoder layers.

### Per-layer PCA vs global PCA (demos850 deployed vs shuffled-image control)

**Status (2026-07-29):** geometric comparison only. Fits one centered PCA per
`(num_layers+1)` activation row (reuse `pca.PCA` 3-D branch), reconstructs
`PC1 + mean`, drops embedding row. Compares deployed vs shuffled-image control
cosine under both the new per-layer scheme and the existing global flatten
scheme. Zero forward passes; CPU-only.

| Piece | Location / fact |
|-------|-----------------|
| Design | `designs/perlayer_pca_control_07_29_26.md` |
| Plan | `implementation_plans/7-29-26/per_layer_pca_deployed_vs_control_cosine_plan_2026-07-29.md` |
| Module | `evaluation/interventions/vti/perlayer_pca.py` |
| Extract CLI | `evaluation/run_scripts/extract_demos850_perlayer_pca_directions.py` |
| Verify CLI | `helper_scripts/verify_perlayer_pca_control_extraction.py --mode record\|verify` |
| Analysis | `diagnostic_experiments/perlayer_pca_control/scripts/compare_perlayer_vs_global_pca_deployed_vs_control.py` |
| Sign-aligned diagnostic | `…/scripts/plot_cosine_pc1_sign_aligned_between_arms.py` |
| Out dir | `diagnostic_experiments/perlayer_pca_control/` with subdirs `scripts/`, `verification/`, `plots/`, `tables/`, `summaries/` (plots/tables/summary + light verification reports are tracked; large sha256 manifests stay local) |
| Deployed out | `experiment_artifacts/vti/{model_short}/textual_v2_perlayer/demos850_ba05bd96_all_nd{N}_s42_r2_partition_perlayer/` |
| Control out | `experiment_artifacts/vti/{model_short}/shuffled_control_demos850_perlayer/all_nd{N}_perlayer/` (local / gitignored with other demos850 shuffled trees) |
| Act caches (read-only) | `textual_v2/_act_cache/` (deployed); `shuffled_control_demos850/_act_cache/` (control) |
| Models | `llava-1.5-7b-hf` `(32, 4096)`; `qwen2.5-vl-7b-instruct` `(28, 3584)` |
| N | 50 / 100 / 200 / 500 (demos850 disjoint partition, hash `ba05bd960cad0c18`) |

**Shadowing guards:** (1) `fit_locus` on `partition_slug` / `textual_v2_slug`
defaults to `"global"` so existing slug strings are unchanged; `"perlayer"`
appends `_perlayer`. (2) Separate parent dirs `textual_v2_perlayer/` and
`shuffled_control_demos850_perlayer/`.

**Metadata extras on per-layer cells:** `fit_locus`, `fit_locus_description`,
`explained_variance_ratio_per_layer`, `pc1_explained_variance_per_layer`,
scalar EVR fields `null`, `global_fit_source_dir`,
`global_fit_{metadata,directions}_sha256`, `forwards_executed: 0`,
`act_cache_read_only: true`, `arm`.

**Analysis artifacts:** under `diagnostic_experiments/perlayer_pca_control/`:
- `tables/cosine_and_norms_deployed_vs_control_by_layer_{model}.csv`,
  `tables/cosine_by_layer_band_and_sample_size_{model}.csv`
- `plots/` — 2×2 facet PNGs (cosine, direction L2, PC1 share, PC1 sign
  agreement); sign-aligned cosine diagnostic under `plots/pc1_sign_aligned/`
  with two variants per model (`with_disagreement_bars` /
  `without_disagreement_bars`); flip flags in `tables/pc1_sign_aligned/`
- `summaries/perlayer_vs_global_pca_control_summary.md`
- `verification/` — sha256 manifests, byte-identity / cache-untouched reports,
  extraction manifest
- Cosine helper imported from `compare_deployed_vs_shuffled_control.py`
  (demos_v2 curve not modified).

**Sign convention:** `svd_flip` as-is; Check 9 reports per-layer
`sign(cos(PC1, mean))` agreement between arms; no sign correction applied.

---

## `src/extraction.py` — Activations & analysis

### GPU

```python
cleanup_gpu()  # gc.collect + torch.cuda.empty_cache()
```

### Last-token hidden states

```python
get_last_token_activations(hidden_states) -> Dict[int, np.ndarray]
# {layer_idx: (hidden_dim,) float array at last prefill token}
```

Used by `extract_vl.py` / `extract_tt.py` when saving caches.

### `ActivationCache`

```python
cache = ActivationCache(cache_dir: str)
cache.save(sample_id, activations: Dict[int, np.ndarray], suffix: str = "vl")
cache.load(sample_id, suffix="vl") -> Dict[int, np.ndarray]
cache.load_or_none(sample_id, suffix="vl") -> Optional[Dict]
cache.exists(sample_id, suffix="vl") -> bool
```

**File naming:** `{cache_dir}/sample_{sample_id}_{suffix}.npz` with keys `layer_{L}`.

**Suffixes in use:** `vl` (image+text forward), `tt` (text-only forward on caption).

### Modality shift matrices

```python
load_activation_matrix(cache, sample_ids, layer, suffix) -> np.ndarray  # (N, hidden_dim)
load_modality_shift_matrix(cache, sample_ids, layer) -> np.ndarray      # VL - TT at layer
```

### Cross-modal attention (optional analysis)

```python
extract_cross_modal_attention(attentions, text_positions, img_start, img_end)
    -> Dict[(layer, head), np.ndarray]   # max text→image attention per image token
rank_heads_by_visual_attention(cross_modal_dict, top_n=3)
```

### SVD / subspace helpers

```python
effective_rank(matrix, tau=0.9) -> int
extract_subspace(matrix, k, ...) -> np.ndarray
pairwise_difference_matrix(...)
principal_angles(V1, V2)
subspace_overlap(V1, V2)
```

### I/O helpers

```python
save_json(obj, path) / load_json(path)
save_pickle(obj, path) / save_npz(arrays, path)
```

---

## Visual encoder hooks — **implemented (LLaVA-1.5 only)**

Visual VTI (ViT encoder steering) is implemented for **`llava-hf/llava-1.5-7b-hf`** only. Other wrappers raise `NotImplementedError` from `get_vision_dispatch()`. The textual arm remains decoder-only; the visual arm is orthogonal (separate registry keys, `alpha` coefficient, ViT hook sites).

### Module map

| Module | Role |
|--------|------|
| `src/vision_dispatch.py` | `VisionDispatch`, `get_vision_dispatch`, `prepare_pixel_tensor`, `denorm_pixel_tensor`, `verify_vision_layout` |
| `evaluation/interventions/vti/visual_perturb.py` | `perturb` — `patch_mask` / `gaussian_noise` on denormalized pixels |
| `evaluation/interventions/vti/visual_capture.py` | `capture_visual_hiddenstates`, `capture_corrupted_mean` — ViT `output_hidden_states` |
| `evaluation/interventions/vti/visual_directions.py` | `obtain_visual_vti`, `compute_or_load_visual_directions`; PCA recon `top_pc` (default) or `legacy_pc_plus_mean` |
| `evaluation/interventions/vti/visual_hooks.py` | `vti_visual_hook_ctx` — per-(layer, token) steering on ViT encoder |
| `evaluation/interventions/vti/visual_intervention.py` | `VTIVisualIntervention` |
| `diagnostic_experiments/vti_visual_smoke/` | Smoke driver + gate analysis (not `run_eval.py`) |

### LLaVA-1.5 ViT layout (verified)

- Tower: `wrapper.model.vision_tower` → `CLIPVisionModel`
- Per-layer hooks: `.vision_model.encoder.layers[i]` — submodules `self_attn`, `mlp`, `layer_norm*`
- **24** encoder layers, **1024** hidden dim, **577** tokens (CLS + 576 patches at 336²/14²)
- `verify_vision_layout(wrapper)` asserts the above on a dummy forward

### Direction extraction (visual)

- **Demos:** seed-42 sample of **70** COCO train2014 images from `data/vti/demos.jsonl` (same pool as textual arm; **no caption text** in the ViT forward).
- **Perturbation:** default `patch_mask`, `mask_ratio=0.99`, `mask_fill=zero`, `num_trials=50` trial average per demo.
- **Diff sign:** Δ = h̄(masked) − h(clean) per demo; PCA rank-1 per (layer, token).
- **Reconstruction:** `top_pc` (paper-faithful per-(layer,token) PCA + sign align to mean diff); `legacy_pc_plus_mean` ports reference layer-major `reshape` scramble for parity.
- **Cache:** `experiment_artifacts/vti/{model_short}/visual/{config_slug}/directions.npz` + `metadata.json`; perturbed PNG dumps under `perturbed_examples/`. Production slug on disk (LLaVA-1.5, default params): `patch_mask_r0.99_zero_top_pc_nd70_nt50_s42_*`.
- **CLS:** `include_cls=True` by default (steer all 577 positions).

### Steering (visual)

- Registry: `vti_visual_{additive|uniform_rotation}_{mlp|layer}` — **4 variants** (no `gated_rotation`).
- Coefficient: **`alpha`** (paper vision α); passed to `steer(..., alpha=alpha)` inside `vti_visual_hook_ctx`.
- Hook sites mirror decoder: `mlp` (ViT MLP sub-block output) | `layer` (full encoder block output).
- ViT runs once per image at prefill; hooks fire on the vision-tower forward inside `generate_vl`.

### Eval wiring

- `run_eval.py` / `run_evaluation(..., alpha=...)` — when `--alpha` is set, visual interventions write under `{iv}__a{alpha}` (textual `beta` suffix unchanged).
- Factory filters: textual registry drops `alpha`; visual registry drops `beta`.
- `ALL_INTERVENTIONS`: `no_intervention` + 6 textual + **4 visual** = **11** total.

### Smoke test — `diagnostic_experiments/vti_visual_smoke/`

```bash
bash diagnostic_experiments/vti_visual_smoke/run_scripts/run_vti_visual_smoke.sh
# Qualitative HTML (after a smoke run_date exists):
RUN_DATE=2026-07-02 bash helper_scripts/run_render_smoke_review.sh
```

| Stage | Content |
|-------|---------|
| `verify` | `verify_vision_layout` on loaded LLaVA |
| `0` | Textual gate positive control (no_intervention + 3 textual cells) on pinned AMBER-25 + CHAIR-5 |
| `1` | Visual shakedown: no_intervention + `additive_layer` + `uniform_rotation_mlp` × α∈{0.2,0.4,0.9} |
| `1b` | Remaining visual variants (`additive_mlp` + `uniform_rotation_layer` × same α grid) |
| `analyze` | Writes `evaluation/results/{run_date}/_diagnostics/vti_visual_smoke_summary.md` (+ gate plots under `_diagnostics/gate_plots/`) |

Pinned subsets: AMBER-25 `ordered` ids from `evaluation/results/2026-06-22/_samples/llava-1.5-7b-hf/amber/...`; CHAIR-5 from chair sample JSON with prompt `"Please Describe this image in detail."`, `max_new_tokens=64`. Per-cell AMBER outputs include `per_item_p_yes.json` for gate readout (`analyze_gate.summarize_cell`: c-AUC, acc@0.5, h+/h− flip decomposition vs baseline).

**Status (2026-07-02, LLaVA-1.5):** stages `verify` / `0` / `1` / `1b` / `analyze` completed on lambdab2. **16 unique cells** per benchmark (1 baseline + 3 textual + 12 visual) under `evaluation/results/2026-07-02/llava-1.5-7b-hf/{amber,chair}/`. Summary: `evaluation/results/2026-07-02/_diagnostics/vti_visual_smoke_summary.md`. Side-by-side HTML galleries: `…/_samples/llava-1.5-7b-hf/{amber,chair}/*_smoke_review.html` (via `render_smoke_review.py`). **Not run:** Stage 2 discrepancy toggles (`legacy_pc_plus_mean`, `mask_fill=mean`, etc.).

### Per-model vision encoder anatomy (other families — not yet wired)

Vision hidden dim **≠** LLM `hidden_dim` on every model below — a projector (or merger) maps ViT patch features into the LLM embedding space. Visual VTI directions live in ViT dim; only LLaVA is hooked today.

| Model family | Wrapper | Vision encoder module(s) | ViT layers | ViT hidden dim | Vision tokens (Policy A) | Projector / merger |
|--------------|---------|--------------------------|------------|----------------|--------------------------|-------------------|
| LLaVA-1.5 | `LLaVAWrapper` | `wrapper.model.vision_tower` → `CLIPVisionModel`; per-layer hook target: `.vision_model.encoder.layers[i]` | 24 | 1024 | **576** patches + CLS (**577** steered) | `wrapper.model.multi_modal_projector` |
| ShareGPT4V | `ShareGPT4VWrapper` | **`wrapper._vision_tower`** (separate `CLIPVisionModel`) | 24 | 1024 | **576** fixed | **`wrapper._mm_projector`** |
| Qwen-VL-Chat | `QwenVLWrapper` | `wrapper.model.transformer.visual` | (remote) | (remote) | **448×448** fixed | Integrated in visual stack |
| Qwen2-VL / 2.5 | `Qwen2VLWrapper` | `wrapper.model.visual` → `Qwen2_5_VisionTransformerPretrainedModel` | 32 | 1280 | **Dynamic** | Merger inside `visual` |
| InternVL2 / 2.5 | `InternVL2Wrapper` | `model.extract_feature(pixel_values)` bundles ViT + adapter | (remote) | (remote) | **256** per tile | Inside `extract_feature` |

**LLaVA feature layer:** inference uses hidden states from `mm_vision_select_layer` (default **-2**) and drops CLS before projection. Visual VTI hooks the full encoder stack (all layers, CLS included when `include_cls=True`).

### Reference implementation (external — `VTI/`)

The vendored VTI repo (`VTI/`, out of pipeline scope) informed the port:

- **Direction extraction:** `VTI/vti_utils/icv_utils.py` — `get_visual_hiddenstates`, `obtain_visual_vti`.
- **Steering:** authors wrap ViT MLP with norm-preserving blend; we reuse `steer()` geometry variants.
- **Layout mismatch:** authors' `model.model.vision_tower.vision_tower.vision_model` ≠ our `transformers==4.50.1` path (`model.vision_tower` is already `CLIPVisionModel`).

### Tests

- `tests/test_visual_vti_reference_parity.py` — synthetic `legacy_pc_plus_mean` math parity (no GPU).
- `tests/test_VTI_text_steer.py` — textual `steer()` geometry (unchanged).

---

## `src/mediation.py` — Causal mediation (FCCT-style, **decoder-only**)

### Family dispatch

Maps wrapper architecture to **LLM decoder** hook targets:

```python
dispatch = get_dispatch(wrapper)       # FamilyDispatch
verify_layout(wrapper, dispatch)       # sanity check module paths
```

Families: `llava`, `llama_raw` (ShareGPT4V / some raw LLaMA stacks), `qwen_vl_chat`, `qwen2_vl` (Qwen2-VL / Qwen2.5-VL).

`FamilyDispatch` methods: `get_lm_head`, `get_layer`, `get_attn`, `get_mlp` — all `(wrapper, layer_idx)`.

### Scoring targets (token probability)

```python
ScoringTarget.yes_no(wrapper) -> (yes_target, no_target)
ScoringTarget.from_variants(wrapper, variants: List[str]) -> ScoringTarget
target_token_probs(wrapper, image, text, target) -> (prob_sum, seq_len)
compute_yes_prob(wrapper, image, text, yes_ids) -> (prob, seq_len)
forward_with_logits(wrapper, image, text) -> (last_token_logits, seq_len)
```

POPE mediation uses yes-token probability on the last prefill position.

### Capture / patch hooks

**Components:** `COMPONENTS = ("hidden_state", "mlp", "attn")`

```python
capture_clean_activations(wrapper, dispatch, image, text, layers=None)
    -> (store: Dict[(layer, component), Tensor], seq_len_placeholder)

# store values: last-token activation at hooked submodule, shape (hidden_dim,)

patch_hook_ctx(wrapper, dispatch, layer_idx, component, cached_act)
    # context manager: replaces last-token activation with cached_act during forward
```

Hooks operate on the **last sequence position** of submodule outputs (`t[..., -1, :]`).

---

## `src/prompt_spans.py` — Prefixed-prompt tokenization spans

Reusable utility for leading-clause / filler experiments.

```python
assert_tokenization_invariant(tokenizer_or_wrapper, prefix=, question=, join="")
    -> PromptSpanLayout
# Hard-fails unless tok(question) is an exact suffix of tok(prefix+join+question).
# (Additive tok(prefix)+tok(question) can fail under BPE at a trailing-space boundary.)
assert_neutral_question_match(tokenizer_or_wrapper, neutral_question=, prefixed_layout=)
map_text_span_to_sequence(input_ids, text_token_ids=, wrapper=)
    -> (start, end)  # decoder indices; LLaVA: no double-expand when processor already
                     # expanded image tokens into input_ids
resolve_clause_and_question_spans(wrapper, input_ids, layout)
    -> (clause_span, question_span)
    # question match in sequence; clause = clause_len tokens immediately before it
    # (leftmost clause token may differ from bare tok(prefix) after template newline)
layer_windows(num_layers, width=10, stride=5) -> List[(start, end_inclusive)]
# LLaVA-32: (0,9),(5,14),(10,19),(15,24),(20,29),(22,31)
# Qwen-28:  (0,9),(5,14),(10,19),(15,24),(18,27)
```

---

## `src/attention_knockout.py` — Attention knockout (Geva / Neo-style)

Blocks (query, key) edges by adding `dtype.min` to a 4D additive attention mask at selected decoder layers (SDPA; flash-attn not installed).

```python
build_knockout_additive_mask(batch_size, seq_len, query_positions, key_positions, device, dtype)
    -> Tensor[B,1,Q,K]
query_positions_for_scope(scope, clause_end=, seq_len=)
    # "block_last_token_reading" | "block_all_downstream_reading"
attention_knockout_ctx(wrapper, layer_indices=, query_positions=, key_positions=)
forward_with_knockout(wrapper, image, text, layer_indices=, query_positions=, key_positions=)
    -> (last_token_logits, seq_len)
verify_knockout_zero_attention(...)  # eager + output_attentions; asserts blocked mass ≈ 0
```

Unit tests: `tests/test_attention_knockout.py` (mask math, windows, tokenization invariant).

---

## `src/paths.py` — Path helpers

Always use these instead of hand-rolled paths:

```python
project_root() -> Path
diagnostic_experiment_dir(experiment: str) -> Path
diagnostic_results_dir(experiment: str, model_short: str) -> Path
diagnostic_plots_dir(experiment: str, model_short: str) -> Path
experiment_artifacts_dir(experiment: str, model_short: str) -> Path
evaluation_results_dir(model_short, benchmark, intervention) -> Path
data_root() -> Path
coco_root() -> Path
coco_val2014_dir() -> Path
coco_train2014_dir() -> Path
coco_annotations_dir() -> Path   # data/coco/annotations (CHAIR gt lookup)
vti_data_dir() -> Path
vti_demos_path() -> Path        # data/vti/demos.jsonl (author demos)
vti_demos_v2_dir() -> Path      # data/vti/v2/ (demos_v2 stage artifacts)
vti_demos_v2_path() -> Path     # data/vti/demos_v2.jsonl
vti_demos_850_path() -> Path    # data/vti/demos_850.jsonl (850-row pool)
vti_demos_850_partition_path() -> Path  # data/vti/demos_850_partition_s42.json
amber_data_dir() / pope_data_dir() / benchmark_data_dir(bench) -> Path
augmented_jsonl_path(bench, stem) -> Path   # data/{amber|pope}/augmented_{stem}.jsonl
perception_dump_dir(bench, model_short, run_tag) -> Path
    # data/{amber|pope}/dumps/{model_short}/{run_tag}/
infer_benchmark_from_run_tag(run_tag) -> str  # amber*|pope* → amber|pope
```

---

## Data scripts

| Script | Purpose |
|--------|---------|
| `download_all_benchmarks.sh` | All benchmark downloads in order |
| `download_chair.py` | COCO val2014 + CHAIR; `--with-train2014` for VTI demos |
| `download_{benchmark}.py` | Per-benchmark manifest + images |
| `generate_captions.py` | API/local captions → `data/captions/` |
| `vti_demos_v2/` | Controlled demos_v2 generation pipeline (see demos_v2 section above) |
| `vti_demos_v2/assemble_demos_850.py` | Build `demos_850.jsonl` (+ optional `--write-partition`); never rewrites `demos_v2.jsonl` |
| `extract_vl.py` | VL activations → `activations/sample_*_vl.npz` |
| `extract_tt.py` | TT activations on captions → `sample_*_tt.npz` |
| `prepare_data.py` | Orchestrates captions / activations / responses phases |

```bash
python data_scripts/prepare_data.py \
    --benchmarks pope \
    --models llava-hf/llava-1.5-7b-hf \
    --phases captions activations responses \
    --caption_provider anthropic
```

---

## Helper scripts (`helper_scripts/`)

### demos_850 top-up / verify

| Script | Purpose |
|--------|---------|
| `run_demos850_topup_stages1to4.sh` | Detached top-up stages 1→4 (resume-safe); does not run `run_full.sh` or touch `demos_v2.jsonl` |
| `run_demos850_continue_after_topup.sh` | After stages 1–4: assemble `demos_850.jsonl` + partition, GPU extract, verify |
| `verify_demos850_partition_extraction.py` | Pass/fail checks on demos850 partition direction cells |
| `verify_perlayer_pca_control_extraction.py` | `--mode record\|verify` for per-layer PCA control extraction |

### `review_responses.py` — inspect responses next to the input image

Pure-stdlib CLI (no torch/PIL) for qualitative review. Resolves a benchmark
sample id (e.g. `pope_random_00166`, `chair_000000357659`) to its image /
question / ground truth via `data/{benchmark}/combined.json`, and pulls the
model response(s) for that id out of zero or more results JSON files.

Shared logic lives in `helper_scripts/review_lib.py` (also used by
`sample_responses.py`).

```bash
python helper_scripts/review_responses.py SAMPLE_ID [results.json ...] \
    [--ids ID ...] [--ids-file PATH] \
    [--run-date YYYY-MM-DD] [--output-dir evaluation/results] \
    [--benchmark KEY] [--open] [--html [--html-out PATH] [--no-open]] [--max-chars N]
```

- Parses sweep JSON (`per_sample`, `changed_examples_by_beta`) and standard
  `run_eval` `responses.json`.
- `--html` writes **one self-contained gallery page** (all samples in a single
  file; images base64-embedded; table of contents with anchor links) under
  **`evaluation/results/{run_date}/_review/gallery.html`** by default.
  Pass multiple ids via positional + `--ids` / `--ids-file`. `run_date` is taken
  from `--run-date` or inferred from the first results path. Override with
  `--html-out`.
- `--open` opens the raw image in the OS viewer.
- Benchmark inferred from id prefix (`chair`, `amber`, `pope`, …).

### `sample_responses.py` — batch qualitative samples for a run_date

Extracts response samples from a dated results tree and writes **one bundle per
(model, benchmark)** under
**`evaluation/results/{run_date}/_samples/{model_short}/{benchmark}/`**:

- `{model_short}_{benchmark}_response_samples.md` / `.json` — markdown + machine bundle
- **`{model_short}_{benchmark}_review.html`** — self-contained gallery for that model × benchmark

A top-level `manifest.json` lists all bundle paths.

**CHAIR:** `n_samples` random ids from the pinned CHAIR-500 subset (default 5).

**AMBER:** `n_amber_per_stratum` ids per (question-type × ground-truth) cell,
ordered existence → attribute → relation, and within each type **no** then **yes**
(default 5 per cell). Drawn from the pinned AMBER-disc-450 eval subset so
responses exist. AMBER existence questions are hallucination probes — gold is
always `no` (no existence/yes stratum in the benchmark).

```bash
python helper_scripts/sample_responses.py --run_date 2026-06-22
RUN_DATE=2026-06-22 bash helper_scripts/run_sample_responses.sh
```

CHAIR+AMBER default β slices are baked in (Exp1 LLaVA baseline/0.4/0.9, Qwen
baseline/0.4; Exp2 both baseline/0.3/0.6). Exp1 includes all three grid
interventions by default (`--exp1-interventions` to narrow). `--no-html` skips
galleries; `--json-only` skips markdown.

### `build_amber_plot_qualitative_html.py` — per-plot AMBER response galleries

Builds one HTML gallery per selected overnight AMBER summary plot under
`evaluation/results/2026-07-30/_analysis_steering_visual_reasoning_validation/{llava,qwen}_amber_results/qualitative/`.
Each page: N=50 independently seeded AMBER-disc examples; image once; question +
gold; every config that plot shows (baseline + full nd×β or window×β grid) with
parsed yes/no (`metrics._normalize_yes_no`), correct/incorrect, and full text.
Gold-yes/no stratification only for the gold-label plot.

```bash
python helper_scripts/build_amber_plot_qualitative_html.py
```

### `render_smoke_review.py` — visual-smoke multi-cell galleries

Builds one self-contained HTML page per (model, benchmark) that shows **every
smoke cell** side-by-side for the pinned AMBER-25 / CHAIR-5 ids (same draw as
`vti_visual_smoke/run_smoke.py`). Reuses `review_lib` + sample-bundle helpers.

```bash
python helper_scripts/render_smoke_review.py --run_date 2026-07-02
RUN_DATE=2026-07-02 bash helper_scripts/run_render_smoke_review.sh
```

**Output:**
`evaluation/results/{run_date}/_samples/{model_short}/{benchmark}/{model_short}_{benchmark}_smoke_review.html`

Default cell order (16 dirs): `no_intervention` → 3 Stage-0 textual
(`uniform_rotation_layer` β∈{0.2,0.4}, `additive_layer` β=0.9) → 12 visual
(`additive_{layer,mlp}` + `uniform_rotation_{mlp,layer}` × α∈{0.2,0.4,0.9}).
Sample-id source defaults to the 2026-06-22 qualitative bundle
(`--sample-run-date` to override).

### `render_demosv2_qual_split_review.py` — demos_v2 split galleries

Preferred qualitative viewer for the demos_v2 grid. One HTML per
`(model, benchmark, intervention, dimension, num_demos)`:

```
evaluation/results/{run_date}/_samples/{model_short}/{benchmark}/
  {additive_layer|additive_mlp|uniform_rotation_layer|uniform_rotation_mlp}/
    {all|existence|attribute|counting|relation}/{50|100|200|500}/
      {model}_{benchmark}_{intervention}_{dimension}_nd{N}.html
```

Each page: baseline + that intervention at β∈{0.2,0.5,0.9} (**ascending**). CHAIR pages annotate
every response with per-caption `CHAIR_s` / `CHAIR_i` (yellow note beside the
response). Includes `uniform_rotation_{layer,mlp}` (the older per-dimension
mega-HTML had a regex bug that dropped those cells). Filenames encode the same
axes as the directory path so downloads do not collide on `review.html`.

```bash
python helper_scripts/render_demosv2_qual_split_review.py --run_date 2026-07-13
```

Sibling `render_demosv2_qual_review.py` still writes one large HTML per dimension
(kept for quick overview; split renderer is the navigation path).

### `render_vti_demos_review.py` — VTI demos caption gallery

HTML gallery of COCO train2014 images with paired `value` / `h_value` captions
and `co_objects` / `uncertain_objects` metadata from `data/vti/demos.jsonl`.

```bash
python helper_scripts/render_vti_demos_review.py
python helper_scripts/render_vti_demos_review.py --open
python helper_scripts/render_vti_demos_review.py --num-demos 20 --seed 42
python helper_scripts/render_vti_demos_review.py --ids 000000103108 000000504235
```

**Default output:** `data/vti/_review/vti_demos_review.html` (base64-embedded
images; not under `evaluation/results/`).

---

## Diagnostic experiments

### Modality shift — `diagnostic_experiments/modality_shift/`

**Question:** how does `||x_vl - x_tt||` vary by layer and label group?

```bash
MODEL=llava-hf/llava-1.5-7b-hf BENCHMARK=pope \
    bash diagnostic_experiments/modality_shift/run_scripts/run_modality_shift.sh
```

| Script | Args | Output |
|--------|------|--------|
| `compute_modality_shift.py` | `--model`, `--benchmark`, `--limit` | `diagnostic_experiments/modality_shift/{model}/results/` |
| `plot_modality_shift.py` | `--model` | `.../results/plots/` |

Requires pre-extracted `vl` and `tt` caches for the model × benchmark.

Artifacts also written under `experiment_artifacts/modality_shift/{model}/` when scripts use `experiment_artifacts_dir()`.

### Causal mediation — `diagnostic_experiments/causal_mediation/`

**Question:** FCCT-style recovery — does patching clean activations restore yes/no logits on POPE?

```bash
MODEL=llava-hf/llava-1.5-7b-hf \
    bash diagnostic_experiments/causal_mediation/run_scripts/run_causal_mediation.sh
```

| Script | Key args | Output |
|--------|----------|--------|
| `run_mediation.py` | `--model`, `--benchmark` (default `pope`), `--limit`, `--layers 0 8`, `--prompt` | `diagnostic_experiments/causal_mediation/{model}/results/` |
| `plot_recovery_rates.py` | `--model` | `.../results/plots/` |

Uses `capture_clean_activations` + `patch_hook_ctx` + `ScoringTarget.yes_no`.

### VTI lambda_sim — `diagnostic_experiments/vti_lambda_sim/`

**Question:** under `gated_rotation`, which decode tokens get amplified `lambda_sim`?

```bash
MODEL=llava-hf/llava-1.5-7b-hf HOOK_SITE=mlp \
    bash diagnostic_experiments/vti_lambda_sim/run_scripts/run_vti_lambda_sim.sh
```

| Script | Key args | Output |
|--------|----------|--------|
| `compute_lambda_sim.py` | `--model`, `--limit` (default 200), `--hook_site`, `--pope_split` | `diagnostic_experiments/vti_lambda_sim/{model}/results/lambda_sim_{hook_site}_{split}.json` |
| `plot_lambda_sim.py` | `--model`, `--hook_site`, `--pope_split` | `.../results/plots/` |

Set `log_lambda_sim=True` on `VTITextualIntervention` (wired in `compute_lambda_sim.py`).

---

## Evaluation — `evaluation/`

### Entry point

```bash
python evaluation/run_eval.py \
    --model HF_MODEL_ID \
    --benchmarks pope amber \
    --interventions no_intervention \
    --output_dir evaluation/results \
    --max_new_tokens 256 \
    --limit N \
    --skip_if_exists \
    --pope_split random \
    --run_date 2026-06-18 \
    --amber_task discriminative \
    --judge mock \
    --chair_max_new_tokens 256
```

**MMHal/CHAIR-specific flags (added 2026-06-22):**
- `--judge` selects the MMHal-Bench LLM judge at runtime: `mock` (default,
  offline — no network/key, returns a fixed parseable rating for dry runs),
  `openai[:model]`, `anthropic[:model]`, `gemini[:model]`. The judge is resolved
  and invoked **only** when `mmhal_bench` is in `--benchmarks`; CHAIR/POPE-only
  runs never need a key. Real providers read their key from env
  (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`|`GEMINI_API_KEY`)
  and raise a clear error if missing. The resolved judge identity is stamped into
  every MMHal `metric_summary.json` as `judge_name` (absolute MMHal scores are
  only comparable across runs judged by the same model). **Public benchmarks
  only** — internal data must not be routed to an external judge.
- `--chair_max_new_tokens` (default **256**, standing default as of 2026-07-30;
  was 64) freezes the CHAIR caption-generation length **independently of
  `--max_new_tokens`** (which other benchmarks, incl. MMHal at 256, still use).
  CHAIR confounds with caption length in both directions, so this value must stay
  constant across baseline and interventions and be recorded. Historical cells
  generated at 64 remain valid records of what was run.
- `--directions_dir PATH` — load a precomputed `textual_v2` direction directory
  (`directions.npz` + `metadata.json`). Takes precedence over demos_v2 / legacy
  extraction inside `VTITextualIntervention`.
- `--layer_set STR` — `all` or inclusive `A-B` (absolute decoder indices). Parsed
  to `layer_indices` + `layer_set_label` (`all` or `A_B`). Forwarded to
  `vti_hook_ctx(layer_indices=...)`.

When `--directions_dir` is set with `--beta`, result directories use
`{iv}__b{beta}__d{dimension}__nd{n_pairs}__meandiff__layers_{layer_set_label}`
so layer sets do not collide. Existing demos_v2 path composition is unchanged
when `directions_dir` is omitted.

**Pinned-subset / prompt flags (added 2026-06-22):**
- `--subset_ids_file PATH` pins the exact sample ids scored for each benchmark
  (overrides `--limit` in the loader, and filters BEFORE image load so a small
  subset of a large benchmark does not open every image). The file is JSON,
  either `{benchmark: [ids]}` (keys matching benchmark names; other keys, e.g.
  `_meta`, are ignored) or a flat `[ids]` list (applied to every benchmark in the
  run). Used for the CHAIR/AMBER diagnostics; pins live at
  `data/chair/pinned_chair_500.json` and `data/amber/pinned_amber_disc_450.json`.
- `--chair_prompt STR` replaces the stored CHAIR caption prompt verbatim for
  every CHAIR sample (e.g. the exact VTI prompt `"Please Describe this image in
  detail."` — note the capital D, vs the lowercase prompt stored in
  `combined.json`). Pins the prompt across all cells.

`--run_date` (default: today, `YYYY-MM-DD`) selects the dated results subtree
(see layout below). Same-day reruns resume; a new day starts a fresh tree.

**Launch scripts:**

```bash
# Fixed-beta (0.9) reproduction: 4 models × {no_intervention, additive_mlp,
# additive_layer, uniform_rotation_mlp} × 3 POPE splits, pinned 200/split.
bash evaluation/run_scripts/run_vti_pope_repro.sh

# Beta-grid reproduction (Deliverable A): beta is the OUTERMOST loop (default
# order leads with 0.4); baselines run once up front, then a full per-model table
# prints after each beta. Results at {date}/{model}/pope_{split}/{iv}__b{beta}/.
# Env: LIMIT, BETAS, MODELS, IVS.
LIMIT=200 bash evaluation/run_scripts/run_vti_pope_beta_grid.sh

# Combined A (beta-grid repro) + B (rotation-strength) drivers, one RUN_DATE:
bash evaluation/run_scripts/run_beta_grid_local.sh    # N=200/split (single A6000)
bash evaluation/run_scripts/run_beta_grid_runai.sh    # N=3000/split (RunAI, full GPU)
```

Env overrides: `LIMIT` (samples/split), `BETAS`, `MODELS`, `IVS`, `OUTPUT_DIR`, `RUN_DATE`, `CUDA_VISIBLE_DEVICES`. The drivers default `VARIANTS=uniform_rotation` for B (gated is numerically identical).

**POPE multi-split workflow:** each split is written to its **own** result tree
(`pope_{split}`), so the three splits do **not** collide or pool. Run each split
separately; resume is by sample `id` within that split's tree.

```bash
for split in random popular adversarial; do
  python evaluation/run_eval.py \
    --model llava-hf/llava-1.5-7b-hf \
    --benchmarks pope --interventions no_intervention \
    --pope_split $split --limit 200 --run_date 2026-06-18
done
```

Each split's `metric_summary.json` covers only that split's 200 records.
`accuracy_overall` / `f1_overall` are that split's metrics. (Earlier code wrote
all splits to a single `pope/{iv}/` dir, which under `--skip_if_exists` silently
ran only `random` and skipped `popular`/`adversarial`; the `pope_{split}` layout
fixes this.)

### Runner

Runner lives in `evaluation/runners/eval_runner.py`; re-exported from `evaluation/runners/__init__.py`.

```python
from evaluation.runners import run_evaluation, print_comparison_table

run_evaluation(
    model_id: str,
    interventions: list[str],
    benchmarks: list[str],
    output_dir: str | Path,
    max_new_tokens: int = 256,
    limit: Optional[int] = None,
    skip_if_exists: bool = False,
    pope_split: str = "random",
    amber_task: Optional[str] = None,
    run_date: Optional[str] = None,   # defaults to today's date (YYYY-MM-DD)
    beta: Optional[float] = None,     # textual coeff for VTI ivs; see below
    alpha: Optional[float] = None,    # visual coeff for VTI ivs; see below
    judge: str = "mock",              # MMHal judge spec; resolved iff mmhal in benchmarks
    chair_max_new_tokens: int = 256,   # frozen CHAIR caption length (see CLI note)
    subset_ids_file: Optional[str] = None,  # pinned per-benchmark id subset (see CLI note)
    chair_prompt: Optional[str] = None,     # verbatim CHAIR prompt override (see CLI note)
    demos_path: Optional[str] = None,       # textual VTI demos JSONL (demos_v2 or author)
    vector_dimension: Optional[str] = None, # demos_v2 h_values key (existence|…|all)
    num_demos: Optional[int] = None,        # override textual VTI demo count
    rank: Optional[int] = None,             # textual PCA rank (demos_v2 caches 2)
    max_pixels: Optional[int] = None,       # Qwen2/2.5-VL only; forwarded to create_wrapper
) -> dict   # returned dict includes "run_date"
```

**Beta grid.** `--beta` / `run_evaluation(beta=...)` overrides the VTI textual
coefficient. When set, it is passed to every intervention (no_intervention
ignores it) and the result dir for interventions whose `config` carries `beta`
becomes `{iv}__b{beta}` — so grid points don't collide and `no_intervention`
(bare `{iv}`, no beta) is computed once across a sweep. When demos_v2 vector
config is also set (`--vector_dimension` / `--demos_path` pointing at
`demos_v2.jsonl`), the dir gains `__d{dim}__nd{N}`. When `--beta` is unset,
behavior and paths are unchanged (intervention default `beta=0.9`).

**Alpha grid (visual arm).** `--alpha` / `run_evaluation(alpha=...)` overrides the
VTI visual coefficient. When set, interventions whose `config` carries `alpha`
write under `{iv}__a{alpha}`. Orthogonal to `--beta`; dual-arm runs are not yet
composed in a single registry key.

`print_comparison_table(model_short, results_root, run_date=None)` prints a
per-intervention summary. When `run_date` is omitted it auto-selects the most
recent dated subtree containing the model (falling back to the legacy
date-less layout). **POPE renders one column per split** (`POPE/rnd`,
`POPE/pop`, `POPE/adv`), each `acc/f1` as percentages (e.g. `84.7/85.2`).
HallusionBench shows accuracy. **AMBER** (discriminative) renders
`accuracy/neg_item_accuracy/yes_ratio` as integer percents (column
`AMBER a/n/yr`) — neg-acc up at flat yes-ratio = grounding (falls back to plain
accuracy for generative AMBER). **CHAIR** renders `chair_s/chair_i` as
percentages, **lower is better** (column `CHAIR s/i v`). **MMHal** renders
`avg_score` (mean judge rating 0-6), **higher is better** (column `MMHal ^`). A
legend line below the table states both directions.

### Interventions registry — `evaluation/interventions/__init__.py`

```python
ALL_INTERVENTIONS: list[str]   # no_intervention + 6 textual + 4 visual variants
get_intervention(name: str, model_id: str, **kwargs) -> InterventionBase
```

`InterventionBase.generate(wrapper, image, question, max_new_tokens, caption=None) -> str`

| Name | Class | Behavior |
|------|-------|----------|
| `no_intervention` | `NoIntervention` | Direct `wrapper.generate_vl(image, question)` |
| `vti_textual_{variant}_{hook_site}` | `VTITextualIntervention` | Decoder VTI steering (see below) |
| `vti_visual_{variant}_{hook_site}` | `VTIVisualIntervention` | ViT encoder VTI steering (see below) |

**VTI textual variants** (`variant` × `hook_site`):

- `variant`: `additive` | `uniform_rotation` | `gated_rotation`
- `hook_site`: `mlp` (reference site — steer MLP output before residual add) | `layer` (full decoder layer output after residuals)

Registry names: `vti_textual_additive_mlp`, `vti_textual_additive_layer`, … (6 total).

**VTI visual variants** (`variant` × `hook_site`):

- `variant`: `additive` | `uniform_rotation` (no `gated_rotation`)
- `hook_site`: `mlp` | `layer` on the **ViT encoder** (not the LLM decoder)

Registry names: `vti_visual_additive_mlp`, `vti_visual_additive_layer`, `vti_visual_uniform_rotation_mlp`, `vti_visual_uniform_rotation_layer` (4 total). **LLaVA-1.5 only.**

**Prompting note:** `generate_vl(image, question)` passes `question` through each wrapper's chat/template — it does **not** substitute `_caption_prompt`. Only `generate_caption()` uses the fixed caption prompt (`"Describe this image in detail."`). POPE eval therefore uses the benchmark yes/no question verbatim.

#### `VTITextualIntervention` — `evaluation/interventions/vti/`

```python
VTITextualIntervention(
    model_id: str,
    variant: str = "uniform_rotation",
    hook_site: str = "mlp",
    beta: float = 0.9,             # textual steering coefficient (paper notation)
    num_demos: int = 70,
    rank: int = 1,
    seed: int = 42,
    eps_coeff: float = 0.1,
    demos_path: Optional[Path] = None,
    direction_cache: Optional[Path] = None,
    log_lambda_sim: bool = False,
    vector_dimension: Optional[str] = None,  # demos_v2: existence|attribute|counting|relation|all
    directions_dir: Optional[Path] = None,   # textual_v2 dir; takes precedence
    layer_indices: Optional[Sequence[int]] = None,
    layer_set_label: Optional[str] = None,
)
```

**Coefficient naming.** The textual strength knob is `beta` (the VTI paper's
notation: α = vision coefficient, β = text coefficient). It is passed into the
geometry-agnostic `steer(..., alpha=beta)` / `vti_hook_ctx(alpha=beta)` — i.e.
`steer()`/`hooks.py` keep a single generic `alpha` coefficient that the textual
arm fills with β (the future vision arm will fill it with α). Renamed from
`alpha_text` on 2026-06-19; pre-rename result/config JSONs use `alpha_text`.

Lifecycle: compute/load directions on first `generate()`; register hooks via `get_dispatch`; run generation inside `vti_hook_ctx` (hooks always removed on exit). When `directions_dir` is set, loads via `load_textual_v2_directions` (metadata loaded eagerly for result-dir naming) and forwards `layer_indices` into `vti_hook_ctx`. When `demos_path` is `demos_v2.jsonl` and/or `vector_dimension` is set (and no `directions_dir`), directions come from `compute_or_load_textual_directions_v2`.

`intervention.config` logs: `variant, hook_site, beta, num_demos, rank, seed, eps_coeff, log_lambda_sim, layer_indices, layer_set_label`, plus demos_v2 or directions_dir fields when applicable (`demos_path`, `demos_content_hash_sha256_16`, `dimension`, `selection_policy`, `steer_component` / `steer_reconstruction`, `diff_polarity`, `token_policy`, `directions_dir`, `direction_slug`).

Visual factory drops `beta`, `directions_dir`, `layer_indices`, `layer_set_label` so a shared kwargs dict cannot raise on visual registry keys.

#### `VTIVisualIntervention` — `evaluation/interventions/vti/visual_intervention.py`

```python
VTIVisualIntervention(
    model_id: str,
    variant: str = "uniform_rotation",   # additive | uniform_rotation
    hook_site: str = "mlp",              # ViT mlp | layer
    alpha: float = 0.9,                  # visual steering coefficient (paper α)
    num_demos: int = 70,
    seed: int = 42,
    eps_coeff: float = 0.1,
    perturb_type: str = "patch_mask",    # patch_mask | gaussian_noise
    mask_ratio: float = 0.99,
    mask_fill: str = "zero",             # zero | mean
    noise_sigma: float = 0.1,
    num_trials: int = 50,
    direction_recon: str = "top_pc",     # top_pc | legacy_pc_plus_mean
    include_cls: bool = True,
    patch_size: int = 14,
    demos_path: Optional[Path] = None,
    direction_cache: Optional[Path] = None,
)
```

Lifecycle: `ensure_directions(wrapper)` on first `generate()` (expensive: ~70 demos × 50 ViT forwards per uncached config); `vti_visual_hook_ctx` during `generate_vl`. Coefficient **`alpha`** passed to `steer(..., alpha=alpha)`.

**Direction extraction** (`evaluation/interventions/vti/visual_directions.py`):

- Demos: seed-42 sample of 70 from `data/vti/demos.jsonl`; images `data/coco/train2014/`.
- Perturb clean pixels (`patch_mask` default); Δ = mean corrupted hidden states − clean; PCA per (layer, token).
- Cache: `experiment_artifacts/vti/{model_short}/visual/{config_slug}/directions.npz` + `metadata.json`.

**Textual direction extraction** (`evaluation/interventions/vti/directions.py` + `directions_v2.py`):

- **Author demos (legacy):** `data/vti/demos.jsonl` (`value` / `h_value`); seed-42 `random.sample` of `num_demos` (default 70); images `data/coco/train2014/`. `forward_vl(image, question + caption)` for paired captions; last-token stack; global flatten PCA; steering = reshape(`PC1 + mean`); cache `experiment_artifacts/vti/{model_short}/textual_directions_nd{N}_rank{R}_seed{S}.npz` (no demos-hash in filename).
- **demos_v2:** see **demos_v2 textual directions** under the demos_v2 section above (`shuffled_prefix`, dimension selector, `textual_v2/` identity-bearing cache).

**Steering op** (`evaluation/interventions/vti/steer.py`): `steer(x, direction, alpha, variant, eps_coeff=0.1)`.

**Hook site sensitivity (important for calibration).** The same `steer()` op behaves very differently by site. At `mlp` the steered tensor is the MLP sub-block output (diluted by the residual add); at `layer` it is the full post-residual stream, so the same `beta` is a much stronger effective intervention and compounds across all decoder layers. On LLaVA-1.5, `uniform_rotation_layer` / `gated_rotation_layer` collapse to empty (immediate-EOS) generations at the default `beta=0.9`, while the `mlp` site and `additive_layer` do not (see `RESEARCH_LOG.md` 2026-06-18). The `*_layer` rotation variants therefore need their own (smaller) `beta` calibration; they are not comparable to `*_mlp` at the same `beta`.

**Debug knobs on `vti_hook_ctx`** (default-off; defaults reproduce production behavior exactly): `steer_prefill=True` (set `False` to steer decode steps only) and `skip_first_token=False` (set `True` to leave sequence position 0 / BOS unsteered during prefill). Both individually prevent the layer-site collapse on LLaVA, implicating position-0 (attention-sink) prefill rotation. Not exposed through the intervention/registry — for diagnostics only.

**`layer_indices` on `vti_hook_ctx`** (optional; default `None` = all decoder layers): when set to a sequence of absolute decoder-layer indices `0..num_layers-1`, hooks are registered only on those layers (both `mlp` and `layer` sites). `directions` stays full-length `(num_layers, hidden_dim)` and is indexed by absolute layer; the existing `len(directions) == n_layers` check is unchanged. Perception-dump cell specs may carry `layer_indices`; `optional_steer_ctx` forwards it. Window helper `layer_windows(num_layers, width=10, stride=5)` in `src/prompt_spans.py` yields the inclusive bands used by the windowed steering dump grid.

**`gated_rotation` note.** Gating only fires on single-token (`x.size(1) < 2`) calls; during prefill (`seq ≥ 2`) it falls through to `lam=1.0`, identical to `uniform_rotation`. At decode steps `lam = 1 + clamp(cos(x, −d), 0)`, but cos(x,−d) is ≈0 in high-dim, so `lam ≈ 1`. Empirically `gated_rotation` ≈ `uniform_rotation` to ≤1 char of mean length (RESEARCH_LOG 2026-06-18 overnight).

**VTI rotation-strength experiment:** `evaluation/vti_rotation_strength/rotation_strength.py` (+ `run_scripts/run_rotation_strength_sweep.sh`). Sweeps the textual coefficient `beta` over N POPE samples for a layer-site rotation variant and reports, per beta: POPE `accuracy`/`precision`/`recall`/`f1`/`yes_ratio`/`n_unparsed` (via `score_pope_records`), `mean_len`, the empty/identical/changed-vs-baseline split, and decision-flip accounting vs the no-hook baseline — `n_decision_flips`, `n_flip_correct_to_wrong`, `n_flip_wrong_to_correct`, plus the **confusion-direction decomposition** `flip_tp_to_fn` (suppressed true detection), `flip_tn_to_fp` (induced hallucination), `flip_fn_to_tp` (recovered miss), `flip_fp_to_tn` (removed hallucination). Console `h+`/`h-` columns = induced/removed hallucinations. Includes decode-only and skip-position-0 probes at the max beta. Writes `evaluation/vti_rotation_strength/results/{run_date}/{model_short}/sweep_{variant}_layer_{split}_n{N}.json` (JSON keys `betas`, `metrics_by_beta`, `mitigation_probes_beta_max`). Args: `--model --variant --num_samples --pope_split --max_new_tokens --betas --run_date`. The run script defaults `VARIANTS=uniform_rotation` (gated dropped as redundant). (Relocated 2026-06-19 from `evaluation/interventions/vti/_debug_layer_rotation.py` + `experiment_artifacts/vti_layer_debug/`; pre-move JSONs in `results/2026-06-18/` carry the older `metrics_by_alpha` schema and lack precision/recall + flip decomposition.)

**CHAIR + AMBER diagnostics** (`evaluation/chair_amber_diagnostics/`, added 2026-06-22). Two experiments sharing one pinned subset draw + the frozen CHAIR cap (64) + the verbatim VTI prompt, characterizing the VTI interventions on a generative (CHAIR) and a multi-dimension discriminative (AMBER) benchmark for `llava-1.5-7b-hf` and `Qwen2.5-VL-7B-Instruct`.

- `draw_subsets.py` (+ `run_scripts/run_prep_subsets.sh`): draws + pins the shared subsets ONCE with a fixed seed (default 1234) to **tracked** files — `data/chair/pinned_chair_500.json` (500 random COCO val2014 image ids) and `data/amber/pinned_amber_disc_450.json` (stratified 150 each existence/attribute/relation, by the annotation-derived `category`). Each file is shaped `{benchmark: [ids], "_meta": {...}}` so it passes straight to `run_eval.py --subset_ids_file`. Deterministic; `--force` to redraw.
- `step0_chair_token_cap.py` (+ `run_scripts/run_step0_chair_cap.sh`): the cap-provenance probe (20 images, caps 64 vs 512, LLaVA no_intervention, verbatim prompt). Writes `evaluation/results/{run_date}/_diagnostics/step0_chair_token_cap.json`. Result frozen the cap at **64** (see RESEARCH_LOG 2026-06-22).
- **Experiment 1 — reproduction grid** (`run_scripts/run_exp1_repro_grid.sh`): the `run_vti_pope_beta_grid.sh` analog on CHAIR+AMBER. Runs `vti_textual_additive_mlp`, `vti_textual_additive_layer`, `vti_textual_uniform_rotation_mlp` (NOT `uniform_rotation_layer` — that is Exp 2) across the β grid (β outermost; baselines once per model/benchmark), both models, via `run_eval.py` with the pinned subsets, `--chair_prompt`, `--chair_max_new_tokens 64`, `--amber_task discriminative`. Per-cell outputs land in the standard tree `{run_date}/{model_short}/{chair|amber}/{iv}__b{beta}/metric_summary.json`. Resumable via `--skip_if_exists`.
- **Experiment 2 — rotation-strength sweep** (`rotation_strength_chair_amber.py` + `run_scripts/run_exp2_rotation_strength.sh`): the `vti_rotation_strength` analog. Sweeps β for `uniform_rotation @ layer` over the pinned subsets, per (model, benchmark), reusing the generic gen/classify/flip helpers from `evaluation/vti_rotation_strength/rotation_strength.py`. **β grid (distinct from Exp1):** `0.6 0.5 0.45 0.4 0.35 0.3 0.25 0.2 0.1` (fine grid in the live zone; same as POPE `run_rotation_strength_sweep.sh`; Exp1 uses the coarser `0.4 0.1 … 0.9` reproduction grid). Reports `metrics_by_beta` (CHAIR: chair_s/chair_i over non-empty, avg_objects_mentioned, avg_caption_len_chars, mean_len, n_empty/empty_fraction; AMBER: accuracy/f1/yes_ratio/neg_item_accuracy/pos_item_accuracy/n_unparsed/by_qtype + POPE-style decision-flip accounting), the empty/identical/changed collapse split, and decode-only + skip-position-0 mitigation probes at β_max. Writes `{run_date}/{model_short}/{chair|amber}_rotation_strength/sweep_uniform_rotation_layer_n{N}.json`. Resumable: it checkpoints per-sample records to `sweep_..._n{N}.checkpoint.json` (every 25 new samples), reloads any complete records by `id` on restart so an interrupted sweep does not regenerate them, deletes the checkpoint on completion, and with `--skip_if_exists` skips a (model,benchmark) sweep whose final JSON already exists.
  - **OOM guard + resolution cap (Qwen).** Each sample's generations are wrapped in a CUDA-OOM guard: on `out of memory` it calls `torch.cuda.empty_cache()`, records the id in `failed_ids`, and skips the sample (excluded from scoring) so one oversized image cannot crash the whole sweep. `kept_samples` mirrors `per_sample` 1:1 to keep the column arrays aligned for scoring; the output JSON adds `n_scored`, `n_failed_oom`, `failed_ids`, and `max_pixels`, and `failed_ids` is persisted in the checkpoint (skipped on resume when `max_pixels` is unchanged). `--max_pixels N` (bash env `MAX_PIXELS`) is forwarded to `create_wrapper` **only for Qwen2 models** (the base wrapper `__init__` rejects unknown kwargs) and is the recommended way to avoid the ViT-attention OOM above — e.g. `MAX_PIXELS=1003520`. Native/uncapped Qwen runs drop the large images via the guard (scoring over a strict subset).
- `make_diagnostic_summary.py` (+ `run_scripts/run_report.sh`): aggregates a `run_date`'s Exp1 cells + Exp2 sweeps into two sibling reports under `evaluation/results/{run_date}/`:
  - `_diagnostic_summary_chair_amber.md` — human-readable tables (facts only).
  - `_diagnostic_summary_chair_amber.json` — machine-readable twin for the analyst (`schema_version: 1`). Top-level keys: `header` (frozen run facts: CHAIR cap, subset seed/id files, β grid, models discovered), `experiment_1` (`chair` / `amber` arrays of per-model blocks), `experiment_2` (`chair` / `amber` arrays of per-model sweep blocks). Each Exp1 block: `baseline` (full `metric_summary.json` dict), `methods`, `betas`, `grid` (list of `{method, beta, …metrics, delta_*_pp}` where deltas are in **percentage points** vs baseline: `delta_chair_i_pp`, `delta_yes_ratio_pp`), plus AMBER-only `by_qtype_at_beta_0.4` (`{method: {existence|attribute|relation: {accuracy, yes_ratio, neg_item_accuracy}}}`). Each Exp2 block: `variant`, `hook_site`, `n_samples`, `source_file`, `collapse_onset_beta` (first β with `empty_fraction >= 0.5` on CHAIR), `curve` (list of `{beta, metrics}` copied from sweep JSON `metrics_by_beta`), `mitigation_probes_beta_max`. Robust to partial runs (missing cells omitted). Appends a pointer to `RESEARCH_LOG.md` unless `--no_log`. CLI: `--run_date --output_dir --no_log`.
- `sample_responses.py` (+ `helper_scripts/run_sample_responses.sh`): see
  **Helper scripts** — batch qualitative samples + HTML under
  `evaluation/results/{run_date}/_samples/`. The chair_amber
  `run_scripts/run_sample_responses.sh` wrapper delegates to the helper script.

**Unit tests:** `tests/test_VTI_text_steer.py`.

### Output files (per model × benchmark × intervention)

```
evaluation/results/{run_date}/{model_short}/{benchmark_key}/{intervention}/
    responses.json
    responses.checkpoint.json   # during run; deleted on success
    metric_summary.json
```

- `{run_date}` is `YYYY-MM-DD` (the `--run_date` value, default today). One day's
  full sweep is self-contained under its date dir.
- `{benchmark_key}` is the benchmark name, except POPE which is `pope_{split}`
  (`pope_random`, `pope_popular`, `pope_adversarial`) so splits never collide.
- Pre-date results (legacy `evaluation/results/{model_short}/...`) are still read
  by `print_comparison_table` as a fallback; old stale trees were moved under
  `evaluation/results/_archive_preschema_*/`.

`_run_one` resumes from `responses.json` and `responses.checkpoint.json` by sample `id` (merge into one record set). Checkpoints every 50 samples. With `--skip_if_exists`, skips entirely if both `responses.json` and `metric_summary.json` exist.

Also mirrored under `data/{benchmark_dir}/{model_short}/responses/{intervention}/` when using `prepare_data.py --phases responses`.

### `metric_summary.json` schema

Top-level fields written by `eval_runner._run_one`:

```python
{
    "model": str,              # normalized short name
    "benchmark": str,          # registry key
    "intervention": str,
    "intervention_config": dict,
    "metric": str,             # scorer-specific type name
    # plus scorer fields below
}
```

**POPE** (`metric: "accuracy"`):

```python
{
    "accuracy_overall", "precision_overall", "recall_overall", "f1_overall",
    "yes_ratio", "n_correct", "n_total", "n_unparsed",
    "by_category": {split: {accuracy, precision, recall, f1, n_total, n_unparsed, yes_ratio}}
}
```

Unparseable responses count as incorrect for accuracy; for P/R/F1 they are FN when ground truth is `yes` and uncounted when ground truth is `no`. Parser (`_normalize_yes_no` in `evaluation/classifiers/metrics.py`): **leading-token priority** (`^\s*yes\b` / `^\s*no\b`), then word-boundaried fallback (`\bno\b` / `\byes\b`) so `not`/`cannot`/`none` do not spuriously match `no`. Positive class for P/R/F1 = `yes` (VTI convention). Track `n_unparsed` — expect ≈0 on POPE.

**AMBER discriminative** (`metric: "amber_discriminative"`) — POPE-style yes-bias /
grounding breakdown (the `score_amber_records` dispatcher routes the
discriminative task here; generative stays on the old `task_accuracy` placeholder):

```python
{
    "metric": "amber_discriminative",
    "positive_class": "yes",            # POPE convention (AMBER-official uses "no")
    "metric_convention": "pope",
    "accuracy_overall", "precision_overall", "recall_overall", "f1_overall",
    "yes_ratio",                        # agreeableness signal (fraction of "yes" answers)
    "n_correct", "n_total", "n_unparsed",
    "neg_item_accuracy",                # accuracy on gold=="no" items — grounding-isolation headline
    "pos_item_accuracy",                # accuracy on gold=="yes" items
    "n_neg_total", "n_pos_total",
    "by_qtype": {                       # existence | attribute | relation
        qtype: {accuracy, precision, recall, f1, yes_ratio,
                neg_item_accuracy, pos_item_accuracy,
                n_correct, n_total, n_neg_total, n_pos_total, n_unparsed}
    },
}
```

Reuses POPE's `_normalize_yes_no` (so AMBER/POPE stay comparable) and the
`yes`-positive convention. **`yes_ratio` + `neg_item_accuracy` are the pair to
read:** a yes-drift (agreeableness) confound raises `yes_ratio` while
`neg_item_accuracy` falls; genuine grounding holds negatives at flat `yes_ratio`.
AMBER discriminative gold is negative-heavy (full set: 9427 `no` / 4789 `yes`).
Unparsed handling matches POPE exactly. (AMBER-official P/R uses positive=`no`;
not emitted — `neg_item_accuracy` captures that negative-detection signal.)

**AMBER generative** (`metric: "task_accuracy"`) — unchanged placeholder, out of
scope (generative gold is object lists in `annotations.json`, not yes/no, and is
not surfaced into `ground_truth`; proper AMBER-generative hallucination scoring is
a future task).

**HallusionBench** (`metric: "accuracy"`): `accuracy_overall`, `n_correct`, `n_total` (no split breakdown; uses `_normalize_yes_no` for yes/no items).

**MMHal-Bench** (`metric: "mmhal_judge"`) — judge-based, replaces the old
`reference_match` stub:

```python
{
    "metric": "mmhal_judge",
    "judge_name": str,              # resolved judge identity (e.g. "mock", "openai:gpt-4-0314")
    "avg_score": float,             # mean 0-6 rating over PARSED records (headline)
    "hallucination_rate": float,    # fraction flagged (rating < 3) over parsed
    "n_total": int,                 # records with a gt answer
    "n_unparsed_judge": int,        # judge replies with no/ambiguous rating (NOT scored 0)
    "by_category": {                # keyed by the 8 MMHal QUESTION TYPES (record.task),
        qtype: {"avg_score", "hallucination_rate", "n_total"}  # incl. "counting"
    },
}
```

Judge prompt = official MMHal GPT-4 template (verbatim, text-only: consumes
`raw.image_content`, not the image). Rating parsed via `rating: {0..6}` (exactly
one match else unparsed). Hallucination cutoff = rating `< 3` (official). n is
tiny (96 total, ~12/type) so per-type numbers are directional; `n_total` per type
is always emitted. Caveat: parse failures are surfaced (not scored 0 as the
official script does) so a non-responding judge doesn't deflate `avg_score`.

**CHAIR** (`metric: "chair"`) — rule-based object inventory, replaces the old
`chair_pending` stub:

```python
{
    "metric": "chair",
    "chair_i": float,              # instance-level: sum|halluc| / sum|mentioned|; LOWER better (over NON-EMPTY)
    "chair_s": float,              # sentence-level: frac captions with >=1 halluc; LOWER better (over NON-EMPTY)
    "n_total": int,                # scoreable captions (have raw.coco_id)
    "n_nonempty": int,             # captions with non-whitespace text (chair_i/s denominator)
    "n_empty": int,                # whitespace-only captions (UNDEFINED for CHAIR; excluded)
    "empty_fraction": float,       # n_empty / n_total — the collapse signal at high beta
    "n_missing_image_id": int,     # records lacking raw.coco_id (excluded)
    "avg_objects_mentioned": float,  # coverage control (over non-empty) — read CHAIR jointly with this
    "avg_caption_len_chars": float,  # length control (over ALL scoreable, incl empties → shows collapse)
}
```

`chair_i/s` must be read **jointly with** `avg_objects_mentioned` and
`avg_caption_len_chars` (caption length confounds CHAIR in both directions, and
the rotation intervention changes length per-model with opposite sign), and only
across runs with the same frozen `--chair_max_new_tokens`. **§C empty-caption
handling (added 2026-06-22):** at high steering strength the layer-site rotation
drives captions empty; `chair_i`'s denominator (`sum|mentioned|`) then collapses
and a degenerate "said nothing" run would look hallucination-free. Empty
captions are therefore treated as UNDEFINED — excluded from `chair_i`/`chair_s`/
`avg_objects_mentioned` (computed over `n_nonempty`) and reported separately as
`n_empty`/`empty_fraction`. `avg_caption_len_chars` deliberately stays over ALL
scoreable captions so the length collapse remains visible.

Per-sample records in `responses.json`:

```python
{"id", "question", "benchmark", "task", "ground_truth", "metadata", "response"}
```

Scorers: `evaluation/classifiers/metrics.py` →
`compute_metric_records(records, benchmark, *, judge=None)`; POPE scorer =
`score_pope_records`. The `judge` kwarg is consumed only by `mmhal_bench`
(`score_mmhal_records(records, *, judge)`); if MMHal is scored without a judge it
falls back to the offline mock judge. CHAIR scorer = `score_chair_records`. AMBER
scorer = `score_amber_records` (dispatches by `task`:
`score_amber_discriminative_records` for discriminative,
`_score_amber_generative` placeholder for generative).

**Supporting modules (added 2026-06-22):**
- `evaluation/classifiers/judges.py` — `Judge` protocol; `get_judge(spec)`
  (`mock` | `openai[:model]` | `anthropic[:model]` | `gemini[:model]`); the
  verbatim official MMHal judge prompt (`MMHAL_JUDGE_TEMPLATE`,
  `build_mmhal_prompt`); `parse_mmhal_rating`, `rating_to_hallucination`
  (`HALLUCINATION_RATING_THRESHOLD = 3`). Judge is text-only; **public benchmarks
  only**.
- `evaluation/classifiers/chair_objects.py` — `load_synonym_map()` (vendored
  canonical Rohrbach synonyms at `evaluation/classifiers/chair_synonyms.txt`,
  80 COCO classes), `load_gt_objects()` / `gt_objects_for_image(image_id)` (from
  `data/coco/annotations/instances_val2014.json`, module-cached),
  `parse_caption_objects(caption)` (longest-n-gram synonym match; no nltk/pattern
  deps — plural handling lives in the expanded key set).
- `src/paths.py` gained `coco_annotations_dir()` → `data/coco/annotations`.
- `evaluation/chair_amber_diagnostics/` — CHAIR+AMBER reproduction-grid + rotation-strength experiments (see the experiments section above); pinned subsets at `data/chair/pinned_chair_500.json` and `data/amber/pinned_amber_disc_450.json`.
- Loader/runner knobs (2026-06-22): `load_chair(subset_ids=, prompt_override=)`, `load_amber(subset_ids=)`, `run_evaluation(subset_ids_file=, chair_prompt=)` / `run_eval.py --subset_ids_file --chair_prompt`. `score_chair_records` now reports `n_empty`/`empty_fraction`/`n_nonempty` and computes chair_i/s over non-empty captions (§C empty-caption handling).

### POPE no-intervention baselines (2026-06-16)

Completed VTI-comparable baselines (`--limit 200` per split, 600 records/model):

| Model (HF id) | `model_short` | Avg Acc | Avg F1 |
|---------------|---------------|--------:|-------:|
| `llava-hf/llava-1.5-7b-hf` | `llava-1.5-7b-hf` | 84.7% | 85.2% |
| `Qwen/Qwen-VL-Chat` | `qwen-vl-chat` | 85.7% | 85.1% |
| `Qwen/Qwen2-VL-7B-Instruct` | `qwen2-vl-7b-instruct` | 88.0% | 87.4% |

Per-split metrics in each `metric_summary.json` under `by_category`. Full run record: `RESEARCH_LOG.md` (2026-06-16 entry).

---

## Adding new components

### Intervention

1. Subclass `InterventionBase` in `evaluation/interventions/`
2. Register in `evaluation/interventions/__init__.py` → `ALL_INTERVENTIONS`
3. Update this file

### Vision-encoder hooks / vision VTI arm (extend beyond LLaVA-1.5)

LLaVA-1.5 path is **implemented** (see **Visual encoder hooks**). To add another family:

1. Extend `VisionDispatch` / `get_vision_dispatch()` in `src/vision_dispatch.py` — do not bolt ViT paths onto decoder `FamilyDispatch` without documenting both.
2. Confirm ViT module paths, hidden dim, token count (including CLS policy) via `verify_vision_layout`.
3. Reuse `visual_{perturb,capture,directions,hooks,intervention}.py`; only dispatch + layout asserts should be family-specific.
4. Keep `alpha` orthogonal to textual `beta`; registry already filters kwargs per arm.
5. Update **Visual encoder hooks** / per-model anatomy table in this file.

### Diagnostic experiment

1. `diagnostic_experiments/{name}/` + `run_scripts/`
2. Use `src.paths` helpers
3. Write to `diagnostic_results_dir()` and optionally `experiment_artifacts_dir()`

### Benchmark

1. Loader + `_register()` in `src/dataset.py`
2. `data_scripts/download_{name}.py`
3. Eval loader in `evaluation/benchmarks/`
4. Scorer in `evaluation/classifiers/metrics.py`

---

## RESEARCH_LOG.md

Append-only run records (date, commit, commands, output paths, headline metrics). Implementation agent appends facts; analyst appends interpretation via Romanus. Do not rewrite history.

---

## Perception dumps (`diagnostic_experiments/perception_diag/` code; data under `data/{amber|pope}/`)

Retained **steered capture dumps** only (responses, first-token yes/no scores, last-prefill activations, prefill position norms). Prior multi-hypothesis analysis plans and derived plots were removed 2026-07-19 pending a simpler redesign. Dump and cell directories use plain-English names (no `s1` / `s3b` / `B0` shorthand).

**Layout (shared across diagnostic experiments):** pins, augmented prompt JSONLs, and dumps live under the **benchmark** data dirs — not under a `data/perception_*` tree and not under the experiment code tree.

```
data/amber/pinned_*.json
data/amber/augmented_amber{25,100}.jsonl
data/amber/dumps/{model_short}/{run_tag}/…
data/pope/pinned_*.json
data/pope/augmented_pope30_{yes,no}.jsonl
data/pope/augmented_pope_{yes,no}_120.jsonl
data/pope/dumps/{model_short}/{run_tag}/…
```

Builder templates + `augmentation_meta.json` / gallery remain under `diagnostic_experiments/perception_diag/{templates,augment/outputs}/`.

### Kept dumps

| Path | Model | Items | Steering settings | `max_new_tokens` |
|------|-------|-------|-------------------|------------------|
| `data/amber/dumps/llava-1.5-7b-hf/amber25_all_steering_settings/` | LLaVA-1.5-7B | AMBER-25 × 5 prompt conditions | all 13 settings below | 512 |
| `data/amber/dumps/qwen2.5-vl-7b-instruct/amber25_steering_settings/` | Qwen2.5-VL-7B | same | default subset (no `additive_layer_0.2` / `0.9`) | 128 |
| `data/amber/dumps/llava-1.5-7b-hf/amber100_baseline/` | LLaVA-1.5-7B | AMBER-100 × 5 | `baseline` only | 128 |
| `data/amber/dumps/qwen2.5-vl-7b-instruct/amber100_baseline/` | Qwen2.5-VL-7B | same | `baseline` only | 128 |
| `data/pope/dumps/{model}/pope30_yes_windowed_steering/` | both | POPE-30-yes × 2 (gold-conditional) | windowed grid + in-grid `baseline` | 128 |
| `data/pope/dumps/{model}/pope30_no_windowed_steering/` | both | POPE-30-no × 2 (gold-conditional) | same | 128 |
| `data/pope/dumps/{model}/pope_yes_120_baseline/` | both | POPE-yes-120 × 3 | `baseline` only | 128 |
| `data/pope/dumps/{model}/pope_no_120_baseline/` | both | POPE-no-120 × 3 | `baseline` only | 128 |

AMBER-100 / POPE-30 dumps are **separate run tags** from the AMBER-25 trees (do not merge manifests).

Per setting directory: `manifest.jsonl` (response + parse + `score_*` logits/probs), `acts/{item}__{condition}.npy` shape `(n_layers+1, hidden)` last-prefill token fp16, `norms/{item}__{condition}.npy` shape `(n_layers+1, seq_len)`, `metadata.json`. Run-level: `run_metadata.json`.

**Steering setting directory names:**

| Directory | Method | Site | Strength |
|-----------|--------|------|----------|
| `baseline` | none | — | 0 |
| `rotation_mlp_0.2` / `_0.5` / `_0.9` | `uniform_rotation` | `mlp` | 0.2 / 0.5 / 0.9 |
| `rotation_layer_0.2` / `_0.5` / `_0.9` | `uniform_rotation` | `layer` | 0.2 / 0.5 / 0.9 |
| `additive_mlp_0.2` / `_0.5` / `_0.9` | `additive` | `mlp` | 0.2 / 0.5 / 0.9 |
| `additive_layer_0.2` / `_0.5` / `_0.9` | `additive` | `layer` | 0.2 / 0.5 / 0.9 |

Direction: demos_v2 `all@nd200` slug `demosv2_9a44f4af_all_nd200_s42_r2_prefix` (loaded for steered cells; baseline dumps still load the model only).

**Pins / subsets:**

| Name | Pin file | Composition |
|------|----------|-------------|
| AMBER-25 | `data/vti/qual_subset_chair5_amber25.json` (`amber`) | 5 per stratum × 5 strata (existence×no, attr×yes/no, rel×yes/no) |
| AMBER-100 | `data/amber/pinned_amber_disc_100.json` | all AMBER-25 IDs + fill from AMBER-450 pin order → **20 per stratum** (same 5 strata; no existence×yes in AMBER) |
| POPE-30-yes | `data/pope/pinned_pope_existence_yes_30.json` | **30 unique gold=yes** from random split (`pinned_eval_ids.json` order); 10 distinct images; `content_hash` over `items`. Pin rebuilt 2026-07-21. Augment: `data/pope/augmented_pope30_yes.jsonl` |
| POPE-30-no | `data/pope/pinned_pope_existence_no_30.json` | **10 gold=no per split** (random/popular/adversarial), image-matched to POPE-30-yes; skip `(image_id, questioned-object)` collisions across splits. Augment: `data/pope/augmented_pope30_no.jsonl` |
| POPE-yes-120 | `data/pope/pinned_pope_existence_yes_120.json` | **40 gold=yes** per split |
| POPE-no-120 | `data/pope/pinned_pope_existence_no_120.json` | **40 gold=no** per split |
| AMBER-450 / POPE-600 | existing pins under `data/amber/`, `data/pope/` | larger eval pins |

**Rebuild script:** `data_scripts/rebuild_pope_yes_no_30_pins.py` (CPU-only; writes both pins + prints verification).

**Prompt conditions:** neutral + {tentative,assertive}×{toward_yes,toward_no} from `templates/leading_clauses_v1.json`, plus single **filler** `filler_b` from `templates/filler_clauses_v1.json` (v3): assertive-matched (`Answer the question about the image. `, **8 tokens** on LLaVA/Qwen2.5 — trailing space required). Builder: `augment/build_augmented_jsonl.py` (supports `--condition_ids`, `--out_name`, subsets `pope30_yes` / `pope30_no` / `pope_yes_120` / `pope_no_120`). Protocol augments: `data/pope/augmented_pope30_yes.jsonl` + `augmented_pope30_no.jsonl` (full leading+filler set); `data/pope/augmented_pope_yes_120.jsonl` (neutral + assertive_toward_no + filler_b); `data/pope/augmented_pope_no_120.jsonl` (neutral + assertive_toward_yes + filler_b).

### Creation scripts (kept)

- `run_dump.py` — dump driver (`capture/steered_capture.py`, `capture/dump_writer.py`); writes to `perception_dump_dir(benchmark, model, run_tag)`; `--cells all` or `default_subset` or explicit cell names (e.g. `baseline`); `--benchmark` optional override (else inferred from `run_tag` prefix). **Windowed grid (2026-07-21, baseline-in-grid 2026-07-21):** `--windowed_grid` builds cell specs after model load as **`baseline` (method=none) first**, then 3 configs (`additive`@`mlp`, `uniform_rotation`@`mlp`, `uniform_rotation`@`layer`) × strengths `{0.2,0.5,0.9}` × (`layer_windows` + `layers_all`) → **64 cells LLaVA / 55 Qwen**. Cell dirs like `baseline`, `rotation_mlp_0.5_layers_10_19`, `rotation_layer_0.9_layers_all`. `--conditions gold_conditional` keeps per item exactly `neutral` + assertive opposing gold (`assertive_toward_yes` if gold=no else `assertive_toward_no`). Both flags recorded in `run_metadata.json` (`window_grid`, `condition_filter`).
- `build_windowed_steering_summary.py` — consolidates windowed dumps into `…/windowed_steering_summary/windowed_steering_consolidated_results.json`. Datasets keyed as `pope30_yes` / `pope30_no` / `amber100` (each block has a `dataset` field). Yes/no run tags load **in-grid** `baseline` from the same dump tree. Partial builds via `_completeness`.
- `build_pope30_mlp_2x2_summary.py` — per-dataset 2×2 JSON (`--datasets pope30_yes pope30_no`): `{dataset}_mlp_2x2_mean_p_yes_raw_summary.json` and `{dataset}_mlp_2x2_accuracy_and_flips_summary.json`. Gold=yes: accuracy = fraction parsed yes; `flips_yes_to_no`. Gold=no: accuracy = fraction parsed no; `flips_no_to_yes`. Axes: model × additive/rotation @ mlp; rotation_layer excluded from 2×2.
- `plot_pope30_windowed_steering.py` — POPE-30-yes / POPE-30-no and AMBER-100 attribute/relation subset windowed-steering plots (`--model llava-1.5-7b-hf|qwen2.5-vl-7b-instruct`, `--datasets` including `pope30_yes`, `pope30_no`, `amber100_attribute_yes`, `amber100_attribute_no`, `amber100_relation_yes`, `amber100_relation_no`). POPE dumps: `pope30_{yes|no}_windowed_steering` with in-grid `baseline`. AMBER subsets: filter `amber100_windowed_steering` manifests by `(qtype, gold)` (n=20 items × 2 gold-conditional conditions); baseline from separate `amber100_baseline/baseline/` (not in-grid). Always draws no-intervention baseline as grey hatched bars. Per-model × dataset dirs: `…/plots/{llava-1.5-7b-hf,qwen2.5-vl-7b-instruct}/{dataset}/` with accuracy / flips / mean first-token probability PNGs per method (`rotation_mlp|rotation_layer|additive_mlp`). Gold=yes: accuracy = fraction parsed yes; flips yes→no; mean **P(yes)** (`score_p_yes_raw`). Gold=no: accuracy = fraction parsed no; flips no→yes; mean **P(no)** (`score_p_no_raw`). Flip plots draw **no** grey baseline bars (steering flips only; leading-clause vs steering are separate contrasts). Flip-plot y-max = `max(round(n_items/4), ceil(observed_max))` via `flips_ylim_max` (shared across panes in a figure; avoids the old 0…n+2 scale while not clipping tall bars). `--joint_mean_p_yes` / `--joint_flips` (exactly one `--datasets` entry) write LLaVA×Qwen × additive/rotation @ mlp consolidated PNGs under `…/plots/merged_plots/pope-30-runs/` (default names `{dataset}_mean_p_{yes|no}_raw_…` and `{dataset}_flips_from_baseline_{yes|no}_…`). `--joint_pope_llava` writes LLaVA-only POPE-30 merged flips PNG under `…/plots/merged_plots/pope-30-runs/` (`pope30_flips_from_baseline_by_layer_window_llava_gold_yes_vs_no_additive_vs_rotation_mlp.png`; same gold-yes/no × additive|rotation layout). `--joint_amber_llava` writes LLaVA-only AMBER-100 merged PNGs under `…/plots/merged_plots/amber-100-runs/`: one capability (`attribute`|`relation`, via `--amber_capabilities`) × metrics `{accuracy, mean_p, flips}`; layout rows = gold=yes then gold=no, columns = additive|rotation @ mlp, each cell stacks neutral + assertive; mean_p figure uses P(yes) on top and P(no) on bottom. **rotation @ layer β=0.9 omitted** from per-method plots; rotation @ layer omitted entirely from merged MLP figures. Qwen AMBER-100 windowed dumps not required for the four amber subset keys until those dumps exist.
- `build_amber100_llava_response_galleries.py` — LLaVA AMBER-100 HTML response galleries under `…/windowed_steering_summary/`. Modes: (1) `relation_yes_leading_toward_no_baseline` → `amber100_relation_gold_yes_leading_toward_no_baseline_response_gallery.html` (baseline only, relation×gold=yes×assertive_toward_no, n=20); (2) `gold_no_additive_mlp_layers_0_9` → `amber100_attribute_relation_gold_no_additive_mlp_layers_0_9_response_gallery.html` (attribute+relation×gold=no; neutral + assertive_toward_yes; baseline vs additive @ mlp layers 0–9 at β∈{0.2,0.5,0.9}). Reads `data/amber/augmented_amber100.jsonl` (pretty JSON objects), `amber100_baseline/baseline/`, `amber100_windowed_steering/`.
- `build_pope30_mlp_beta09_response_gallery.py` / `build_qwen_pope30_rotation_mlp_response_gallery.py` — **stale**: still point at deleted dump tags `pope30_windowed_steering` / `pope30_existence_yes_baseline`. Their HTML outputs were removed from `windowed_steering_summary/` (2026-07-22 cleanup).
- `augment/build_augmented_jsonl.py` — build leading-clause (+ filler) JSONL under `data/{amber|pope}/` (subset keys include `pope30_yes`, `pope30_no`, `pope_yes_120`, `pope_no_120`)
- `control/geometric_comparison/compare_deployed_vs_shuffled_control.py` — CPU-only per-layer cosine + magnitude of deployed `all`/nd200 vs shuffled-image control; writes CSV/plots/summary under the same directory (see **Geometric comparison** under demos_v2 / shuffled-control above)
- `data_scripts/rebuild_pope_yes_no_30_pins.py` — rebuild `pinned_pope_existence_{yes,no}_30.json`
- `run_scripts/run_pope30_yes_no_orchestrator.sh` — wait for GPU free → concurrent LLaVA yes+no → concurrent Qwen yes+no
- `run_scripts/run_llava_pope30_yes_no_concurrent.sh` / `run_qwen_pope30_yes_no_concurrent.sh` — concurrent yes+no dumps on one A6000 (Alex chat override vs plan’s sequential order)
- `run_scripts/run_llava_amber25_dump.sh` — recreate LLaVA AMBER-25 dump
- `run_scripts/run_qwen25_amber25_dump.sh` — recreate Qwen2.5 AMBER-25 dump
- `run_scripts/run_amber100_pope30_baseline_dumps.sh` — LLaVA or Qwen2.5 baseline dumps for AMBER-100 then POPE-30 (`llava`|`qwen25` arg; `max_new_tokens=128`)

### Windowed steering run tags

| Run tag | Model(s) | Items × conditions | Cells |
|---------|----------|--------------------|-------|
| `pope30_yes_windowed_steering` | LLaVA then Qwen (concurrent yes+no per model) | POPE-30-yes × 2 (neutral + assertive_toward_no) | **64** LLaVA / **55** Qwen (includes in-grid `baseline`) |
| `pope30_no_windowed_steering` | same | POPE-30-no × 2 (neutral + assertive_toward_yes) | same |
| `amber100_windowed_steering` | same | AMBER-100 × 2 | 63 / 54 steered (+ separate `amber100_baseline`) |

Direction: demos_v2 `all@nd200`. `max_new_tokens=128`. Qwen `max_pixels=1003520`. Dumps under `data/pope/dumps/{model}/{pope30_yes_windowed_steering,pope30_no_windowed_steering}/`.
---

## Leading-clause attention knockout (`diagnostic_experiments/leading_clause_attention_knockout/`)

Localization experiment (no steering): block attention from downstream query positions into the leading-clause token span over coarse layer windows; score first-token yes/no via `forward_with_knockout` / `ScoringTarget.yes_no`.

| Script | Role |
|--------|------|
| `compute_flip_sets.py` | Stage 0 (zero GPU): first-token↔parsed agreement; flip set = correct under neutral & wrong under misleading; stratum-matched stable set; writes `results/stage0/` + per-model `…/results/stage0/flip_and_stable_sets.json`. Reads manifests from `data/{amber|pope}/dumps/…`. Normalizes pretty-printed manifests to JSONL when `--normalize_manifests`. Exit code 2 if any model flip union `< min_flip_union` (default 15). |
| `run_knockout.py` | Prefill-only knockout cells; hard-fails tokenization invariant; eager verify optional. Loads prompts from `data/amber/augmented_amber100.jsonl` + `data/pope/augmented_pope30.jsonl` (legacy filename still referenced by the script; prefer `augmented_pope30_yes.jsonl` for new POPE-30 work). |
| `analyze_knockout.py` | Flip recovery / logit-difference recovery / stable false-flip / filler dilution / full-depth sanity; plain-English CSVs + PNGs. |
| `run_scripts/run_stage0_flip_sets.sh` | Wrapper for Stage 0 |
| `run_scripts/run_knockout_cell.sh` | `CUDA_VISIBLE_DEVICES=N bash … llava\|qwen25 last_token\|all_downstream` |

**Cells** (under `{model_short}/results/`): `llava_block_last_token_reading`, `llava_block_all_downstream_reading`, `qwen25_block_last_token_reading`, `qwen25_block_all_downstream_reading`.

**`knockout_records.jsonl` fields (per record):** `record_key`, `item_id`, `set_membership`, `condition_id`, `prompt` (full text scored for that row), `question_neutral`, `knockout`, `query_scope`, `layer_window`, `key_span_kind`, `scores`, `seq_len`, `gold`, `qtype`; knockout rows also include `layer_indices`, `clause_span`, `question_span`, `query_positions_n`.

**Target:** lambdab2, one free A6000, unsharded. Qwen uses `max_pixels=1003520`.

**Gate (2026-07-20 Stage 0, AMBER-100+POPE-30):** LLaVA flip union=27. Qwen2.5 flip union=11 (below default min 15). Alex override 2026-07-20: proceed with Qwen via `--allow_small_flip_set`. Continuous recovery / filler-shift reporting uses **mean yes-probability recovery** on `p_yes_raw` (not logits); ε is the 10th percentile of `|Δ p_yes|` on flip rows (probability scale). Windows plotted in ascending layer order. LLaVA full-depth anchor agreement=0.5135 (Gate 3 still open for discussion).

**POPE-120 assertive protocol (2026-07-20):**
- **POPE-yes-120:** pin `data/pope/pinned_pope_existence_yes_120.json`; augment `data/pope/augmented_pope_yes_120.jsonl` (neutral + assertive_toward_no + filler_b); dumps `data/pope/dumps/{model}/pope_yes_120_baseline/`; Stage 0 `results/stage0_pope_yes_120/`. LLaVA flip_union=6 (gate fail); Qwen flip_union=24 (pass).
- **POPE-no-120:** pin `data/pope/pinned_pope_existence_no_120.json`; augment `data/pope/augmented_pope_no_120.jsonl` (neutral + assertive_toward_yes + filler_b); dumps `data/pope/dumps/{model}/pope_no_120_baseline/`; Stage 0 `results/stage0_pope_no_120/`. LLaVA flip_union=2 (fail); Qwen flip_union=0 (fail).
- Dump script: `run_scripts/run_pope120_baseline_dumps.sh llava|qwen25 yes|no`.
