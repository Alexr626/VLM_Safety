# VLM hallucination control

Starter codebase for **"Which hallucination signals can control a vision-language
model?"**: a gated minimum-norm hidden-state correction (LiSeCo / BRACS style)
held fixed while the signal it acts on varies, scored on object hallucination
(CHAIR, POPE, AMBER) with LLaVA-1.5-7B and Qwen2.5-VL-7B-Instruct.

This repo is a skeleton. It has model wrappers, model-internals access, residual
hooks, benchmark loaders, and scorers. None of the experiments are implemented
yet (see [`experiments/README.md`](experiments/README.md)).

## Setup

```bash
git clone <this repo> && cd <this repo>

# LiSeCo reference code (gitignored; imported through src/liseco.py)
git clone https://github.com/chengemily1/llm-control.git

# Option A: conda (local / lab GPUs)
conda env create -f environment.yml
conda activate vlm_hallucination_control

# Option B: pip (Kaggle / Colab, where torch is preinstalled)
pip install -r requirements.txt
```

Notes:
- **Kaggle T4s have no native bf16.** The Qwen wrapper falls back to fp16 there,
  and LLaVA always runs in fp16. Check Qwen fp16 outputs for NaNs or garbage
  before trusting them.
- On a 16 GB T4, Qwen2.5-VL-7B (~16.6 GB) is partially CPU-offloaded
  automatically. Pass `max_pixels` to cap its vision tokens.
- Set `HF_HOME` to a large disk if your home directory is small.

## Data

```bash
bash data_scripts/download_all_benchmarks.sh
# or individually (CHAIR first: it fetches COCO val2014, which POPE also uses)
python data_scripts/download_chair.py   # COCO val2014 + 2014 annotations (~6.5 GB)
python data_scripts/download_pope.py    # POPE random/popular/adversarial
python data_scripts/download_amber.py   # AMBER repo + images (Google Drive)
```

Everything lands under `data/` (gitignored). Each benchmark gets a
`data/{benchmark}/combined.json` manifest with absolute image paths, so rebuild
it on each machine instead of copying it.

## Check the install

```bash
pytest -q tests/                                                   # CPU only
python scripts/smoke_test.py --model llava-hf/llava-1.5-7b-hf --pad_to_square
python scripts/smoke_test.py --model Qwen/Qwen2.5-VL-7B-Instruct
```

The smoke test captions one COCO image and resolves the decoder internals. It
then captures one layer's residual stream during `generate()`, checks that a
no-op edit hook leaves the greedy caption unchanged, and imports LiSeCo.

## Baseline evaluation

```bash
python evaluation/run_eval.py --model llava-hf/llava-1.5-7b-hf \
    --benchmarks chair --limit 50 --pad_to_square
python evaluation/run_eval.py --model llava-hf/llava-1.5-7b-hf \
    --benchmarks pope --pope_split adversarial --pad_to_square
```

CHAIR uses the OPERA protocol by default: "Please describe this image in
detail.", greedy decoding, and 512 new tokens. Results go to
`evaluation/results/{date}/{model}/{benchmark}/{intervention}/`. CHAIR summaries
report CHAIR_s, CHAIR_i, object recall, mean objects mentioned, and caption
length.

## Layout

```
src/
  model.py            LLaVAWrapper (pad_to_square), Qwen2VLWrapper, create_wrapper
  model_internals.py  decoder layers, RMSNorm gain/eps, q/k proj, rotary, lm_head, spans
  hooks.py            capture_residual / edit_residual (work inside generate())
  dataset.py          POPE / AMBER / CHAIR loaders (+ CAPTION_PROMPT, CAPTION_MAX_NEW_TOKENS)
  extraction.py       ActivationCache, last-token activations, JSON/NPZ I/O
  paths.py            path helpers (data, experiments, artifacts, llm-control)
  liseco.py           import shim for llm-control/src
evaluation/
  benchmarks/         EvalSample + benchmark loaders
  classifiers/        CHAIR (synonyms, COCO GT, recall), POPE, AMBER scorers
  interventions/      InterventionBase + no_intervention registry
  runners/            eval_runner (resume, checkpoints, comparison table)
  run_eval.py         CLI
data_scripts/         COCO / POPE / AMBER downloaders
experiments/          experiment code goes here (empty)
scripts/smoke_test.py GPU sanity check
tests/                CPU tests
```

See [`CLAUDE.md`](CLAUDE.md) for module-level details and conventions.
