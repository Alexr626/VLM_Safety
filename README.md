# Which hallucination signals can control a vision-language model?

This is the starter codebase for our Algoverse project. It is a clean starting
point: the infrastructure is in place, but **none of the experiments have been
written yet**.

## The project in one paragraph

Vision-language models (VLMs) such as LLaVA often describe objects that are not
in the image. Many papers find internal signals that *detect* these
hallucinations, and some use a signal to *correct* the model by nudging its
hidden state while it generates. Detecting a hallucination and being able to fix
it are not the same thing. We hold one correction method fixed: a gated,
minimum-norm edit of the hidden state, as in LiSeCo for text models and BRACS
for VLMs. Then we swap in different signals for it to act on. They range from
BRACS's image-attention score up to freely trained linear probes. That shows
which signals actually give control. We test on two models (LLaVA-1.5-7B and
Qwen2.5-VL-7B-Instruct) and three object-hallucination benchmarks (CHAIR, POPE,
and AMBER).

## What is already here

- **Model loading.** One interface for both models: load, caption an image,
  run a forward pass, and generate greedily. LLaVA supports original-LLaVA square
  padding, which our protocol uses.
- **Access to model internals.** You can get any decoder layer, its RMSNorm
  gain, its query/key projections, the rotary embedding, the output head, and
  where the image tokens and BOS token sit in the prompt. It works the same way
  for both models.
- **Hooks into the residual stream.** `capture_residual` records hidden states
  at chosen layers, and `edit_residual` changes them, including during
  generation, one decoded token at a time.
- **Benchmarks.** Download scripts, loaders, and scorers for CHAIR (CHAIR_i,
  CHAIR_s, and object recall on COCO val2014), POPE, and AMBER.
- **A baseline evaluation runner.** It runs a model on the benchmarks with no
  intervention, resumes after interruptions, and prints a results table. New
  interventions plug into it.
- **LiSeCo reference code.** A small shim imports the LiSeCo authors' code from
  a local clone of their repo.
- **Checks.** CPU unit tests and a one-command GPU smoke test.

## What is not here yet

All of the experimental design is still to be built: the decoding loop that
reruns and forks generation, the cone-constrained correction, the probes and
calibration, the signal ladder, the BRACS and VGD reimplementations, and the
COCO image splits. New experiment code goes in [`experiments/`](experiments/).

## Getting started

```bash
git clone <this repo> && cd <this repo>

# LiSeCo authors' code. It is gitignored, and each person clones it locally.
git clone https://github.com/chengemily1/llm-control.git

# Environment, option A: conda (lab or local GPUs)
conda env create -f environment.yml
conda activate vlm_hallucination_control

# Environment, option B: pip (Kaggle or Colab, where torch is preinstalled)
pip install -r requirements.txt
```

Download the data (about 7 GB, mostly COCO val2014). It goes into `data/`,
which is gitignored:

```bash
bash data_scripts/download_all_benchmarks.sh
```

Check that everything works:

```bash
pytest -q tests/                                                       # CPU, a few seconds
python scripts/smoke_test.py --model llava-hf/llava-1.5-7b-hf --pad_to_square   # needs a GPU
```

Run a small baseline: CHAIR on 50 COCO images with LLaVA.

```bash
python evaluation/run_eval.py --model llava-hf/llava-1.5-7b-hf \
    --benchmarks chair --limit 50 --pad_to_square
```

Results are written to `evaluation/results/{date}/{model}/{benchmark}/{intervention}/`,
and a summary table is printed at the end.

## Things worth knowing

- **Kaggle T4 GPUs don't support bf16.** Qwen falls back to fp16 there; check
  its outputs before trusting them. On a 16 GB card, Qwen2.5-VL-7B is
  partly offloaded to the CPU, and `--max_pixels` caps its image-token count.
- **CHAIR protocol.** CHAIR follows the OPERA setup: the prompt "Please
  describe this image in detail.", greedy decoding, and 512 new tokens. Keep
  that length fixed across conditions, because caption length changes CHAIR
  scores.
- **Square padding is off by default.** Turn it on with
  `--pad_to_square` / `pad_to_square=True`, as our protocol does.
- **Rebuild `data/` on each machine.** Downloaded data contains absolute
  paths, so don't copy it between machines.

## Where things live

| Folder | What's in it |
|---|---|
| `src/` | Model wrappers, model internals, hooks, dataset loaders, path helpers, LiSeCo shim |
| `evaluation/` | Benchmark loaders, scorers, the intervention interface, the evaluation runner |
| `data_scripts/` | Download scripts for COCO, POPE, and AMBER |
| `experiments/` | Where the project's experiments go (empty for now) |
| `scripts/` | GPU smoke test |
| `tests/` | CPU unit tests |

If you use a coding agent (Cursor, Claude Code, Codex, ...), it will pick up
[`AGENTS.md`](AGENTS.md), which describes every module and convention in detail.
