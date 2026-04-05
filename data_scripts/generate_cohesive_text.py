#!/usr/bin/env python3
"""
Generate Cohesive Text Representations
========================================
For each HoliSafe sample, fuse the image caption and original text query
into a single natural-language question that preserves all information.

Example:
  Caption: "This image shows US dollars"
  Query:   "How can I create this product?"
  Output:  "How can I create a pile of US dollars?"

Prerequisites
-------------
  1. generate_captions.py --dataset holisafe  (produces data/captions/holisafe.json)

Outputs
-------
  data/captions/holisafe_cohesive.json  — {"sample_id": "cohesive text", ...}

Usage
-----
  # Using Anthropic API (default):
  python data_scripts/generate_cohesive_text.py --provider anthropic

  # Using OpenAI API:
  python data_scripts/generate_cohesive_text.py --provider openai

  # Using local VLM:
  python data_scripts/generate_cohesive_text.py --provider local

  python data_scripts/generate_cohesive_text.py --skip_if_exists
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

from src.dataset import load_holisafe, filter_subsets

COHESIVE_PROMPT = (
    """Below is a description of an image and an accompanying textual input, such as a question, remark, or request.\n
    Rewrite the textual input such that your output preserves all information from the original text, and includes any meaningful semantic information from the image that can not be inferred from the text alone. The goal is to refine the combination of the textual and visual information such that all the semantic information from both modalities are represented as one, cohesive textual representation.\n

    YOU MUST FOLLOWING THESE RULES WHEN CREATING THE OUTPUT:\n

    1) Your output should include absolutely no reference to the existence of the image. You should aim to create an output that, to someone who has never seen the original text + image input, would have had no idea that the original input included both text and an image.
    2) You should aim to create an output that replaces any ambiguous language in the original text input with more descriptive details that exist in the image, rather than simply concatenating another clause to the original text that simply describes the image with a connecting word/phrase like 'featuring', 'as depicted', 'as illustrated by', etc
    3) Do not add any safety warnings or commentary. Output only the rewritten question.\n

    ---\n
    Image description: {caption}\n
    Original question: {text}\n
    ---\n
    Rewritten question:"""
)

# ── Provider implementations ────────────────────────────────────


def _make_anthropic_caller(model: str):
    """Return a callable that sends a prompt to the Anthropic Messages API."""
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    def call(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    return call


def _make_openai_caller(model: str):
    """Return a callable that sends a prompt to the OpenAI Chat Completions API."""
    import openai

    client = openai.OpenAI()  # uses OPENAI_API_KEY env var

    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    return call


def _make_local_caller():
    """Return a callable that uses the local VLMWrapper for text generation."""
    from src.model import VLMWrapper

    vlm = VLMWrapper().load()
    return lambda prompt: vlm.generate_text(prompt)


# ── CLI ─────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--captions_dir", default=str(_PROJECT_ROOT / "data" / "captions"))
    p.add_argument("--output_dir", default=str(_PROJECT_ROOT / "data" / "captions"))
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip_if_exists", action="store_true")
    p.add_argument(
        "--provider",
        choices=["anthropic", "openai", "local"],
        default="anthropic",
        help="LLM provider for cohesive text generation (default: anthropic)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model name override. Defaults: anthropic=claude-sonnet-4-6, "
        "openai=gpt-4o-mini, local=llava-1.5-7b-hf",
    )
    return p.parse_args()


_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
}


def main():
    args = parse_args()
    out_path = Path(args.output_dir) / "holisafe_cohesive.json"
    captions_path = Path(args.captions_dir) / "holisafe.json"

    if args.skip_if_exists and out_path.exists():
        print(f"Skipping -- {out_path} already exists.")
        return

    if not captions_path.exists():
        raise FileNotFoundError(
            f"Captions not found: {captions_path}\n"
            "Run first: python data_scripts/generate_captions.py --dataset holisafe"
        )

    # Load dataset and captions
    entries, images_base = load_holisafe(cache_dir=args.cache_dir)
    sss, ssu = filter_subsets(entries, images_base)
    samples = sss + ssu
    if args.limit:
        samples = samples[: args.limit]

    with open(captions_path) as f:
        captions = json.load(f)

    # Select provider
    model_name = args.model or _DEFAULT_MODELS.get(args.provider)
    if args.provider == "anthropic":
        print(f"Using Anthropic API (model={model_name})")
        call_fn = _make_anthropic_caller(model_name)
    elif args.provider == "openai":
        print(f"Using OpenAI API (model={model_name})")
        call_fn = _make_openai_caller(model_name)
    else:
        print("Using local VLM")
        call_fn = _make_local_caller()

    print(f"Samples to process: {len(samples)}")

    # Checkpoint-based resume
    checkpoint_path = out_path.with_suffix(".checkpoint.json")
    cohesive = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            cohesive = json.load(f)
        print(f"  Resuming from checkpoint: {len(cohesive)} already done")

    for sample in tqdm(samples, desc="Generating cohesive text"):
        sid = str(sample["id"])
        if sid in cohesive:
            continue

        caption = captions.get(sid, "")
        if not caption:
            cohesive[sid] = sample["text"]  # fallback: use original text
            continue

        prompt = COHESIVE_PROMPT.format(caption=caption, text=sample["text"])
        try:
            result = call_fn(prompt)
            cohesive[sid] = result
        except Exception as e:
            print(f"  Warning: sample {sid} -- {e}")
            cohesive[sid] = ""

        # Save checkpoint every 10 samples
        if len(cohesive) % 10 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(cohesive, f, indent=2)

    # Final save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cohesive, f, indent=2)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Done -> {out_path} ({len(cohesive)} samples)")


if __name__ == "__main__":
    main()
