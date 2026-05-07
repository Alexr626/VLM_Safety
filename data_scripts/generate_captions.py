#!/usr/bin/env python3
"""
Generate Image Captions
========================
Produces text descriptions of images for a dataset. Captions are a prerequisite
for TT extraction and safety direction computation.

Captions can come from either a locally hosted VLM (e.g. LLaVA 1.5) or a hosted
vision-capable LLM (Anthropic / OpenAI).

Supported datasets
------------------
  holisafe       — main evaluation dataset (SSS/SSU samples)
  mm-safetybench — unsafe reference (ShiftDC Appendix A.3, Scenarios 01-07 & 09)
  llava-instruct — safe reference (ShiftDC Appendix A.3)

Outputs (under data/captions/)
-------
  {dataset_name}.json   — {sample_id: caption_str}

Usage
-----
  # Local VLM (default — preserves existing behavior):
  python data_scripts/generate_captions.py --dataset holisafe

  # Anthropic Claude API:
  python data_scripts/generate_captions.py --dataset holisafe --provider anthropic

  # OpenAI vision API:
  python data_scripts/generate_captions.py --dataset holisafe --provider openai

  # Override the API model:
  python data_scripts/generate_captions.py --provider anthropic --api_model claude-opus-4-7

  python data_scripts/generate_captions.py --dataset holisafe --skip_if_exists
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

from src.dataset import (
    load_holisafe, filter_subsets, filter_reference_subsets,
    load_image_for_sample, REFERENCE_REGISTRY, load_mssbench,
)

CAPTION_PROMPT = "Describe this image in detail in several sentences. The first sentence of your response should be 'This image . . .'"

_DEFAULT_API_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
}


# ── Provider implementations ────────────────────────────────────


def _encode_image(image: Image.Image, max_side: int = 1568) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image. Downscales if oversized."""
    img = image.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


def _make_anthropic_caller(model: str, max_new_tokens: int):
    """Return a callable that captions a PIL image via the Anthropic Messages API."""
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    def call(image: Image.Image) -> str:
        b64, media_type = _encode_image(image)
        resp = client.messages.create(
            model=model,
            max_tokens=max_new_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media_type, "data": b64,
                    }},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }],
        )
        return resp.content[0].text.strip()

    return call


def _make_openai_caller(model: str, max_new_tokens: int):
    """Return a callable that captions a PIL image via the OpenAI Chat Completions API."""
    import openai

    client = openai.OpenAI()  # uses OPENAI_API_KEY env var

    def call(image: Image.Image) -> str:
        b64, media_type = _encode_image(image)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_new_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{media_type};base64,{b64}",
                    }},
                ],
            }],
        )
        return resp.choices[0].message.content.strip()

    return call


# ── Sample loading ─────────────────────────────────────────────


def _load_eval_benchmark_for_captioning(benchmark_name: str, limit=None):
    """Load samples from an evaluation benchmark for captioning.

    Returns a list of sample dicts with 'id' and 'image_pil' (PIL.Image).
    Uses the evaluation benchmark loaders, which already produce matching
    EvalSample.id values. Only samples with images are included.

    Adding a new benchmark: import its loader in evaluation/benchmarks/__init__.py,
    then add an entry here. The sample_id convention must match what
    eval_runner.py uses for caption lookup.
    """
    if benchmark_name == "mm_safetybench":
        from evaluation.benchmarks import load_mm_safetybench
        eval_samples = load_mm_safetybench(limit_per_scenario=limit)
    elif benchmark_name == "figstep":
        from evaluation.benchmarks import load_figstep
        eval_samples = load_figstep(limit=limit)
    else:
        return None  # not a recognized eval benchmark

    # Deduplicate by image: same image may pair with multiple questions,
    # but we only need one caption per image id.
    seen = set()
    samples = []
    for s in eval_samples:
        if s.image is None or s.id in seen:
            continue
        seen.add(s.id)
        samples.append({
            "id": s.id,
            "image_pil": s.image,
            "text": s.question,
        })
    return samples


# Registry of eval benchmarks that can be captioned (for extensibility).
BENCHMARK_CAPTION_REGISTRY = {
    "mm_safetybench": lambda limit=None: _load_eval_benchmark_for_captioning("mm_safetybench", limit),
    "figstep":        lambda limit=None: _load_eval_benchmark_for_captioning("figstep", limit),
}


def load_samples(dataset, cache_dir, limit, ref_samples, ref_seed,
                 holisafe_subsets=None, holisafe_eval_only=False,
                 mssbench_split: str = "all"):
    if dataset == "holisafe":
        entries, images_base = load_holisafe(cache_dir=cache_dir)
        if holisafe_subsets is None and not holisafe_eval_only:
            sss, ssu = filter_subsets(entries, images_base)
            samples = sss + ssu
        else:
            buckets = filter_reference_subsets(entries, images_base)
            wanted = (set(holisafe_subsets) if holisafe_subsets
                      else set(buckets.keys()))
            samples = [s for k, lst in buckets.items() if k in wanted
                       for s in lst]
            if holisafe_eval_only:
                split_path = (_PROJECT_ROOT / "data" / "holisafe-bench"
                              / "train_eval_split.json")
                if not split_path.exists():
                    raise FileNotFoundError(
                        "--holisafe_eval_only requires "
                        f"{split_path}. Run: python -m src.dataset"
                    )
                with open(split_path) as f:
                    split = json.load(f)
                eval_ids = set()
                for k in ("sss", "ssu", "suu", "usu", "uuu"):
                    eval_ids.update(split.get(f"{k}_eval_ids", []))
                samples = [s for s in samples if s["id"] in eval_ids]
    elif dataset == "mssbench":
        samples = load_mssbench()
        if mssbench_split != "all":
            split_path = _PROJECT_ROOT / "data" / "mssbench" / "train_eval_split.json"
            if not split_path.exists():
                raise FileNotFoundError(
                    f"--mssbench_split={mssbench_split} requires {split_path}. "
                    "Run: python -m src.dataset --mssbench_split"
                )
            with open(split_path) as f:
                split = json.load(f)
            wanted = set(split[f"{mssbench_split}_sample_ids"])
            samples = [s for s in samples if s["id"] in wanted]
    elif dataset in BENCHMARK_CAPTION_REGISTRY:
        samples = BENCHMARK_CAPTION_REGISTRY[dataset](limit=limit)
        if samples is None:
            raise ValueError(f"Failed to load benchmark '{dataset}' for captioning.")
        return samples  # limit already applied by loader
    elif dataset in REFERENCE_REGISTRY:
        loader = REFERENCE_REGISTRY[dataset]["loader"]
        samples = loader(n_samples=ref_samples, seed=ref_seed)
    else:
        all_known = sorted(
            set(BENCHMARK_CAPTION_REGISTRY) | set(REFERENCE_REGISTRY)
            | {"holisafe", "mssbench"}
        )
        raise ValueError(
            f"Unknown dataset: '{dataset}'. Available: {all_known}"
        )
    return samples[:limit] if limit else samples


# ── CLI ─────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["local", "anthropic", "openai"], default="local",
                   help="Captioner backend. 'local' uses a hosted VLM (default); "
                        "'anthropic' / 'openai' use the respective hosted vision API.")
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf",
                   help="Local VLM model ID (used when --provider=local).")
    p.add_argument("--api_model", default=None,
                   help="Hosted-API model name (used when --provider is anthropic/openai). "
                        f"Defaults: {_DEFAULT_API_MODELS}.")
    p.add_argument("--dataset", default="holisafe",
                   help="Dataset to caption. 'holisafe' or any key in REFERENCE_REGISTRY.")
    p.add_argument("--output_dir", default=str(_PROJECT_ROOT / "data" / "captions"))
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ref_samples", type=int, default=160)
    p.add_argument("--ref_seed", type=int, default=42)
    p.add_argument("--skip_if_exists", action="store_true")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size for local VLM. Ignored for hosted APIs (one call per image).")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--holisafe_eval_only", action="store_true",
                   help="When --dataset=holisafe, restrict to samples in the "
                        "eval splits of train_eval_split.json (run "
                        "`python -m src.dataset` first to populate it).")
    p.add_argument("--holisafe_subsets", nargs="+", default=None,
                   choices=["SSS", "SSU", "SUU", "USU", "UUU"],
                   help="When --dataset=holisafe, restrict to samples whose "
                        "raw HoliSafe `type` matches one of these. Combined "
                        "with --holisafe_eval_only as intersection.")
    p.add_argument("--mssbench_split", choices=["all", "train", "eval"],
                   default="all",
                   help="When --dataset=mssbench, restrict to a split. "
                        "'all' captions every sample; 'train'/'eval' use "
                        "data/mssbench/train_eval_split.json.")
    return p.parse_args()


def _run_local(args, samples, captions):
    from src.model import create_wrapper
    from src.extraction import cleanup_gpu

    wrapper = create_wrapper(args.model).load()
    bs = args.batch_size
    pending = [s for s in samples if str(s["id"]) not in captions]
    if len(pending) < len(samples):
        print(f"  Skipping {len(samples) - len(pending)} samples already captioned.")
    for i in tqdm(range(0, len(pending), bs), desc=f"Captioning {args.dataset} (local)"):
        batch = pending[i:i + bs]
        images = [load_image_for_sample(s) for s in batch]
        valid_pairs = [(s, img) for s, img in zip(batch, images) if img is not None]
        for s, img in zip(batch, images):
            if img is None:
                captions[str(s["id"])] = ""
        if not valid_pairs:
            continue
        try:
            batch_caps = wrapper.generate_captions_batch(
                [img for _, img in valid_pairs],
                max_new_tokens=args.max_new_tokens,
            )
            for (s, _), cap in zip(valid_pairs, batch_caps):
                captions[str(s["id"])] = cap
        except Exception as e:
            print(f"Warning: batch {i//bs} — {e}")
            for s, _ in valid_pairs:
                captions[str(s["id"])] = ""
        cleanup_gpu()


def _run_api(args, samples, captions, out_path):
    """Per-image API path with checkpoint-based resume.

    For MSSBench, each image is reused across (variant × queries), so we
    deduplicate by image filename stem before calling the API. The on-disk
    schema stays per-sample-id.
    """
    api_model = args.api_model or _DEFAULT_API_MODELS[args.provider]
    print(f"Using {args.provider} API (model={api_model})")
    if args.provider == "anthropic":
        call_fn = _make_anthropic_caller(api_model, args.max_new_tokens)
    else:
        call_fn = _make_openai_caller(api_model, args.max_new_tokens)

    checkpoint_path = out_path.with_suffix(".checkpoint.json")
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            captions.update(json.load(f))
        print(f"  Resuming from checkpoint: {len(captions)} already done")

    dedup_by_stem = args.dataset == "mssbench"
    seen_stems: dict[str, str] = {}

    for sample in tqdm(samples, desc=f"Captioning {args.dataset} ({args.provider})"):
        sid = str(sample["id"])
        if sid in captions:
            if dedup_by_stem and sample.get("image_path"):
                seen_stems.setdefault(Path(sample["image_path"]).stem, captions[sid])
            continue
        if dedup_by_stem and sample.get("image_path"):
            stem = Path(sample["image_path"]).stem
            if stem in seen_stems:
                captions[sid] = seen_stems[stem]
                continue
        image = load_image_for_sample(sample)
        if image is None:
            captions[sid] = ""
            continue
        try:
            cap = call_fn(image)
            captions[sid] = cap
            if dedup_by_stem and sample.get("image_path"):
                seen_stems[Path(sample["image_path"]).stem] = cap
        except Exception as e:
            print(f"  Warning: sample {sid} — {e}")
            captions[sid] = ""

        if len(captions) % 10 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(captions, f, indent=2)

    if checkpoint_path.exists():
        checkpoint_path.unlink()


def main():
    args = parse_args()
    out_path = Path(args.output_dir) / f"{args.dataset}.json"

    if args.skip_if_exists and out_path.exists():
        print(f"Skipping — {out_path} already exists.")
        return

    samples = load_samples(
        args.dataset, args.cache_dir, args.limit,
        args.ref_samples, args.ref_seed,
        holisafe_subsets=args.holisafe_subsets,
        holisafe_eval_only=args.holisafe_eval_only,
        mssbench_split=args.mssbench_split,
    )
    print(f"Samples to process: {len(samples)}")

    captions: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            captions.update(json.load(f))
        print(f"Loaded {len(captions)} existing captions; will skip those.")

    # Image-stem dedup for MSSBench: each image is reused across
    # (variant × queries). Caption one representative per stem; back-fill the
    # rest from the stem→caption mapping. ~50% fewer VLM/API calls.
    samples_to_run = samples
    if args.dataset == "mssbench":
        stem_to_caption: dict[str, str] = {}
        # Seed the map from any captions that already exist on disk.
        for sample in samples:
            sid = str(sample["id"])
            if sid in captions and captions[sid] and sample.get("image_path"):
                stem_to_caption.setdefault(Path(sample["image_path"]).stem, captions[sid])
        # Pick one un-captioned sample per stem as the representative; the rest
        # will be back-filled after the captioning pass.
        seen_stems_in_run: set[str] = set(stem_to_caption.keys())
        representatives: list = []
        for sample in samples:
            sid = str(sample["id"])
            if sid in captions:
                continue
            if not sample.get("image_path"):
                representatives.append(sample)
                continue
            stem = Path(sample["image_path"]).stem
            if stem in seen_stems_in_run:
                continue
            seen_stems_in_run.add(stem)
            representatives.append(sample)
        n_back_fill = sum(
            1 for s in samples
            if str(s["id"]) not in captions
            and s.get("image_path")
            and Path(s["image_path"]).stem in seen_stems_in_run
            and s not in representatives
        )
        print(f"  MSSBench dedup: {len(representatives)} representative images "
              f"(of {len(samples)} samples); ~{n_back_fill} will be back-filled.")
        samples_to_run = representatives

    if args.provider == "local":
        _run_local(args, samples_to_run, captions)
    else:
        _run_api(args, samples_to_run, captions, out_path)

    # Back-fill any remaining MSSBench samples sharing an image stem.
    if args.dataset == "mssbench":
        stem_to_caption_post: dict[str, str] = {}
        for sample in samples:
            sid = str(sample["id"])
            if sid in captions and captions[sid] and sample.get("image_path"):
                stem_to_caption_post.setdefault(
                    Path(sample["image_path"]).stem, captions[sid]
                )
        n_back_filled = 0
        for sample in samples:
            sid = str(sample["id"])
            if sid in captions or not sample.get("image_path"):
                continue
            stem = Path(sample["image_path"]).stem
            if stem in stem_to_caption_post:
                captions[sid] = stem_to_caption_post[stem]
                n_back_filled += 1
        if n_back_filled:
            print(f"  Back-filled {n_back_filled} captions from shared image stems.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Done → {out_path} ({len(captions)} samples)")


if __name__ == "__main__":
    main()
