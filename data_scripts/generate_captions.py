#!/usr/bin/env python3
"""Generate image captions for hallucination benchmarks."""

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

from src.dataset import load_benchmark, load_image_for_sample, ALL_BENCHMARKS

CAPTION_PROMPT = (
    "Describe this image in detail in several sentences. "
    "The first sentence of your response should be 'This image . . .'"
)
_DEFAULT_API_MODELS = {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o-mini"}


def _encode_image(image: Image.Image, max_side: int = 1568) -> tuple[str, str]:
    img = image.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


def _make_anthropic_caller(model: str, max_new_tokens: int):
    import anthropic
    client = anthropic.Anthropic()

    def call(image: Image.Image) -> str:
        b64, media_type = _encode_image(image)
        resp = client.messages.create(
            model=model, max_tokens=max_new_tokens,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": CAPTION_PROMPT},
            ]}],
        )
        return resp.content[0].text.strip()
    return call


def _make_openai_caller(model: str, max_new_tokens: int):
    import openai
    client = openai.OpenAI()

    def call(image: Image.Image) -> str:
        b64, media_type = _encode_image(image)
        resp = client.chat.completions.create(
            model=model, max_tokens=max_new_tokens,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
            ]}],
        )
        return resp.choices[0].message.content.strip()
    return call


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["local", "anthropic", "openai"], default="local")
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--api_model", default=None)
    p.add_argument("--dataset", default="pope", choices=ALL_BENCHMARKS)
    p.add_argument("--output_dir", default=str(_PROJECT_ROOT / "data" / "captions"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip_if_exists", action="store_true")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output_dir) / f"{args.dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    captions = {}
    if out_path.exists():
        with open(out_path) as f:
            captions = json.load(f)
        if args.skip_if_exists and captions:
            print(f"Skipping — {len(captions)} captions already at {out_path}")
            return

    samples = load_benchmark(args.dataset, limit=args.limit)
    pending = [s for s in samples if str(s["id"]) not in captions]

    if args.provider == "local":
        from src.model import create_wrapper
        from src.extraction import cleanup_gpu
        wrapper = create_wrapper(args.model).load()
        bs = args.batch_size
        for i in tqdm(range(0, len(pending), bs), desc=f"Captioning {args.dataset}"):
            batch = pending[i:i + bs]
            images = [load_image_for_sample(s) for s in batch]
            valid = [(s, img) for s, img in zip(batch, images) if img is not None]
            for s, img in zip(batch, images):
                if img is None:
                    captions[str(s["id"])] = ""
            if not valid:
                continue
            try:
                caps = wrapper.generate_captions_batch(
                    [img for _, img in valid], max_new_tokens=args.max_new_tokens)
                for (s, _), cap in zip(valid, caps):
                    captions[str(s["id"])] = cap
            except Exception as e:
                print(f"Warning batch {i//bs}: {e}")
                for s, _ in valid:
                    captions[str(s["id"])] = ""
            cleanup_gpu()
    else:
        api_model = args.api_model or _DEFAULT_API_MODELS[args.provider]
        caller = (_make_anthropic_caller if args.provider == "anthropic"
                  else _make_openai_caller)(api_model, args.max_new_tokens)
        for s in tqdm(pending, desc=f"Captioning {args.dataset} ({args.provider})"):
            img = load_image_for_sample(s)
            captions[str(s["id"])] = caller(img) if img else ""

    with open(out_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Wrote {len(captions)} captions -> {out_path}")


if __name__ == "__main__":
    main()
