#!/usr/bin/env python3
"""GPU smoke test: load a wrapper, caption one image, read and edit one layer.

Checks, in order:
  1. the model loads and captions an image under the OPERA protocol
     (CAPTION_PROMPT, greedy, --max_new_tokens);
  2. model_internals resolves the decoder, RMSNorm gain and q/k projections;
  3. capture_residual records one state per forward call during generate();
  4. a no-op edit_residual leaves the greedy caption unchanged;
  5. the LiSeCo shim imports llm-control (skipped with a warning if not cloned).

Usage:
    python scripts/smoke_test.py --model llava-hf/llava-1.5-7b-hf --pad_to_square
    python scripts/smoke_test.py --model Qwen/Qwen2.5-VL-7B-Instruct --image path.jpg
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src.dataset import CAPTION_PROMPT  # noqa: E402
from src.hooks import capture_residual, edit_residual  # noqa: E402
from src.model import create_wrapper  # noqa: E402
from src.model_internals import get_internals  # noqa: E402
from src.paths import coco_val2014_dir  # noqa: E402


def _pick_image(path_arg):
    if path_arg:
        return Path(path_arg)
    first = next(iter(sorted(coco_val2014_dir().glob("*.jpg"))), None)
    if first is None:
        sys.exit("No --image given and data/coco/val2014/ is empty. "
                 "Run: python data_scripts/download_chair.py")
    return first


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--image", default=None, help="Image path (default: first COCO val2014 image).")
    p.add_argument("--layer", type=int, default=None, help="0-based decoder layer (default: middle).")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--pad_to_square", action="store_true")
    p.add_argument("--max_pixels", type=int, default=None)
    args = p.parse_args()

    img_path = _pick_image(args.image)
    image = Image.open(img_path).convert("RGB")
    print(f"image: {img_path}")

    wrapper = create_wrapper(args.model, pad_to_square=args.pad_to_square,
                             max_pixels=args.max_pixels).load()

    caption = wrapper.generate_vl(image, CAPTION_PROMPT, max_new_tokens=args.max_new_tokens)
    print(f"\n[1] caption ({args.max_new_tokens} tokens max):\n{caption}\n")

    mi = get_internals(wrapper)
    layer = args.layer if args.layer is not None else mi.num_layers // 2
    gain, eps = mi.rmsnorm_params(layer)
    q_proj, k_proj = mi.qk_proj(layer)
    inputs = wrapper.prepare_vl(image, CAPTION_PROMPT)
    span = mi.image_token_span(inputs["input_ids"])
    print(f"[2] family={mi.family} layers={mi.num_layers} heads={mi.num_heads} "
          f"kv_heads={mi.num_kv_heads} head_dim={mi.head_dim}")
    print(f"    layer {layer}: gain{tuple(gain.shape)} eps={eps} "
          f"q_proj{tuple(q_proj.weight.shape)} k_proj{tuple(k_proj.weight.shape)} "
          f"q_bias={q_proj.bias is not None}")
    print(f"    prompt len={inputs['input_ids'].shape[1]} image span={span} "
          f"bos={mi.bos_position(inputs['input_ids'])}")

    with capture_residual(mi, [layer], site="output") as store:
        caption_hooked = wrapper.generate_vl(image, CAPTION_PROMPT,
                                             max_new_tokens=args.max_new_tokens)
    states = store[layer]
    print(f"[3] captured {len(states)} forward calls at layer {layer}; "
          f"state shape {tuple(states[0].shape)}")
    assert len(states) >= 2, "expected prefill + at least one decode step"

    with edit_residual(mi, layer, lambda h, call_idx: None):
        caption_noop = wrapper.generate_vl(image, CAPTION_PROMPT,
                                           max_new_tokens=args.max_new_tokens)
    same = caption_noop == caption == caption_hooked
    print(f"[4] no-op edit preserves greedy caption: {same}")
    assert same, "hooks changed greedy output"

    try:
        from src.liseco import load_liseco_module
        cw = load_liseco_module("control_wrapper")
        print(f"[5] LiSeCo import ok: {cw.LiSeCoWrapper.__name__}")
    except FileNotFoundError as e:
        print(f"[5] WARNING: {e}")

    print("\nsmoke test passed")


if __name__ == "__main__":
    main()
