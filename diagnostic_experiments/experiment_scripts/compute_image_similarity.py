#!/usr/bin/env python3
"""
Compute DINOv2 cosine similarity between SSS and SSU image variants for
every MSSBench chat record.

Output JSON is *model-independent* (DINOv2 features only — has nothing to
do with the VLM under test) and is consumed by the causal-mediation
script's tier filter and by the similarity-distribution plotter.

Embodied records use a different schema (no `queries` field) and are
silently skipped — diagnostic-side `load_mssbench` already filters them
out for the same reason.

Usage
-----
    python diagnostic_experiments/experiment_scripts/compute_image_similarity.py \
        --dataset mssbench \
        --dinov2_model facebook/dinov2-large \
        --batch_size 16 \
        [--skip_if_exists]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_DIAGNOSTIC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DIAGNOSTIC_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import split_mssbench_train_eval


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mssbench"], default="mssbench",
                   help="Currently only MSSBench is supported.")
    p.add_argument("--dinov2_model", default="facebook/dinov2-large")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--output", default=None,
                   help="Override output JSON path. Default: "
                        "data/mssbench/image_similarity/dinov2_similarity_scores.json")
    p.add_argument("--skip_if_exists", action="store_true",
                   help="If the output JSON already exists, exit early.")
    return p.parse_args()


def _enumerate_chat_stems(data_dir: Path):
    """Yield (rec_idx, record_dict, safe_path, unsafe_path) for every
    MSSBench *chat* record that has both image variants present on disk.

    Embodied records are skipped (different schema; not needed for this
    analysis).
    """
    combined_path = data_dir / "combined.json"
    with open(combined_path) as f:
        combined = json.load(f)
    chat_records = combined.get("chat", [])
    base = data_dir / "chat"
    for rec_idx, rec in enumerate(chat_records):
        safe_rel = rec.get("safe_image_path")
        unsafe_rel = rec.get("unsafe_image_path")
        if not (isinstance(safe_rel, str) and isinstance(unsafe_rel, str)):
            continue
        safe_path = base / safe_rel
        unsafe_path = base / unsafe_rel
        if not (safe_path.exists() and unsafe_path.exists()):
            continue
        yield rec_idx, rec, safe_path, unsafe_path


def _load_dinov2(model_id: str, device: str, dtype):
    from transformers import AutoImageProcessor, AutoModel
    print(f"Loading DINOv2 ({model_id}) ...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device).eval()
    return processor, model


@torch.no_grad()
def _embed_batch(processor, model, images, device):
    """Return L2-normalised CLS-token embeddings, shape (B, D)."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"].to(model.dtype)
    out = model(pixel_values=pixel_values)
    cls = out.last_hidden_state[:, 0, :]
    cls = cls.float()
    cls = torch.nn.functional.normalize(cls, dim=-1)
    return cls.cpu().numpy()


def main():
    args = parse_args()

    data_dir = _PROJECT_ROOT / "data" / "mssbench"
    if args.output is None:
        out_path = data_dir / "image_similarity" / "dinov2_similarity_scores.json"
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_if_exists and out_path.exists():
        print(f"Output already exists at {out_path} — skipping.")
        return

    # ── Train/eval split (record-level) so we can flag stems ─────────────
    split = split_mssbench_train_eval(seed=42, train_frac=0.75)
    train_records = set(split["train_record_ids"])

    # ── Enumerate chat stems ─────────────────────────────────────────────
    stems_meta = list(_enumerate_chat_stems(data_dir))
    print(f"Found {len(stems_meta)} chat records with both image variants.")
    if not stems_meta:
        raise RuntimeError(
            f"No MSSBench chat records found under {data_dir}/chat/. "
            "Run evaluation/scripts/download_mssbench.py first."
        )

    # ── Load DINOv2 ──────────────────────────────────────────────────────
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.torch_dtype]
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"  CUDA unavailable — falling back to {device}")
    processor, model = _load_dinov2(args.dinov2_model, device, dtype)

    # ── Compute per-stem similarity in batches of paired images ──────────
    out_stems = []
    safe_paths = [m[2] for m in stems_meta]
    unsafe_paths = [m[3] for m in stems_meta]
    n = len(stems_meta)
    bs = max(1, args.batch_size)

    cos_scores = np.zeros(n, dtype=np.float32)
    for start in range(0, n, bs):
        end = min(n, start + bs)
        safe_imgs = [Image.open(p).convert("RGB") for p in safe_paths[start:end]]
        unsafe_imgs = [Image.open(p).convert("RGB") for p in unsafe_paths[start:end]]
        safe_emb = _embed_batch(processor, model, safe_imgs, device)
        unsafe_emb = _embed_batch(processor, model, unsafe_imgs, device)
        # Both already L2-normalised → dot product == cosine similarity.
        cos = (safe_emb * unsafe_emb).sum(axis=-1)
        cos_scores[start:end] = cos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (start // bs) % 5 == 0:
            print(f"  [{end}/{n}] last batch mean cos = {float(cos.mean()):.4f}")

    for (rec_idx, rec, safe_path, unsafe_path), cos in zip(stems_meta, cos_scores):
        out_stems.append({
            "rec_idx": int(rec_idx),
            "subset": "chat",
            "type": rec.get("Type") or rec.get("type") or "unknown",
            "safe_image_path": str(safe_path.relative_to(_PROJECT_ROOT)),
            "unsafe_image_path": str(unsafe_path.relative_to(_PROJECT_ROOT)),
            "cosine_similarity": float(cos),
            "in_train_split": bool(rec_idx in train_records),
        })

    summary = {
        "model": args.dinov2_model,
        "dtype": args.torch_dtype,
        "n_stems": len(out_stems),
        "n_train_split_stems": sum(1 for s in out_stems if s["in_train_split"]),
        "train_split_meta": {
            "seed": split["seed"],
            "train_frac": split["train_frac"],
        },
        "stems": out_stems,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {len(out_stems)} stem records → {out_path}")


if __name__ == "__main__":
    main()
