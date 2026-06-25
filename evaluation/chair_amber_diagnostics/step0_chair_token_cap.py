#!/usr/bin/env python3
"""Step 0 — CHAIR max-new-tokens provenance check.

The VTI paper is internally inconsistent on the CHAIR caption cap (appendix
says max_new_tokens=64 "due to computational complexity"; Table 2 caption says
512). Before freezing a cap for the CHAIR+AMBER diagnostics we generate captions
for a small fixed set of COCO val2014 images at BOTH caps on the LLaVA-1.5
no-intervention baseline and score each with the project CHAIR scorer, so the
chosen cap can be compared against the authors' reported LLaVA-1.5 baseline.

This is a provenance probe, not an experiment cell: no intervention, tiny n.
Captions are generated through `wrapper.generate_vl(image, prompt)` (the same
path the steered runs use) with the verbatim VTI prompt.

Usage:
  CUDA_VISIBLE_DEVICES=0 \\
    python evaluation/chair_amber_diagnostics/step0_chair_token_cap.py \\
      --num_images 20 --caps 64 512
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src.model import create_wrapper, _normalize_model_name  # noqa: E402
from src.dataset import combined_json_path  # noqa: E402
from evaluation.classifiers.metrics import score_chair_records  # noqa: E402
from evaluation.classifiers.chair_objects import (  # noqa: E402
    parse_caption_objects, gt_objects_for_image,
)

# Verbatim VTI CHAIR prompt (capital D, as written in the paper / released code).
VTI_CHAIR_PROMPT = "Please Describe this image in detail."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("MODEL", "llava-hf/llava-1.5-7b-hf"))
    p.add_argument("--num_images", type=int, default=20)
    p.add_argument("--caps", type=int, nargs="+", default=[64, 512])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--prompt", default=VTI_CHAIR_PROMPT)
    p.add_argument("--run_date", default=None)
    return p.parse_args()


def _draw_images(n: int, seed: int):
    """Fixed random sample of CHAIR (COCO val2014) entries: (id, coco_id, path)."""
    entries = json.load(open(combined_json_path("chair")))
    rng = random.Random(seed)
    chosen = rng.sample(range(len(entries)), n)
    out = []
    for idx in sorted(chosen):
        e = entries[idx]
        coco_id = (e.get("raw") or {}).get("coco_id")
        out.append({"id": e["id"], "coco_id": coco_id,
                    "image_path": e["image_path"]})
    return out


def _coverage(records) -> float:
    """Recall/coverage proxy: fraction of GT objects mentioned, pooled over n."""
    hit = tot = 0
    for r in records:
        cid = (r.get("metadata") or {}).get("raw", {}).get("coco_id")
        if cid is None:
            continue
        mentioned = parse_caption_objects(r.get("response") or "")
        gt = gt_objects_for_image(cid)
        hit += len(mentioned & gt)
        tot += len(gt)
    return hit / tot if tot else 0.0


def main():
    args = parse_args()
    run_date = args.run_date or datetime.now().strftime("%Y-%m-%d")
    model_short = _normalize_model_name(args.model)

    imgs = _draw_images(args.num_images, args.seed)
    print(f"model={args.model} ({model_short})  n_images={len(imgs)}  "
          f"seed={args.seed}  caps={args.caps}")
    print(f"prompt={args.prompt!r}")

    wrapper = create_wrapper(args.model).load()

    # Open each image once and reuse across caps (same ids both caps).
    pil_by_id = {im["id"]: Image.open(im["image_path"]).convert("RGB")
                 for im in imgs}

    rows = {}
    captions_by_cap = {}
    for cap in args.caps:
        print(f"\n--- generating @ max_new_tokens={cap} ---")
        records = []
        caps_out = []
        for i, im in enumerate(imgs, 1):
            resp = wrapper.generate_vl(pil_by_id[im["id"]], args.prompt,
                                       max_new_tokens=cap)
            records.append({"response": resp,
                            "metadata": {"raw": {"coco_id": im["coco_id"]}}})
            caps_out.append({"id": im["id"], "coco_id": im["coco_id"],
                             "caption": resp})
            if i % 5 == 0:
                print(f"  [{i}/{len(imgs)}]")
        m = score_chair_records(records)
        rows[str(cap)] = {
            "chair_s": round(m["chair_s"], 4),
            "chair_i": round(m["chair_i"], 4),
            "avg_objects_mentioned": round(m["avg_objects_mentioned"], 3),
            "avg_caption_len_chars": round(m["avg_caption_len_chars"], 1),
            "coverage_recall": round(_coverage(records), 4),
            "n_total": m["n_total"],
            "n_empty": m.get("n_empty", 0),
        }
        captions_by_cap[str(cap)] = caps_out

    print("\n=== Step 0 — CHAIR cap provenance (LLaVA-1.5, no_intervention) ===")
    hdr = (f"{'cap':>6} {'chair_s':>8} {'chair_i':>8} {'avg_obj':>8} "
           f"{'avg_len':>8} {'coverage':>9} {'n':>4}")
    print(hdr)
    for cap in args.caps:
        r = rows[str(cap)]
        print(f"{cap:>6} {r['chair_s']:>8} {r['chair_i']:>8} "
              f"{r['avg_objects_mentioned']:>8} {r['avg_caption_len_chars']:>8} "
              f"{r['coverage_recall']:>9} {r['n_total']:>4}")

    out_dir = (_PROJECT_ROOT / "evaluation" / "results" / run_date
               / "_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step0_chair_token_cap.json"
    payload = {
        "experiment": "chair_amber_diagnostics_step0",
        "model": model_short,
        "run_date": run_date,
        "seed": args.seed,
        "prompt": args.prompt,
        "num_images": len(imgs),
        "image_ids": [im["id"] for im in imgs],
        "coco_ids": [im["coco_id"] for im in imgs],
        "metrics_by_cap": rows,
        "captions_by_cap": captions_by_cap,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
