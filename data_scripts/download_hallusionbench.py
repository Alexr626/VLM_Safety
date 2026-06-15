#!/usr/bin/env python3
"""Download HallusionBench from HuggingFace and build combined.json."""

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    out_root = _PROJECT_ROOT / "data" / "hallusionbench"
    img_root = out_root / "images"
    img_root.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    print("Loading tianyi-lab/HallusionBench from HuggingFace ...")
    ds = load_dataset("tianyi-lab/HallusionBench", split="train")

    entries = []
    for i, row in enumerate(ds):
        img = row.get("image")
        img_path = None
        if img is not None:
            img_path = img_root / f"hallusion_{i:05d}.jpg"
            if not img_path.exists():
                img.convert("RGB").save(img_path)
        q = row.get("question") or row.get("query", "")
        gt = row.get("gt_answer") or row.get("answer") or row.get("label", "")
        entries.append({
            "id": f"hallusionbench_{i:05d}",
            "image_path": str(img_path) if img_path else None,
            "text": q,
            "label": str(gt),
            "label_idx": None,
            "task": row.get("subcategory") or row.get("category"),
            "category": row.get("category"),
            "raw": {k: v for k, v in row.items() if k != "image"},
        })

    combined = out_root / "combined.json"
    with open(combined, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} HallusionBench entries -> {combined}")


if __name__ == "__main__":
    main()
