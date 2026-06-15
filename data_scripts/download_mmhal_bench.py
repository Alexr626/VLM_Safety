#!/usr/bin/env python3
"""Download MMHal-Bench from HuggingFace and build combined.json."""

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    out_root = _PROJECT_ROOT / "data" / "mmhal-bench"
    img_root = out_root / "images"
    img_root.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    repo = "SYHao/MMHAL-Bench"
    print(f"Loading {repo} from HuggingFace ...")
    try:
        ds = load_dataset(repo, split="train")
    except Exception:
        ds = load_dataset(repo, split="test")

    entries = []
    for i, row in enumerate(ds):
        img = row.get("image") or row.get("image_path")
        img_path = None
        if img is not None and hasattr(img, "convert"):
            img_path = img_root / f"mmhal_{i:05d}.jpg"
            if not img_path.exists():
                img.convert("RGB").save(img_path)
        elif isinstance(img, str) and img:
            img_path = Path(img)

        q = row.get("question") or row.get("query", "")
        ref = row.get("gt_answer") or row.get("answer") or row.get("response", "")
        entries.append({
            "id": f"mmhal_{i:05d}",
            "image_path": str(img_path) if img_path else None,
            "text": q,
            "label": str(ref),
            "label_idx": None,
            "task": row.get("question_type"),
            "category": row.get("question_topic") or row.get("category"),
            "raw": {k: v for k, v in row.items() if k != "image"},
        })

    combined = out_root / "combined.json"
    with open(combined, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} MMHal-Bench entries -> {combined}")


if __name__ == "__main__":
    main()
