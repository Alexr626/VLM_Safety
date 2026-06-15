#!/usr/bin/env python3
"""Download POPE benchmark files and build combined.json manifest."""

import json
import sys
import urllib.request
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

POPE_BASE = "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco"
POPE_SPLITS = ["coco_pope_random.json", "coco_pope_popular.json", "coco_pope_adversarial.json"]


def _load_pope_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    out_dir = _PROJECT_ROOT / "data" / "pope" / "output" / "coco"
    out_dir.mkdir(parents=True, exist_ok=True)
    coco_dir = _PROJECT_ROOT / "data" / "coco" / "val2014"
    coco_dir.mkdir(parents=True, exist_ok=True)

    print("Download POPE question files...")
    for fname in POPE_SPLITS:
        dest = out_dir / fname
        if not dest.exists():
            url = f"{POPE_BASE}/{fname}"
            print(f"  {url}")
            urllib.request.urlretrieve(url, dest)

    entries = []
    for jf in sorted(out_dir.glob("*.json")):
        split = jf.stem.replace("coco_pope_", "")
        rows = _load_pope_jsonl(jf)
        for i, row in enumerate(rows):
            img_key = row.get("image") or row.get("image_id", "")
            if str(img_key).endswith(".jpg"):
                img_fname = str(img_key)
            else:
                img_fname = f"COCO_val2014_{str(img_key).zfill(12)}.jpg"
            entries.append({
                "id": f"pope_{split}_{i:05d}",
                "image_path": str(coco_dir / img_fname),
                "text": row.get("text") or row.get("question", ""),
                "label": row.get("label", ""),
                "label_idx": 1 if str(row.get("label", "")).lower() == "yes" else 0,
                "task": split,
                "category": split,
                "raw": row,
            })

    combined = _PROJECT_ROOT / "data" / "pope" / "combined.json"
    with open(combined, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} entries -> {combined}")
    print("Note: COCO val2014 images must be present under data/coco/val2014/.")
    print("Run: python data_scripts/download_chair.py  (also fetches COCO val2014)")


if __name__ == "__main__":
    main()
