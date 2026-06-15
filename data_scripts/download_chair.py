#!/usr/bin/env python3
"""Download COCO val2014 + instances and build CHAIR combined.json."""

import json
import sys
import zipfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2014.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"


def _download(url: str, dest: Path):
    if dest.exists():
        return
    import urllib.request
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)


def main():
    coco_root = _PROJECT_ROOT / "data" / "coco"
    coco_root.mkdir(parents=True, exist_ok=True)
    val_dir = coco_root / "val2014"
    ann_path = coco_root / "annotations" / "instances_val2014.json"

    if not val_dir.exists() or not any(val_dir.glob("*.jpg")):
        zip_path = coco_root / "val2014.zip"
        _download(COCO_VAL_URL, zip_path)
        print("Extracting val2014 ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(coco_root)

    if not ann_path.exists():
        zip_path = coco_root / "annotations_trainval2014.zip"
        _download(COCO_ANN_URL, zip_path)
        print("Extracting annotations ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(coco_root)

    with open(ann_path) as f:
        coco = json.load(f)

    entries = []
    for img in coco["images"]:
        img_path = val_dir / img["file_name"]
        if not img_path.exists():
            continue
        entries.append({
            "id": f"chair_{img['id']:012d}",
            "image_path": str(img_path),
            "text": "Please describe this image in detail.",
            "label": "caption",
            "task": "generative",
            "category": "coco",
            "raw": {"coco_id": img["id"], "file_name": img["file_name"]},
        })

    out = _PROJECT_ROOT / "data" / "chair" / "combined.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} CHAIR entries -> {out}")


if __name__ == "__main__":
    main()
