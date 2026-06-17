#!/usr/bin/env python3
"""Download HallusionBench from GitHub + Google Drive and build combined.json."""

import json
import shutil
import sys
import zipfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data_scripts.gdrive_utils import download_gdrive, download_url, is_valid_zip

HALLUSION_JSON_URL = (
    "https://raw.githubusercontent.com/tianyi-lab/HallusionBench/main/HallusionBench.json"
)
# Images: hallusion_bench.zip (see AMBER README on the HallusionBench repo).
HALLUSION_GDRIVE_ID = "1eeO1i0G9BSZTE1yd5XeFwmrbe1hwyf_0"
_IMAGE_MARKER = "images/VD"


def _normalize_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    return filename.removeprefix("./").lstrip("/")


def _ensure_images(out_root: Path) -> None:
    marker = out_root / _IMAGE_MARKER
    if marker.parent.exists() and any(out_root.glob("images/**/*.png")):
        return

    zip_path = out_root / "hallusion_bench.zip"
    if not is_valid_zip(zip_path):
        if zip_path.exists():
            print(f"Removing invalid download: {zip_path}")
            zip_path.unlink()
        print("Downloading HallusionBench images from Google Drive ...")
        download_gdrive(
            HALLUSION_GDRIVE_ID,
            zip_path,
            label="HallusionBench image archive",
        )

    extract_root = out_root / "_hallusion_extract"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    print("Extracting HallusionBench images ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    img_dir = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for png in extract_root.rglob("*.png"):
        rel = png.relative_to(extract_root)
        dest = img_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.move(str(png), str(dest))

    nested = img_dir / "hallusion_bench"
    if nested.is_dir():
        for child in nested.iterdir():
            dest = img_dir / child.name
            if dest.exists():
                continue
            child.rename(dest)
        nested.rmdir()

    shutil.rmtree(extract_root, ignore_errors=True)

    if not any(out_root.glob("images/**/*.png")):
        raise FileNotFoundError(
            f"No HallusionBench images found under {out_root / 'images'}. "
            "Download manually from "
            "https://drive.google.com/file/d/1eeO1i0G9BSZTE1yd5XeFwmrbe1hwyf_0/view"
        )


def main():
    out_root = _PROJECT_ROOT / "data" / "hallusionbench"
    out_root.mkdir(parents=True, exist_ok=True)

    json_path = out_root / "HallusionBench.json"
    download_url(HALLUSION_JSON_URL, json_path)
    with open(json_path) as f:
        rows = json.load(f)

    _ensure_images(out_root)

    entries = []
    for i, row in enumerate(rows):
        rel = _normalize_filename(row.get("filename"))
        gt = str(row.get("gt_answer", ""))
        entries.append({
            "id": f"hallusionbench_{i:05d}",
            "image_path": rel,
            "text": row.get("question") or row.get("query", ""),
            "label": gt,
            "label_idx": int(gt) if gt in {"0", "1"} else None,
            "task": row.get("subcategory") or row.get("category"),
            "category": row.get("category"),
            "raw": row,
        })

    combined = out_root / "combined.json"
    with open(combined, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} HallusionBench entries -> {combined}")


if __name__ == "__main__":
    main()
