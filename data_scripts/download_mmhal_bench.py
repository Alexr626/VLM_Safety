#!/usr/bin/env python3
"""Download MMHal-Bench from HuggingFace and build combined.json."""

import json
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data_scripts.gdrive_utils import download_url, is_valid_zip

# Official dataset repo (not SYHao/MMHAL-Bench).
MMHAL_HF_BASE = "https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/resolve/main"
MMHAL_DATA_ZIP = f"{MMHAL_HF_BASE}/test_data.zip"
MMHAL_TEMPLATE = f"{MMHAL_HF_BASE}/response_template.json"


def _image_filename(image_src: str) -> str:
    return Path(urlparse(image_src).path).name


def _ensure_data(out_root: Path) -> None:
    img_dir = out_root / "images"
    if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) >= 90:
        return

    zip_path = out_root / "test_data.zip"
    if not is_valid_zip(zip_path):
        if zip_path.exists():
            print(f"Removing invalid download: {zip_path}")
            zip_path.unlink()
        print("Downloading MMHal-Bench test_data.zip from HuggingFace ...")
        print("(~170 MB; this may take a few minutes.)")
        download_url(MMHAL_DATA_ZIP, zip_path)

    extract_root = out_root / "_mmhal_extract"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    print("Extracting MMHal-Bench images ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    img_dir.mkdir(parents=True, exist_ok=True)
    src_images = extract_root / "images"
    if src_images.is_dir():
        for jpg in src_images.glob("*.jpg"):
            dest = img_dir / jpg.name
            if not dest.exists():
                shutil.move(str(jpg), str(dest))

    shutil.rmtree(extract_root, ignore_errors=True)

    if len(list(img_dir.glob("*.jpg"))) < 90:
        raise FileNotFoundError(
            f"Expected ~96 MMHal-Bench images under {img_dir}, found fewer. "
            f"Re-download from {MMHAL_DATA_ZIP}"
        )


def _load_questions(out_root: Path) -> list:
    template_path = out_root / "response_template.json"
    if not template_path.exists():
        download_url(MMHAL_TEMPLATE, template_path)
    with open(template_path) as f:
        return json.load(f)


def main():
    out_root = _PROJECT_ROOT / "data" / "mmhal-bench"
    out_root.mkdir(parents=True, exist_ok=True)

    _ensure_data(out_root)
    rows = _load_questions(out_root)

    entries = []
    for i, row in enumerate(rows):
        img_name = _image_filename(row.get("image_src", ""))
        entries.append({
            "id": f"mmhal_{i:05d}",
            "image_path": img_name,
            "text": row.get("question") or row.get("query", ""),
            "label": str(row.get("gt_answer") or row.get("answer", "")),
            "label_idx": None,
            "task": row.get("question_type"),
            "category": row.get("question_topic") or row.get("category"),
            "raw": row,
        })

    combined = out_root / "combined.json"
    with open(combined, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} MMHal-Bench entries -> {combined}")


if __name__ == "__main__":
    main()
