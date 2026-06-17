#!/usr/bin/env python3
"""Download AMBER benchmark and build combined.json."""

import json
import sys
import zipfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data_scripts.gdrive_utils import download_gdrive, download_url, is_valid_zip

# Repo default branch is master (not main).
AMBER_REPO_ZIP = "https://github.com/junyangwang0410/AMBER/archive/refs/heads/master.zip"
# Images are hosted separately on Google Drive (see AMBER README).
AMBER_GDRIVE_ID = "1MaCHgtupcZUjf007anNl4_MV0o4DjXvl"
_QUERY_MARKER = "data/query/query_discriminative.json"


def _ensure_repo(out_root: Path) -> None:
    query_path = out_root / _QUERY_MARKER
    if query_path.exists():
        return

    zip_path = out_root / "amber_repo.zip"
    download_url(AMBER_REPO_ZIP, zip_path)

    print("Extracting AMBER repo ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_root)

    extracted = next(out_root.glob("AMBER-*"), None)
    if extracted and extracted.is_dir():
        for child in extracted.iterdir():
            dest = out_root / child.name
            if dest.exists():
                continue
            child.rename(dest)


def _ensure_images(out_root: Path) -> None:
    img_dir = out_root / "images"
    if img_dir.exists() and any(img_dir.glob("*.jpg")):
        return

    zip_path = out_root / "amber_images.zip"
    if not is_valid_zip(zip_path):
        if zip_path.exists():
            print(f"Removing invalid download: {zip_path}")
            zip_path.unlink()
        print("Downloading AMBER images from Google Drive ...")
        print("(~400 MB; this may take several minutes.)")
        download_gdrive(AMBER_GDRIVE_ID, zip_path, label="AMBER image archive")

    extract_root = out_root / "_amber_images_extract"
    if extract_root.exists():
        import shutil
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    print("Extracting AMBER images ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    img_dir.mkdir(parents=True, exist_ok=True)
    nested = extract_root / "images"
    if nested.is_dir() and any(nested.glob("*.jpg")):
        import shutil
        for jpg in nested.glob("*.jpg"):
            dest = img_dir / jpg.name
            if not dest.exists():
                shutil.move(str(jpg), str(dest))
    else:
        for jpg in extract_root.rglob("*.jpg"):
            dest = img_dir / jpg.name
            if not dest.exists():
                import shutil
                shutil.move(str(jpg), str(dest))

    import shutil
    shutil.rmtree(extract_root, ignore_errors=True)

    if not any(img_dir.glob("*.jpg")):
        raise FileNotFoundError(
            f"No AMBER images found after extracting {zip_path}. "
            "Download manually from "
            "https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view "
            f"and place JPGs under {img_dir}/"
        )


def main():
    out_root = _PROJECT_ROOT / "data" / "amber"
    out_root.mkdir(parents=True, exist_ok=True)

    _ensure_repo(out_root)
    _ensure_images(out_root)

    entries = []
    disc_path = out_root / "data" / "query" / "query_discriminative.json"
    if disc_path.exists():
        with open(disc_path) as f:
            disc = json.load(f)
        for i, row in enumerate(disc):
            img_name = row.get("image") or row.get("image_id", "")
            entries.append({
                "id": f"amber_disc_{i:05d}",
                "image_path": str(out_root / "images" / img_name),
                "text": row.get("query") or row.get("question", ""),
                "label": row.get("label", ""),
                "label_idx": None,
                "task": "discriminative",
                "category": row.get("type", "discriminative"),
                "raw": row,
            })

    gen_path = out_root / "data" / "query" / "query_generative.json"
    if gen_path.exists():
        with open(gen_path) as f:
            gen = json.load(f)
        for i, row in enumerate(gen):
            img_name = row.get("image") or row.get("image_id", "")
            entries.append({
                "id": f"amber_gen_{i:05d}",
                "image_path": str(out_root / "images" / img_name),
                "text": row.get("query") or row.get("question", "Describe this image."),
                "label": row.get("label", ""),
                "task": "generative",
                "category": row.get("type", "generative"),
                "raw": row,
            })

    if not entries:
        raise FileNotFoundError(
            f"No AMBER query files found under {out_root}. "
            "Check the extracted repo layout."
        )

    combined = out_root / "combined.json"
    with open(combined, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} AMBER entries -> {combined}")


if __name__ == "__main__":
    main()
