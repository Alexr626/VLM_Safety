#!/usr/bin/env python3
"""Download AMBER benchmark and build combined.json."""

import json
import sys
import zipfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

AMBER_REPO = "https://github.com/junyangwang0410/AMBER/archive/refs/heads/main.zip"


def main():
    out_root = _PROJECT_ROOT / "data" / "amber"
    out_root.mkdir(parents=True, exist_ok=True)
    zip_path = out_root / "amber_main.zip"

    if not (out_root / "images").exists():
        import urllib.request
        print(f"Downloading AMBER from {AMBER_REPO} ...")
        urllib.request.urlretrieve(AMBER_REPO, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_root)
        extracted = next(out_root.glob("AMBER-*"), None)
        if extracted:
            for child in extracted.iterdir():
                dest = out_root / child.name
                if dest.exists():
                    continue
                child.rename(dest)

    entries = []
    # Discriminative: data/query/query_discriminative.json
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

    # Generative: data/query/query_generative.json
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
