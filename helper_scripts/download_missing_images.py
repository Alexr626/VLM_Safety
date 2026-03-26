#!/usr/bin/env python3
"""
Download missing HoliSafe-Bench images one by one with retry and timeout.

Reads holisafe_bench.json to determine which files are needed,
skips files that already exist locally, and retries failed downloads.

Usage:
    source /etc/network_turbo
    python download_missing_images.py

    HF_ENDPOINT=https://hf-mirror.com python download_missing_images.py

    # Re-download even files that already exist
    python download_missing_images.py --force
"""
import argparse
import json
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
HOLISAFE_DIR = PROJECT_ROOT / "data" / "holisafe-bench"
REPO_ID      = "etri-vilab/holisafe-bench"
MAX_RETRIES  = 3
RETRY_DELAY  = 5   # seconds between retries


def get_all_image_paths(json_path: Path) -> list[str]:
    """Return deduplicated list of image paths from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    seen, result = set(), []
    for entry in data:
        img = entry.get("image", "")
        if img and img not in seen:
            seen.add(img)
            result.append(img)
    return result


def download_one(img_rel: str, force: bool) -> bool:
    """
    Download a single image. Returns True on success, False on failure.
    Skips if the file already exists and force=False.
    """
    local_path = HOLISAFE_DIR / "images" / img_rel
    if not force and local_path.exists():
        return True

    local_path.parent.mkdir(parents=True, exist_ok=True)
    repo_filename = f"images/{img_rel}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_filename,
                repo_type="dataset",
                token=True,
                local_dir=str(HOLISAFE_DIR),
            )
            return True
        except KeyboardInterrupt:
            raise
        except Exception as e:
            msg = str(e)[:120]
            tqdm.write(f"  [attempt {attempt}/{MAX_RETRIES}] FAILED: {repo_filename} — {msg}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-download files even if they already exist locally")
    args = parser.parse_args()

    json_path = HOLISAFE_DIR / "holisafe_bench.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found.")
        sys.exit(1)

    all_images = get_all_image_paths(json_path)

    if args.force:
        to_download = all_images
    else:
        to_download = [
            img for img in all_images
            if not (HOLISAFE_DIR / "images" / img).exists()
        ]

    print(f"Total unique images in JSON : {len(all_images)}")
    print(f"To download                 : {len(to_download)}")

    if not to_download:
        print("Nothing to download.")
        sys.exit(0)

    failed = []
    with tqdm(to_download, desc="Downloading", unit="file") as pbar:
        for img_rel in pbar:
            pbar.set_postfix_str(Path(img_rel).name[:30])
            ok = download_one(img_rel, force=args.force)
            if not ok:
                failed.append(img_rel)

    print(f"\n{'='*55}")
    print(f"  Succeeded : {len(to_download) - len(failed)}/{len(to_download)}")
    print(f"  Failed    : {len(failed)}")

    if failed:
        failed_log = HOLISAFE_DIR / "download_failed.txt"
        with open(failed_log, "w") as f:
            f.write("\n".join(failed))
        print(f"  Failed list saved to: {failed_log}")
        print("\n  Run check_data_integrity.py to see remaining gaps.")
        sys.exit(1)
    else:
        print("\n  All images downloaded. Run check_data_integrity.py to verify.")
        sys.exit(0)


if __name__ == "__main__":
    main()
