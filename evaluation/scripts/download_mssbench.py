#!/usr/bin/env python3
"""
Download MSSBench (Multimodal Situational Safety Benchmark) to data/mssbench/.

Source: kzhou35/mssbench on HuggingFace (the official mirror, per the paper's
project page mssbench.github.io). Layout after download:

    data/mssbench/
    ├── combined.json    (record list with safe/unsafe image paths + queries)
    ├── chat/{N}.jpg     (image files referenced by combined.json)
    └── embodied/...     (separate embodied-agent split; not used by default)
"""

from pathlib import Path
import sys


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print("huggingface_hub is required:", e, file=sys.stderr)
        sys.exit(1)

    dest = Path("data/mssbench")
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MSSBench (kzhou35/mssbench) to {dest} ...")
    snapshot_download(
        repo_id="kzhou35/mssbench",
        repo_type="dataset",
        local_dir=str(dest),
    )
    print("Done.")


if __name__ == "__main__":
    main()
