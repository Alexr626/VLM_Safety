#!/usr/bin/env python3
"""
Download MM-SafetyBench to data/mm-safetybench/ from the HuggingFace mirror.

Source: PKU-Alignment/MM-SafetyBench. Images are EMBEDDED as bytes inside
parquet files (one parquet per (scenario, image_type)), so no separate image
download is needed.

After this script completes, the loader at
`evaluation/benchmarks/mm_safetybench.py` reads:

    data/mm-safetybench/data/{Scenario}/{SD,SD_TYPO,TYPO,Text_only}.parquet

with columns `id` (int), `question` (str), `image` (bytes-or-None).
"""

from pathlib import Path
import sys


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print("huggingface_hub is required:", e, file=sys.stderr)
        sys.exit(1)

    dest = Path("data/mm-safetybench")
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MM-SafetyBench to {dest} ...")
    snapshot_download(
        repo_id="PKU-Alignment/MM-SafetyBench",
        repo_type="dataset",
        local_dir=str(dest),
    )
    print(f"Done. Images are embedded inside the parquet files at "
          f"{dest}/data/{{scenario}}/*.parquet — no separate image folder needed.")


if __name__ == "__main__":
    main()
