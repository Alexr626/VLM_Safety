#!/usr/bin/env python3
"""
Download SIUO (Safe Inputs but Unsafe Outputs) to data/siuo/ from HuggingFace.

Source: sinwang/SIUO — 168 human-crafted test cases where individually safe
image + text inputs combine to produce unsafe outputs (cross-modality safety).

Layout after download:

    data/siuo/
    ├── data/train-00000-of-00001.parquet   (HF-hosted images + metadata)
    └── ...

The HuggingFace version has an `image` column with embedded image data.
The full annotation JSONs (siuo_gen.json, siuo_mcqa.json) with questions,
categories, and responses are available in the GitHub repo.

9 safety domains: Self-Harm, Dangerous Behavior, Morality, Illegal Activities
& Crime, Controversial Topics & Politics, Discrimination & Stereotyping,
Religion Beliefs, Information Misinterpretation, Privacy Violation.
33 subcategories total.

GitHub: https://github.com/sinwang20/SIUO
Paper: https://arxiv.org/abs/2406.15279
License: CC-BY-NC-4.0
"""

from pathlib import Path
import sys


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print("huggingface_hub is required:", e, file=sys.stderr)
        sys.exit(1)

    dest = Path("data/siuo")
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading SIUO (sinwang/SIUO) to {dest} ...")
    snapshot_download(
        repo_id="sinwang/SIUO",
        repo_type="dataset",
        local_dir=str(dest),
    )

    # Also clone the GitHub repo for the full annotation JSONs
    # (siuo_gen.json, siuo_mcqa.json) which have the actual questions
    github_dir = dest / "github"
    if not github_dir.exists():
        import subprocess
        print("Cloning GitHub repo for annotation JSONs ...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/sinwang20/SIUO.git",
             str(github_dir)],
            check=True,
        )
        print(f"GitHub repo cloned to {github_dir}/")
    else:
        print(f"GitHub repo already at {github_dir}/, skipping clone.")

    print(f"\nDone. Key files:")
    print(f"  HF data:    {dest}/data/")
    print(f"  Annotations: {github_dir}/data/siuo_gen.json")
    print(f"  MCQA:        {github_dir}/data/siuo_mcqa.json")
    print(f"\nNote: Full-resolution images require separate download from")
    print(f"  Google Drive (see {github_dir}/README.md for links).")


if __name__ == "__main__":
    main()
