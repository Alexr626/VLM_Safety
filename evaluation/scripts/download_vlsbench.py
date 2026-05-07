#!/usr/bin/env python3
"""
Download VLSBench to data/vlsbench/ from HuggingFace.

Source: Foreshhh/vlsbench — 2,241 samples evaluating information leakage in
MLLMs by pairing unsafe images with deceptive instructions.

Layout after download:

    data/vlsbench/
    ├── data.json          (metadata: instruction_id, instruction, image_path,
    │                       category, sub_category, source, image_description,
    │                       safety_reason)
    └── imgs/              (extracted from imgs.tar: 0.png .. 2240.png)

Categories (6): Illegal Activity, Violent, Hate, Privacy, Self-Harm, Erotic
Subcategories: 19 total
Sources: generation, MultiTrust, MLLMGuard, Ch3Ef, Unsafebench, coco2017

GitHub: https://github.com/ai45lab/vlsbench
Paper: https://arxiv.org/abs/2411.19939
"""

from pathlib import Path
import sys


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print("huggingface_hub is required:", e, file=sys.stderr)
        sys.exit(1)

    dest = Path("data/vlsbench")
    dest.mkdir(parents=True, exist_ok=True)

    # Download data.json (metadata, fast)
    json_path = dest / "data.json"
    if not json_path.exists():
        print("Downloading data.json ...")
        downloaded = hf_hub_download(
            "Foreshhh/vlsbench", "data.json", repo_type="dataset",
            local_dir=str(dest),
        )
        print(f"  -> {downloaded}")
    else:
        print(f"data.json already exists at {json_path}")

    # Download imgs.tar (all images in one file, much faster than 1968 individual files)
    imgs_dir = dest / "imgs"
    tar_path = dest / "imgs.tar"
    if not imgs_dir.exists() or len(list(imgs_dir.glob("*.png"))) < 2000:
        print("Downloading imgs.tar (~images archive) ...")
        hf_hub_download(
            "Foreshhh/vlsbench", "imgs.tar", repo_type="dataset",
            local_dir=str(dest),
        )
        # Extract
        import tarfile
        print("Extracting imgs.tar ...")
        with tarfile.open(str(tar_path), "r") as tf:
            tf.extractall(path=str(dest))
        n_imgs = len(list(imgs_dir.glob("*.png")))
        print(f"  Extracted {n_imgs} images to {imgs_dir}/")
    else:
        print(f"imgs/ already populated ({len(list(imgs_dir.glob('*.png')))} images)")

    print(f"\nDone. Dataset at {dest}/")
    print(f"  Metadata: {json_path}")
    print(f"  Images:   {imgs_dir}/")


if __name__ == "__main__":
    main()
