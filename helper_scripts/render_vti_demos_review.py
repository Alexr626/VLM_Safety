#!/usr/bin/env python3
"""Build a self-contained HTML gallery for VTI paired-caption demos.

Shows each COCO train2014 image alongside the truthful caption (``value``) and
the hallucinated caption (``h_value``) from ``data/vti/demos.jsonl``, plus
``co_objects`` / ``uncertain_objects`` metadata from the authors' bundle.

Output (default)::

    data/vti/_review/vti_demos_review.html

Usage::

    python helper_scripts/render_vti_demos_review.py
    python helper_scripts/render_vti_demos_review.py --open
    python helper_scripts/render_vti_demos_review.py --num-demos 20 --seed 42
    python helper_scripts/render_vti_demos_review.py --ids 000000103108 000000504235
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "helper_scripts"))

from review_lib import (  # noqa: E402
    GalleryPanel,
    ResponseSet,
    open_in_os,
    write_gallery_html,
)
from src.paths import coco_train2014_dir, vti_data_dir, vti_demos_path  # noqa: E402


def _load_demos(path: Path, num_demos: int, seed: int) -> list[dict]:
    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    if num_demos < len(data):
        data = random.Random(seed).sample(data, num_demos)
    return data


def _default_html_out() -> Path:
    return vti_data_dir() / "_review" / "vti_demos_review.html"


def _demo_meta(demo: dict, image_root: Path) -> dict:
    co = demo.get("co_objects") or []
    uncertain = demo.get("uncertain_objects") or []
    return {
        "image_path": str(image_root / demo["image"]),
        "text": demo.get("question", "Describe this image in detail."),
        "label": (
            f"co_objects: {', '.join(co) if co else '—'}"
            f" · uncertain: {', '.join(uncertain) if uncertain else '—'}"
        ),
    }


def _demo_response_sets(demo: dict, source: Path) -> list[ResponseSet]:
    rs = ResponseSet("VTI paired captions", source)
    rs.question = demo.get("question")
    rs.add("truthful caption (value)", demo.get("value", ""))
    rs.add("hallucinated caption (h_value)", demo.get("h_value", ""))
    return [rs]


def _filter_ids(demos: list[dict], ids: list[str]) -> list[dict]:
    if not ids:
        return demos
    by_id = {d["id"]: d for d in demos}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"demo id(s) not found in demos.jsonl: {', '.join(missing)}")
    return [by_id[i] for i in ids]


def _build_panels(
    demos: list[dict],
    image_root: Path,
    source: Path,
) -> list[GalleryPanel]:
    panels: list[GalleryPanel] = []
    for demo in demos:
        sid = demo["id"]
        meta = _demo_meta(demo, image_root)
        resp_sets = _demo_response_sets(demo, source)
        panels.append(GalleryPanel(sid, "vti_demos", meta, resp_sets))
    return panels


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--demos-file",
        type=Path,
        default=None,
        help=f"path to demos JSONL (default: {vti_demos_path()})",
    )
    p.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=f"COCO train2014 directory (default: {coco_train2014_dir()})",
    )
    p.add_argument(
        "--num-demos",
        type=int,
        default=0,
        help="subsample N demos (0 = all; uses same seed logic as direction extraction)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --num-demos subsample")
    p.add_argument("--ids", nargs="+", default=[], help="include only these demo ids (COCO id)")
    p.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="output HTML path (default: data/vti/_review/vti_demos_review.html)",
    )
    p.add_argument("--open", action="store_true", help="open the gallery in a browser")
    p.add_argument("--no-open", action="store_true", help="write HTML but do not open it")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    demos_path = args.demos_file or vti_demos_path()
    image_root = args.image_root or coco_train2014_dir()
    out_path = args.html_out or _default_html_out()

    if not demos_path.is_file():
        raise SystemExit(
            f"VTI demos file not found: {demos_path}\n"
            "Expected data/vti/demos.jsonl in the repo."
        )
    if not image_root.is_dir():
        raise SystemExit(
            f"COCO train2014 directory not found: {image_root}\n"
            "Run: python data_scripts/download_chair.py --with-train2014"
        )

    num_demos = args.num_demos if args.num_demos > 0 else 10**9
    demos = _load_demos(demos_path, num_demos=num_demos, seed=args.seed)
    demos = _filter_ids(demos, args.ids)

    missing_images = [
        d["id"] for d in demos
        if not (image_root / d["image"]).is_file()
    ]
    if missing_images:
        print(
            f"[warning] {len(missing_images)} demo image(s) missing under {image_root}",
            file=sys.stderr,
        )

    panels = _build_panels(demos, image_root, demos_path)
    title = "VTI demo captions review"
    subtitle = (
        f"{len(panels)} demo(s) · {demos_path.name} · images from {image_root.name}/"
    )
    write_gallery_html(title, subtitle, panels, out_path)
    print(f"Wrote {out_path} ({len(panels)} demos)")

    if args.open and not args.no_open:
        open_in_os(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
