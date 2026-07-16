#!/usr/bin/env python3
"""Stage 0 — mine v2.1 option-set candidates from COCO train2014 annotations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG.parent))

from vti_demos_v2 import config  # noqa: E402
from vti_demos_v2.geometry import box_from_coco_bbox, evaluate_all_relation_types  # noqa: E402
from vti_demos_v2.io_utils import (  # noqa: E402
    load_exclude_ids_from_demos,
    load_ids,
    write_jsonl,
    write_summary,
)
from src.paths import (  # noqa: E402
    coco_annotations_dir,
    coco_train2014_dir,
    vti_demos_path,
    vti_demos_v2_dir,
)


def _load_instances(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Download COCO 2014 trainval annotations and extract "
            f"instances_train2014.json into {coco_annotations_dir()}."
        )
    print(f"Loading {path} ...")
    with open(path) as f:
        return json.load(f)


def _build_indexes(coco: dict):
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    cat_to_supercat = {c["name"]: c["supercategory"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    return cat_id_to_name, cat_to_supercat, images, anns_by_image


def build_cooccurrence(
    anns_by_image: Dict[int, List[dict]],
    cat_id_to_name: Dict[int, str],
) -> Dict[str, Dict[str, float]]:
    """P(c | p) over train2014 images (category co-presence, ignoring iscrowd)."""
    present_sets: List[Set[str]] = []
    for anns in anns_by_image.values():
        cats = {cat_id_to_name[a["category_id"]] for a in anns}
        if cats:
            present_sets.append(cats)
    count_p = Counter()
    count_cp = Counter()
    for cats in present_sets:
        for p in cats:
            count_p[p] += 1
            for c in cats:
                if c != p:
                    count_cp[(c, p)] += 1
    out: Dict[str, Dict[str, float]] = {}
    all_cats = list(cat_id_to_name.values())
    for c in all_cats:
        out[c] = {}
        for p in all_cats:
            if c == p:
                continue
            denom = count_p[p]
            out[c][p] = (count_cp[(c, p)] / denom) if denom else 0.0
    return out


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    m = len(ys) // 2
    if len(ys) % 2:
        return ys[m]
    return 0.5 * (ys[m - 1] + ys[m])


def _counting_options(
    anns: List[dict],
    cat_id_to_name: Dict[int, str],
    img_area: float,
) -> List[dict]:
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for a in anns:
        by_cat[cat_id_to_name[a["category_id"]]].append(a)
    out = []
    for cat, items in by_cat.items():
        if any(a.get("iscrowd", 0) for a in items):
            continue
        n = len(items)
        if not (config.COUNT_MIN <= n <= config.COUNT_MAX):
            continue
        areas = [(a["bbox"][2] * a["bbox"][3]) / img_area for a in items]
        med = _median(areas)
        if med < config.COUNT_MIN_AREA_FRAC:
            continue
        out.append({"category": cat, "count": n, "median_area_frac": round(med, 6)})
    return sorted(out, key=lambda x: (-x["median_area_frac"], x["category"]))


def _relation_options(
    anns: List[dict],
    cat_id_to_name: Dict[int, str],
    width: float,
    height: float,
    img_area: float,
) -> List[dict]:
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for a in anns:
        if a.get("iscrowd", 0):
            continue
        by_cat[cat_id_to_name[a["category_id"]]].append(a)

    usable: Dict[str, List[dict]] = {}
    for cat, items in by_cat.items():
        if len(items) == 1:
            usable[cat] = items

    out = []
    cats = sorted(usable)
    for a_cat in cats:
        for b_cat in cats:
            if a_cat == b_cat:
                continue
            a, b = usable[a_cat][0], usable[b_cat][0]
            for result in evaluate_all_relation_types(
                box_from_coco_bbox(a["bbox"]), box_from_coco_bbox(b["bbox"]),
                b_category=b_cat, img_w=width, img_h=height, img_area=img_area, cfg=config,
            ):
                out.append({"a": a_cat, "b": b_cat, **result})
    return out


def _distractor_candidates(
    present: Set[str],
    cooc: Dict[str, Dict[str, float]],
    all_cats: List[str],
    top_k: int,
) -> List[dict]:
    absent = [c for c in all_cats if c not in present]
    scored = []
    for c in absent:
        if not present:
            score = 0.0
        else:
            score = sum(cooc.get(c, {}).get(p, 0.0) for p in present) / len(present)
        scored.append((score, c))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [{"category": c, "score": round(score, 8)} for score, c in scored[:top_k]]


def _score_candidate(rec: dict) -> float:
    gap = max(float(x.get("gap_frac", 0.0)) for x in rec["relation_options"])
    area = max(float(x["median_area_frac"]) for x in rec["counting_options"])
    n_cats = len(rec["present_categories"])
    # normalize n_cats roughly into [0,1] with soft cap at 10
    n_norm = min(n_cats / 10.0, 1.0)
    return (
        config.W_REL_GAP * min(gap / 0.3, 1.0)
        + config.W_COUNT_AREA * min(area / 0.05, 1.0)
        + config.W_N_CATS * n_norm
        + config.W_N_GE3 * float(any(x["count"] >= 3 for x in rec["counting_options"]))
        + config.W_REL_TYPE_DIV * min(len({x["type"] for x in rec["relation_options"]}) / 4, 1.0)
    )


def mine_candidates(
    *,
    n_candidates: int,
    exclude: Set[str],
    cooc: Dict[str, Dict[str, float]],
    cat_id_to_name: Dict[int, str],
    images: dict,
    anns_by_image: Dict[int, List[dict]],
    cat_to_supercat: Dict[str, str] | None = None,
) -> List[dict]:
    all_cats = sorted(cat_id_to_name.values())
    rows: List[dict] = []
    for img_id, im in images.items():
        sid = f"{int(img_id):012d}"
        if sid in exclude or str(img_id) in exclude:
            continue
        anns = anns_by_image.get(img_id, [])
        if not anns:
            continue
        width = float(im["width"])
        height = float(im["height"])
        img_area = width * height
        present_counts: Dict[str, int] = Counter(
            cat_id_to_name[a["category_id"]] for a in anns
        )
        present = set(present_counts)
        if len(present) < 2:
            continue
        counting = _counting_options(anns, cat_id_to_name, img_area)
        if not counting:
            continue
        relation = _relation_options(anns, cat_id_to_name, width, height, img_area)
        if not relation:
            continue
        distractors = _distractor_candidates(
            present, cooc, all_cats, config.DISTRACTOR_TOP_K
        )
        if not distractors:
            continue
        rec = {
            "id": sid,
            "image": im["file_name"],
            "present_categories": dict(present_counts),
            "counting_options": counting,
            "relation_options": relation,
            "distractor_candidates": distractors,
        }
        rec["score"] = round(_score_candidate(rec), 6)
        rows.append(rec)
    rows.sort(key=lambda r: (-r["score"], r["id"]))
    cap = max(1, int(n_candidates * config.SUPERCAT_CAP_FRAC))
    emitted, supercats = [], Counter()
    for row in rows:
        # Use the best available counting option's supercategory for a deterministic
        # scene stratum while retaining every option for the allocator.
        supercat = (cat_to_supercat or {}).get(row["counting_options"][0]["category"], "unknown")
        if supercats[supercat] >= cap:
            continue
        emitted.append(row)
        supercats[supercat] += 1
        if len(emitted) == n_candidates:
            break
    return emitted


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-candidates", type=int, default=config.N_CANDIDATES)
    p.add_argument("--exclude-ids-file", type=Path, default=None,
                   help="JSONL with id fields to exclude (default: data/vti/demos.jsonl)")
    p.add_argument("--emit-next-batch", action="store_true",
                   help="Exclude ids already present in any stage artifact; emit next N")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cooccurrence-cache", type=Path, default=None)
    p.add_argument("--instances", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    v2 = vti_demos_v2_dir()
    v2.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (v2 / "stage0_candidates.jsonl")
    cooc_path = args.cooccurrence_cache or (v2 / "cooccurrence.json")
    instances_path = args.instances or (
        coco_annotations_dir() / "instances_train2014.json"
    )

    exclude = load_exclude_ids_from_demos(args.exclude_ids_file or vti_demos_path())
    # also accept unpadded ids
    exclude |= {str(int(x)) for x in exclude if str(x).isdigit()}

    if args.emit_next_batch:
        stage_paths = [
            v2 / "stage0_candidates.jsonl",
            v2 / "stage1_verified.jsonl",
            v2 / "stage1_rejected.jsonl",
            v2 / "stage2_captions.jsonl",
            v2 / "stage2_rejected.jsonl",
            v2 / "stage3_variants.jsonl",
            v2 / "stage4_verdicts.jsonl",
            v2 / "stage4_rejected.jsonl",
        ]
        for pth in stage_paths:
            exclude |= load_ids(pth)
        print(f"--emit-next-batch: excluding {len(exclude)} known ids")

    coco = _load_instances(instances_path)
    cat_id_to_name, cat_to_supercat, images, anns_by_image = _build_indexes(coco)

    if cooc_path.is_file():
        print(f"Loading co-occurrence cache {cooc_path}")
        cooc = json.loads(cooc_path.read_text())
    else:
        print("Building co-occurrence table ...")
        cooc = build_cooccurrence(anns_by_image, cat_id_to_name)
        cooc_path.write_text(json.dumps(cooc))
        print(f"Wrote {cooc_path}")

    rows = mine_candidates(
        n_candidates=args.n_candidates,
        exclude=exclude,
        cooc=cooc,
        cat_id_to_name=cat_id_to_name,
        images=images,
        anns_by_image=anns_by_image,
        cat_to_supercat=cat_to_supercat,
    )
    write_jsonl(out_path, rows)

    cat_hist = Counter(x["category"] for r in rows for x in r["counting_options"])
    summary = {
        "n_candidates": len(rows),
        "out": str(out_path),
        "n_excluded": len(exclude),
        "train2014_dir": str(coco_train2014_dir()),
        "counting_category_histogram": dict(cat_hist.most_common()),
        "score_min": rows[-1]["score"] if rows else None,
        "score_max": rows[0]["score"] if rows else None,
    }
    write_summary(v2 / "stage0_summary.json", summary)
    print(f"Wrote {len(rows)} candidates -> {out_path}")
    print(f"Top counting categories: {cat_hist.most_common(10)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
