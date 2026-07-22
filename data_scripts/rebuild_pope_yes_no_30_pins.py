#!/usr/bin/env python3
"""Rebuild POPE-30-yes and POPE-30-no pinned subsets under data/pope/.

POPE-30-yes: first 30 gold=yes from the random split in pinned_eval_ids.json
order (replaces the 2026-07-19 triplicated 10×3 pin in place).

POPE-30-no: 10 gold=no per split (random/popular/adversarial), restricted to
images in the yes set, skipping (image_id, questioned-object) collisions
across splits.

See implementation_plans/7-21-26/pope_yes_no_30_dataset_rebuild_plan_2026-07-22.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import pope_data_dir, project_root  # noqa: E402

SPLITS = ("random", "popular", "adversarial")
OBJECT_RE = re.compile(r"^Is there an? (.+) in the image\?\s*$", re.IGNORECASE)
SUPERSEDES_NOTE = (
    "2026-07-19 triplicated version; ids archived in "
    "pope_yes_no_30_dataset_rebuild_plan_2026-07-22.md"
)


def _load_combined() -> dict[str, dict]:
    path = pope_data_dir() / "combined.json"
    rows = json.loads(path.read_text())
    return {r["id"]: r for r in rows}


def _load_eval_ids() -> dict:
    path = pope_data_dir() / "pinned_eval_ids.json"
    return json.loads(path.read_text())


def _label(row: dict) -> str:
    return str(row.get("label") or "").strip().lower()


def _image_id(row: dict) -> str:
    raw = row.get("raw") or {}
    img = raw.get("image") or row.get("image_path") or ""
    return Path(str(img)).name


def _question(row: dict) -> str:
    return str(row.get("text") or (row.get("raw") or {}).get("text") or "")


def _questioned_object(question: str) -> str:
    m = OBJECT_RE.match(question.strip())
    if not m:
        raise ValueError(f"Cannot parse questioned object from: {question!r}")
    return m.group(1).strip().lower()


def _content_hash(items: list[dict]) -> str:
    blob = json.dumps(items, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def _item_record(row: dict, *, include_split: bool) -> dict:
    rec = {
        "id": row["id"],
        "image_id": _image_id(row),
        "question": _question(row),
        "gold": _label(row),
    }
    if include_split:
        split = row.get("task") or row.get("category")
        rec["split"] = split
    return rec


def build_yes_pin(by_id: dict[str, dict], eval_ids: dict) -> dict:
    selection = (
        "first 30 gold=yes items from the RANDOM split, in "
        "data/pope/pinned_eval_ids.json order (the random split is the "
        "canonical source since yes-questions are identical across splits)"
    )
    items: list[dict] = []
    for iid in eval_ids["splits"]["random"]:
        row = by_id[iid]
        if _label(row) != "yes":
            continue
        items.append(_item_record(row, include_split=False))
        if len(items) >= 30:
            break

    assert len(items) == 30, f"POPE-30-yes: expected n=30, got {len(items)}"
    assert all(it["gold"] == "yes" for it in items), "POPE-30-yes: non-yes gold"
    pairs = [(it["image_id"], it["question"]) for it in items]
    assert len(set(pairs)) == 30, "POPE-30-yes: duplicate (image_id, question)"
    n_images = len({it["image_id"] for it in items})
    if not (10 <= n_images <= 12):
        raise SystemExit(
            f"POPE-30-yes: distinct image count={n_images} outside expected "
            f"~10–12 (~3 yes-questions per image); stopping."
        )

    ids = [it["id"] for it in items]
    assert all(i.startswith("pope_random_") for i in ids)

    return {
        "benchmark": "pope",
        "n": 30,
        "generated": date.today().isoformat(),
        "selection": selection,
        "supersedes": SUPERSEDES_NOTE,
        "gold_filter": "yes",
        "protocol_name": "POPE-30-yes",
        "ids": ids,
        "items": items,
        "content_hash": _content_hash(items),
        "n_distinct_images": n_images,
    }


def build_no_pin(
    by_id: dict[str, dict],
    eval_ids: dict,
    yes_images: set[str],
) -> dict:
    selection = (
        "10 gold=no items from EACH split (random, popular, adversarial), "
        "restricted to images in the POPE-30-yes image set, taken in "
        "data/pope/pinned_eval_ids.json order within each split; if a "
        "(image_id, questioned-object) pair collides with one already "
        "selected from another split, skip and take the next candidate"
    )
    splits: dict[str, list[str]] = {s: [] for s in SPLITS}
    items: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    collisions_skipped: list[dict] = []

    for split in SPLITS:
        for iid in eval_ids["splits"][split]:
            row = by_id[iid]
            if _label(row) != "no":
                continue
            img = _image_id(row)
            if img not in yes_images:
                continue
            q = _question(row)
            obj = _questioned_object(q)
            key = (img, obj)
            if key in seen_pairs:
                collisions_skipped.append(
                    {"id": iid, "image_id": img, "object": obj, "split": split}
                )
                continue
            seen_pairs.add(key)
            rec = _item_record(row, include_split=True)
            rec["split"] = split
            items.append(rec)
            splits[split].append(iid)
            if len(splits[split]) >= 10:
                break
        if len(splits[split]) != 10:
            raise SystemExit(
                f"POPE-30-no: split={split} got {len(splits[split])} "
                f"(need 10) after image-match + collision filter"
            )

    assert len(items) == 30, f"POPE-30-no: expected n=30, got {len(items)}"
    assert all(it["gold"] == "no" for it in items), "POPE-30-no: non-no gold"
    pairs = [(it["image_id"], it["question"]) for it in items]
    assert len(set(pairs)) == 30, "POPE-30-no: duplicate (image_id, question)"
    no_images = {it["image_id"] for it in items}
    assert no_images <= yes_images, "POPE-30-no: image not in yes image set"
    assert all(len(splits[s]) == 10 for s in SPLITS)

    ids = [it["id"] for it in items]
    return {
        "benchmark": "pope",
        "n": 30,
        "generated": date.today().isoformat(),
        "selection": selection,
        "gold_filter": "no",
        "per_split": 10,
        "protocol_name": "POPE-30-no",
        "image_matched_to": "POPE-30-yes",
        "splits": splits,
        "ids": ids,
        "items": items,
        "content_hash": _content_hash(items),
        "n_distinct_images": len(no_images),
        "collisions_skipped": collisions_skipped,
    }


def _verify_and_print(yes_pin: dict, no_pin: dict) -> None:
    yes_pairs = {(it["image_id"], it["question"]) for it in yes_pin["items"]}
    no_pairs = {(it["image_id"], it["question"]) for it in no_pin["items"]}
    yes_images = {it["image_id"] for it in yes_pin["items"]}
    no_images = {it["image_id"] for it in no_pin["items"]}
    gold_yes = Counter(it["gold"] for it in yes_pin["items"])
    gold_no = Counter(it["gold"] for it in no_pin["items"])
    per_split = Counter(it["split"] for it in no_pin["items"])

    print("\n=== Verification ===")
    print(
        f"{'pin':<14} {'n':>3} {'gold':<12} {'uniq_(img,q)':>12} "
        f"{'n_images':>8} {'content_hash'}"
    )
    print(
        f"{'POPE-30-yes':<14} {yes_pin['n']:>3} {dict(gold_yes)!s:<12} "
        f"{len(yes_pairs):>12} {len(yes_images):>8} {yes_pin['content_hash']}"
    )
    print(
        f"{'POPE-30-no':<14} {no_pin['n']:>3} {dict(gold_no)!s:<12} "
        f"{len(no_pairs):>12} {len(no_images):>8} {no_pin['content_hash']}"
    )
    print(f"image overlap (yes ∩ no): {len(yes_images & no_images)}")
    print(f"no-set images ⊆ yes-set: {no_images <= yes_images}")
    print(f"no-set per-split counts: {dict(per_split)}")
    print("\nno-set objects by split:")
    for split in SPLITS:
        objs = [
            _questioned_object(it["question"])
            for it in no_pin["items"]
            if it["split"] == split
        ]
        print(f"  {split}: {objs}")
    print("\nfirst 3 yes questions:")
    for it in yes_pin["items"][:3]:
        print(f"  {it['id']}: {it['question']}  [{it['image_id']}]")
    print("first 3 no questions:")
    for it in no_pin["items"][:3]:
        print(f"  {it['id']}: {it['question']}  [{it['image_id']}]")
    if no_pin.get("collisions_skipped"):
        print(f"\ncollisions skipped: {len(no_pin['collisions_skipped'])}")
        for c in no_pin["collisions_skipped"]:
            print(f"  {c}")


def main() -> None:
    by_id = _load_combined()
    eval_ids = _load_eval_ids()
    # pinned_eval_ids lists both yes and no in interleaved order per split;
    # gold filtering uses combined.json labels (open-question fallback unused).
    yes_pin = build_yes_pin(by_id, eval_ids)
    yes_images = {it["image_id"] for it in yes_pin["items"]}
    no_pin = build_no_pin(by_id, eval_ids, yes_images)

    out_yes = pope_data_dir() / "pinned_pope_existence_yes_30.json"
    out_no = pope_data_dir() / "pinned_pope_existence_no_30.json"
    # Drop helper collision list from on-disk pin (kept in verify print only).
    no_pin_disk = {k: v for k, v in no_pin.items() if k != "collisions_skipped"}

    out_yes.write_text(json.dumps(yes_pin, indent=2, ensure_ascii=False) + "\n")
    out_no.write_text(json.dumps(no_pin_disk, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {out_yes}")
    print(f"Wrote {out_no}")
    _verify_and_print(yes_pin, no_pin)
    print(f"\nproject_root={project_root()}")


if __name__ == "__main__":
    main()
