#!/usr/bin/env python3
"""Draw the expanded AMBER discriminative 1500-item pin.

Strict superset of ``data/amber/pinned_amber_disc_450.json``. Per-(question type,
gold) targets match the extraction spec; at most one item per (image, question
text); image-spread fill with seed 1234.

Refuses to write any path other than
``data/amber/pinned_amber_disc_1500.json``.

Usage:
  python data_scripts/draw_amber_discriminative_1500.py --seed 1234
  python data_scripts/draw_amber_discriminative_1500.py --seed 1234 --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import (  # noqa: E402
    _amber_discriminative_qtype,
    _load_amber_annotations,
    benchmark_data_dir,
    combined_json_path,
)

# Spec targets (verbatim).
EXISTENCE_GOLD_NO = 500
ATTRIBUTE_GOLD_YES = 250
ATTRIBUTE_GOLD_NO = 250
RELATION_GOLD_YES = 293
RELATION_GOLD_NO = 207

ALLOWED_OUT = Path("data/amber/pinned_amber_disc_1500.json")
DEFAULT_PINNED_450 = Path("data/amber/pinned_amber_disc_450.json")

# Fill order: scarcest pool_size / items_still_needed first.
BUCKET_ORDER = (
    ("relation", "no", RELATION_GOLD_NO),
    ("relation", "yes", RELATION_GOLD_YES),
    ("existence", "no", EXISTENCE_GOLD_NO),
    ("attribute", "no", ATTRIBUTE_GOLD_NO),
    ("attribute", "yes", ATTRIBUTE_GOLD_YES),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--pinned_450",
        type=Path,
        default=DEFAULT_PINNED_450,
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ALLOWED_OUT,
        help="Must be data/amber/pinned_amber_disc_1500.json",
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    out = args.out
    if out.resolve() != (_PROJECT_ROOT / ALLOWED_OUT).resolve() and out != ALLOWED_OUT:
        # Allow either relative-from-root or absolute pointing at the same file.
        allowed_abs = (_PROJECT_ROOT / ALLOWED_OUT).resolve()
        if Path(out).resolve() != allowed_abs:
            raise SystemExit(
                f"Refusing to write {out}; only allowed path is {ALLOWED_OUT}"
            )
    out_abs = (_PROJECT_ROOT / ALLOWED_OUT).resolve()
    if out_abs.exists() and not args.force:
        raise SystemExit(f"[exists] {out_abs} (use --force to overwrite)")

    pinned_450_path = (
        args.pinned_450
        if args.pinned_450.is_absolute()
        else _PROJECT_ROOT / args.pinned_450
    )
    pinned_450 = json.loads(pinned_450_path.read_text())
    pinned_ids = list(pinned_450["amber"])
    pinned_set = set(pinned_ids)
    if len(pinned_ids) != 450 or len(pinned_set) != 450:
        raise SystemExit(
            f"Expected 450 unique pinned ids, got n={len(pinned_ids)} "
            f"unique={len(pinned_set)}"
        )

    root = benchmark_data_dir("amber")
    annotations = _load_amber_annotations(root)
    entries = json.load(open(combined_json_path("amber")))

    # Build eligible pool: discriminative, annotated, deduped by (image, text).
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        if e.get("task", "discriminative") != "discriminative":
            continue
        raw = e.get("raw") or {}
        rid = raw.get("id")
        ann = annotations.get(rid)
        if ann is None:
            continue
        image = e.get("image_path") or e.get("image") or ""
        text = e.get("text") or ""
        by_key[(image, text)].append(e)

    # Dedupe: prefer pinned member, else lowest-sorted id.
    kept: list[dict] = []
    for group in by_key.values():
        ids = [g["id"] for g in group]
        pinned_members = [g for g in group if g["id"] in pinned_set]
        if len(pinned_members) > 1:
            raise SystemExit(
                f"Pinned 450 contains >1 member of a duplicate prompt group: "
                f"{[g['id'] for g in pinned_members]}"
            )
        if pinned_members:
            kept.append(pinned_members[0])
        else:
            kept.append(sorted(group, key=lambda g: g["id"])[0])

    # Index by (qtype, gold) and by id.
    by_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_id: dict[str, dict] = {}
    for e in kept:
        raw = e.get("raw") or {}
        ann = annotations[raw["id"]]
        qtype = _amber_discriminative_qtype(ann.get("type", ""))
        gold = (ann.get("truth") or "").strip().lower()
        image = e.get("image_path") or e.get("image") or ""
        rec = {
            "id": e["id"],
            "image": image,
            "text": e.get("text") or "",
            "qtype": qtype,
            "gold": gold,
        }
        by_id[e["id"]] = rec
        by_bucket[(qtype, gold)].append(rec)

    missing_pinned = [pid for pid in pinned_ids if pid not in by_id]
    if missing_pinned:
        raise SystemExit(
            f"{len(missing_pinned)} pinned-450 ids missing from eligible pool "
            f"(first: {missing_pinned[:5]})"
        )

    # Seed selection with all 450; initialise per-image usage from them.
    selected: list[str] = list(pinned_ids)
    selected_set = set(selected)
    image_usage: Counter = Counter(by_id[pid]["image"] for pid in selected)

    # Count how many pinned items already fill each bucket.
    pinned_bucket_counts: Counter = Counter(
        (by_id[pid]["qtype"], by_id[pid]["gold"]) for pid in selected
    )

    rng = random.Random(args.seed)
    for qtype, gold, target in BUCKET_ORDER:
        have = pinned_bucket_counts[(qtype, gold)]
        need = target - have
        if need < 0:
            raise SystemExit(
                f"Pinned 450 already exceeds target for {(qtype, gold)}: "
                f"have={have} target={target}"
            )
        if need == 0:
            continue
        candidates = [
            rec
            for rec in by_bucket[(qtype, gold)]
            if rec["id"] not in selected_set
        ]
        # Shuffle once with seeded RNG, then fill by increasing image-usage
        # level. Within a pass, re-check usage after every pick so a second
        # item on an image just used is deferred to the next pass.
        rng.shuffle(candidates)
        picked = 0
        usage_level = 0
        while picked < need:
            made_progress = False
            for rec in candidates:
                if picked >= need:
                    break
                if rec["id"] in selected_set:
                    continue
                if image_usage[rec["image"]] != usage_level:
                    continue
                selected.append(rec["id"])
                selected_set.add(rec["id"])
                image_usage[rec["image"]] += 1
                picked += 1
                made_progress = True
            if picked >= need:
                break
            if not made_progress:
                usage_level += 1
                if usage_level > 20:
                    raise SystemExit(
                        f"Could not fill bucket {(qtype, gold)}: "
                        f"picked={picked} need={need}"
                    )

    if len(selected) != 1500 or len(selected_set) != 1500:
        raise SystemExit(
            f"Expected 1500 unique ids, got n={len(selected)} "
            f"unique={len(selected_set)}"
        )
    if not pinned_set.issubset(selected_set):
        raise SystemExit("Superset check failed: pinned 450 not fully contained")

    # Realised composition.
    gold_by_qtype: dict[str, dict[str, int]] = defaultdict(lambda: {"yes": 0, "no": 0})
    strata: dict[str, dict] = {}
    for qtype in ("existence", "attribute", "relation"):
        avail = sum(
            len(by_bucket[(qtype, g)]) for g in ("yes", "no")
        )
        drawn = sum(
            1 for sid in selected if by_id[sid]["qtype"] == qtype
        )
        strata[qtype] = {"available": avail, "drawn": drawn}
    for sid in selected:
        rec = by_id[sid]
        gold_by_qtype[rec["qtype"]][rec["gold"]] += 1
    gold_counts = Counter(by_id[sid]["gold"] for sid in selected)
    hist = Counter(image_usage.values())
    n_images = len(image_usage)

    # Verify exact targets.
    expected = {
        ("existence", "no"): EXISTENCE_GOLD_NO,
        ("existence", "yes"): 0,
        ("attribute", "yes"): ATTRIBUTE_GOLD_YES,
        ("attribute", "no"): ATTRIBUTE_GOLD_NO,
        ("relation", "yes"): RELATION_GOLD_YES,
        ("relation", "no"): RELATION_GOLD_NO,
    }
    for (qtype, gold), exp in expected.items():
        got = gold_by_qtype[qtype].get(gold, 0)
        if got != exp:
            raise SystemExit(
                f"Target miss for {(qtype, gold)}: got={got} expected={exp}"
            )

    payload = {
        "amber": selected,
        "_meta": {
            "benchmark": "amber",
            "task": "discriminative",
            "seed": args.seed,
            "n": 1500,
            "superset_of": {
                "path": str(pinned_450_path.relative_to(_PROJECT_ROOT))
                if pinned_450_path.is_relative_to(_PROJECT_ROOT)
                else str(pinned_450_path),
                "sha256": _sha256(pinned_450_path),
                "n": 450,
            },
            "dedupe_policy": (
                "at most one item per (image, question text); "
                "pinned member preferred, else lowest id"
            ),
            "strata": strata,
            "gold_counts_by_question_type": {
                q: dict(gold_by_qtype[q]) for q in ("existence", "attribute", "relation")
            },
            "gold_counts": dict(gold_counts),
            "n_distinct_images": n_images,
            "items_per_image_histogram": {
                str(k): hist[k] for k in sorted(hist)
            },
            "source": (
                "data/amber/combined.json + data/amber/data/annotations.json; "
                "superset of pinned_amber_disc_450.json"
            ),
            "drawn_at": datetime.now().isoformat(timespec="seconds"),
        },
    }

    out_abs.parent.mkdir(parents=True, exist_ok=True)
    out_abs.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"[wrote] {out_abs}")
    print(f"  n={len(selected)}  distinct_images={n_images}")
    print(f"  gold_counts={dict(gold_counts)}")
    print(f"  gold_by_qtype={ {q: dict(gold_by_qtype[q]) for q in gold_by_qtype} }")
    print(f"  items_per_image_histogram={dict(sorted(hist.items()))}")
    print(f"  superset_of_450=True  pinned_sha256={_sha256(pinned_450_path)[:16]}…")


if __name__ == "__main__":
    main()
