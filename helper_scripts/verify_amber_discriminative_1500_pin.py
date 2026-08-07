#!/usr/bin/env python3
"""Gating verification for data/amber/pinned_amber_disc_1500.json.

Writes amber_1500_pin_verification.json under the analysis directory.
All listed checks are gating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import load_amber  # noqa: E402

PIN_PATH = _PROJECT_ROOT / "data/amber/pinned_amber_disc_1500.json"
PIN_450_PATH = _PROJECT_ROOT / "data/amber/pinned_amber_disc_450.json"

EXPECTED_BUCKETS = {
    ("existence", "no"): 500,
    ("existence", "yes"): 0,
    ("attribute", "yes"): 250,
    ("attribute", "no"): 250,
    ("relation", "yes"): 293,
    ("relation", "no"): 207,
}
EXPECTED_HIST = {1: 515, 2: 482, 3: 7}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pin",
        type=Path,
        default=PIN_PATH,
    )
    p.add_argument(
        "--pinned_450",
        type=Path,
        default=PIN_450_PATH,
    )
    p.add_argument(
        "--out",
        type=Path,
        default=(
            _PROJECT_ROOT
            / "evaluation/results/2026-08-05"
            / "_analysis_steering_vector_validation_continuation"
            / "amber_1500_pin_verification.json"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pin = json.loads(args.pin.read_text())
    ids = pin["amber"]
    pin_450 = json.loads(args.pinned_450.read_text())
    ids_450 = set(pin_450["amber"])

    checks: list[dict] = []

    samples = load_amber(task="discriminative", subset_ids=set(ids))
    checks.append({
        "name": "load_amber returns exactly 1500 samples",
        "pass": len(samples) == 1500,
        "detail": f"n={len(samples)}",
    })
    null_pil = [s["id"] for s in samples if s.get("image_pil") is None]
    checks.append({
        "name": "every sample has non-null image_pil",
        "pass": len(null_pil) == 0,
        "detail": f"n_null={len(null_pil)} first={null_pil[:3]}",
    })
    checks.append({
        "name": "pinned 450 ids all present in 1500",
        "pass": ids_450.issubset(set(ids)),
        "detail": f"missing={len(ids_450 - set(ids))}",
    })

    bucket = Counter((s["category"], (s["label"] or "").lower()) for s in samples)
    bucket_ok = all(bucket.get(k, 0) == v for k, v in EXPECTED_BUCKETS.items())
    checks.append({
        "name": "per (question type, gold) counts exact",
        "pass": bucket_ok,
        "detail": {f"{q}|{g}": bucket.get((q, g), 0) for q, g in EXPECTED_BUCKETS},
    })
    yes = sum(1 for s in samples if (s["label"] or "").lower() == "yes")
    no = sum(1 for s in samples if (s["label"] or "").lower() == "no")
    checks.append({
        "name": "overall gold 543 yes / 957 no",
        "pass": yes == 543 and no == 957,
        "detail": f"yes={yes} no={no}",
    })

    images = [Path(s["image_path"]).name if s.get("image_path") else "" for s in samples]
    n_distinct = len(set(images))
    checks.append({
        "name": "1004 distinct images",
        "pass": n_distinct == 1004,
        "detail": f"n_distinct={n_distinct}",
    })

    key_counts = Counter(
        (Path(s["image_path"]).name if s.get("image_path") else "", s["text"])
        for s in samples
    )
    dup_keys = [k for k, c in key_counts.items() if c > 1]
    checks.append({
        "name": "no duplicate (image, question text) pairs",
        "pass": len(dup_keys) == 0,
        "detail": f"n_dup_keys={len(dup_keys)}",
    })

    img_hist = Counter(Counter(images).values())
    hist_ok = all(img_hist.get(k, 0) == v for k, v in EXPECTED_HIST.items())
    checks.append({
        "name": "items-per-image histogram 515/482/7",
        "pass": hist_ok,
        "detail": dict(sorted(img_hist.items())),
    })

    # Every image used three times must already be triple in the 450.
    usage_1500 = Counter(images)
    samples_450 = load_amber(task="discriminative", subset_ids=ids_450)
    images_450 = [
        Path(s["image_path"]).name if s.get("image_path") else ""
        for s in samples_450
    ]
    usage_450 = Counter(images_450)
    triples_1500 = {img for img, c in usage_1500.items() if c == 3}
    triples_450 = {img for img, c in usage_450.items() if c == 3}
    checks.append({
        "name": "every triple-used image was already triple in the 450",
        "pass": triples_1500.issubset(triples_450),
        "detail": {
            "n_triples_1500": len(triples_1500),
            "n_not_in_450_triples": len(triples_1500 - triples_450),
        },
    })

    pin_sha = _sha256(args.pin)
    all_pass = all(c["pass"] for c in checks)
    report = {
        "pin_path": str(args.pin),
        "pin_sha256": pin_sha,
        "pinned_450_sha256": _sha256(args.pinned_450),
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "all_pass": all_pass,
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[wrote] {args.out}")
    for c in checks:
        mark = "PASS" if c["pass"] else "FAIL"
        print(f"  [{mark}] {c['name']}: {c['detail']}")
    print(f"all_pass={all_pass}  pin_sha256={pin_sha}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
