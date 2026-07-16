#!/usr/bin/env python3
"""Deterministically assign one verified v2.1 option per dimension.

Existing allocation rows are immutable: delete stage1b–5 artifacts for a full
reallocation. Top-ups only receive assignments for previously unseen ids.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parents[1]
sys.path[:0] = [str(_ROOT), str(_PKG.parent)]

from vti_demos_v2 import config  # noqa: E402
from vti_demos_v2.io_utils import read_jsonl, write_jsonl, write_summary  # noqa: E402
from src.paths import vti_demos_v2_dir  # noqa: E402


def _phrase(kind: str, side: str) -> str:
    return {
        "horizontal": side,
        "vertical": side,
        "support": "on top of",
        "proximity": "right next to" if side == "near" else "far away from",
    }[kind]


def _flip_side(rel_type: str, side: str) -> str:
    return {
        "horizontal": {"left": "right", "right": "left"},
        "vertical": {"above": "below", "below": "above"},
        "proximity": {"near": "far", "far": "near"},
        "support": {},
    }.get(rel_type, {}).get(side, side)


def _direction_key(rel: dict) -> str:
    """Bucket used for left/right and above/below balancing."""
    t = rel["type"]
    side = rel.get("true") or rel.get("a_side")
    if t == "horizontal":
        return f"horizontal:{side}"
    if t == "vertical":
        return f"vertical:{side}"
    if t == "proximity":
        return f"proximity:{side}"
    return f"{t}:fixed"


def _side(opt: dict) -> str:
    """Truthful pole for A relative to B (from stage-0 geometry)."""
    if opt.get("a_side"):
        return str(opt["a_side"])
    if opt.get("type") == "support":
        return "on_top_of"
    raise KeyError(f"relation option missing a_side: {opt}")


def allocate(
    rows: list[dict],
    existing: list[dict] | None = None,
    seed: int = config.ALLOC_SEED,
) -> list[dict]:
    """Allocate feasible records, preserving existing ids and deterministic order."""
    existing = existing or []
    done = {r["id"] for r in existing}
    rng = random.Random(seed)
    hist: Counter = Counter()
    dir_hist: Counter = Counter()
    n_ge3 = 0
    for r in existing:
        hist[("relation", r["relation"]["type"])] += 1
        hist[("attribute", r["attribute"]["type"])] += 1
        dcat = r["distractor"] if isinstance(r["distractor"], str) else r["distractor"]["category"]
        hist[("distractor", dcat)] += 1
        hist[("counting", r["counting"]["category"])] += 1
        hist[("attr_value", r["attribute"]["true_value"].lower())] += 1
        dir_hist[_direction_key(r["relation"])] += 1
        if int(r["counting"]["count"]) >= 3:
            n_ge3 += 1

    def constrained(r: dict) -> tuple:
        return (
            sum(
                len(r.get(k, []))
                for k in (
                    "verified_counting_options",
                    "verified_relation_options",
                    "verified_distractor_options",
                    "verified_attribute_options",
                )
            ),
            r["id"],
        )

    out = list(existing)
    for r in sorted((x for x in rows if x["id"] not in done), key=constrained):
        pool_n = max(len(out) + 1, 1)

        # Backfill a_side from stage0 relation_options when model stripped geometry.
        s0_rels = {
            (str(x.get("a", "")).lower(), str(x.get("b", "")).lower(), str(x.get("type", "")).lower()): x
            for x in (r.get("relation_options") or [])
        }
        rels = []
        for opt in r.get("verified_relation_options") or []:
            if opt.get("a_side") or opt.get("type") == "support":
                rels.append(opt)
                continue
            key = (
                str(opt.get("a", "")).lower(),
                str(opt.get("b", "")).lower(),
                str(opt.get("type", "")).lower(),
            )
            base = s0_rels.get(key)
            if base is not None:
                rels.append({**base, **opt})
        if not rels:
            continue

        def rel_score(opt: dict) -> tuple:
            t = opt["type"]
            side = _side(opt)
            type_frac = hist[("relation", t)] / max(pool_n * config.REL_TYPE_TARGET.get(t, 0.25), 1e-6)
            dir_frac = dir_hist[f"{t}:{side}"] / max(pool_n * 0.5, 1e-6)
            return (type_frac, dir_frac, t, opt["a"], opt["b"])

        relation = min(rels, key=rel_score)
        side = _side(relation)
        a, b = relation["a"], relation["b"]
        # Balance horizontal/vertical by swapping A/B when the opposite pole is scarcer.
        if relation["type"] in ("horizontal", "vertical"):
            flipped = _flip_side(relation["type"], side)
            if dir_hist[f"{relation['type']}:{flipped}"] < dir_hist[f"{relation['type']}:{side}"]:
                a, b = b, a
                side = flipped
        elif relation["type"] == "proximity":
            flipped = _flip_side("proximity", side)
            alt = next(
                (
                    o for o in rels
                    if o["type"] == "proximity"
                    and o["a"] == relation["a"]
                    and o["b"] == relation["b"]
                    and _side(o) == flipped
                ),
                None,
            )
            if alt and dir_hist[f"proximity:{flipped}"] < dir_hist[f"proximity:{side}"]:
                relation = alt
                side = flipped

        relation = {
            **relation,
            "a": a,
            "b": b,
            "a_side": side,
            "true": side if relation["type"] != "support" else "on_top_of",
            "true_phrase": _phrase(relation["type"], side),
        }

        counts = list(r["verified_counting_options"])
        # Prefer non-overlapping categories; soft person / N>=3 targets.
        def count_score(opt: dict) -> tuple:
            cat = opt["category"]
            overlap = 1 if cat in (relation["a"], relation["b"]) else 0
            person_pen = 0
            if cat == "person":
                person_pen = hist[("counting", "person")] / max(
                    pool_n * config.PERSON_COUNT_CAP_FRAC, 1e-6
                )
            ge3_need = (n_ge3 / pool_n) < config.N_GE3_TARGET_FRAC
            ge3_bonus = 0 if (ge3_need and opt["count"] >= 3) else 1
            return (overlap, person_pen, ge3_bonus, hist[("counting", cat)], -opt["count"], cat)

        counts_sorted = sorted(counts, key=count_score)
        count = next(
            (c for c in counts_sorted if c["category"] not in (relation["a"], relation["b"])),
            None,
        )
        if count is None:
            continue

        eligible_most = (
            count["count"] >= config.AT_MOST_MIN_N and count.get("count_complete", False)
        )
        mode = (
            "at_most"
            if eligible_most and rng.random() < config.COUNT_MODE_AT_MOST_FRAC
            else "at_least"
        )

        distractors = list(r["verified_distractor_options"])
        if not distractors:
            continue
        weights = [
            (max(float(x.get("score", 0)), 1e-12) ** config.DISTRACTOR_TAU)
            / (1 + hist[("distractor", x["category"])])
            for x in distractors
        ]
        # Soft cap: down-weight categories already over DISTRACTOR_CAP_FRAC.
        for i, x in enumerate(distractors):
            if hist[("distractor", x["category"])] / pool_n >= config.DISTRACTOR_CAP_FRAC:
                weights[i] *= 0.05
        distractor_opt = rng.choices(distractors, weights=weights, k=1)[0]
        distractor_cat = distractor_opt["category"]

        attrs = list(r["verified_attribute_options"])
        if not attrs:
            continue

        def attr_score(opt: dict) -> tuple:
            t = opt["type"]
            type_frac = hist[("attribute", t)] / max(
                pool_n * config.ATTR_TYPE_TARGET.get(t, 0.2), 1e-6
            )
            val_frac = hist[("attr_value", opt["true_value"].lower())] / max(
                pool_n * config.ATTR_VALUE_CAP_FRAC, 1e-6
            )
            return (type_frac, val_frac, opt["true_value"], opt["object"])

        attr = min(attrs, key=attr_score)

        item = {
            **r,
            "counting": {**count, "mode": mode},
            "relation": relation,
            "distractor": distractor_cat,
            "distractor_meta": distractor_opt,
            "attribute": attr,
        }
        out.append(item)
        hist[("relation", relation["type"])] += 1
        hist[("attribute", attr["type"])] += 1
        hist[("distractor", distractor_cat)] += 1
        hist[("counting", count["category"])] += 1
        hist[("attr_value", attr["true_value"].lower())] += 1
        dir_hist[_direction_key(relation)] += 1
        if int(count["count"]) >= 3:
            n_ge3 += 1
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    v2 = vti_demos_v2_dir()
    out = args.out or (v2 / "stage1b_allocation.jsonl")
    rows = read_jsonl(args.input or (v2 / "stage1_verified.jsonl"))
    existing = read_jsonl(out)
    assigned = allocate(rows, existing)
    write_jsonl(out, assigned)
    summary = {
        "n_allocated": len(assigned),
        "n_new": len(assigned) - len(existing),
        "relation_types": dict(Counter(r["relation"]["type"] for r in assigned)),
        "relation_directions": dict(Counter(_direction_key(r["relation"]) for r in assigned)),
        "attribute_types": dict(Counter(r["attribute"]["type"] for r in assigned)),
        "count_modes": dict(Counter(r["counting"]["mode"] for r in assigned)),
        "counting_categories": dict(Counter(r["counting"]["category"] for r in assigned)),
        "distractors": dict(Counter(
            r["distractor"] if isinstance(r["distractor"], str) else r["distractor"]["category"]
            for r in assigned
        )),
        "n_ge3": sum(1 for r in assigned if int(r["counting"]["count"]) >= 3),
    }
    write_summary(v2 / "stage1b_summary.json", summary)
    print(f"Wrote {len(assigned)} allocations -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
