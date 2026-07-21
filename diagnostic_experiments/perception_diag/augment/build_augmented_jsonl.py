#!/usr/bin/env python3
"""Build leading-clause (+ filler) augmented JSONL for AMBER/POPE subsets.

Writes JSONL under ``data/amber/`` or ``data/pope/`` (matched to the subset).
Builder-local meta + HTML gallery stay under ``augment/outputs/``.
Filler condition (``filler_b``) comes from
``templates/filler_clauses_v1.json`` and is appended alongside leading-clause
conditions without changing existing leading prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import load_amber, load_pope  # noqa: E402
from src.paths import (  # noqa: E402
    amber_data_dir,
    augmented_jsonl_path,
    pope_data_dir,
    project_root,
)
from src.prompt_spans import encode_no_special  # noqa: E402

ROOT = project_root() / "diagnostic_experiments" / "perception_diag"
TEMPLATES_PATH = ROOT / "templates" / "leading_clauses_v1.json"
FILLER_TEMPLATES_PATH = ROOT / "templates" / "filler_clauses_v1.json"
# Builder-local meta/gallery only (JSONLs land under data/{amber|pope}/).
OUT_DIR = ROOT / "augment" / "outputs"
DATA = project_root() / "data"

# Tokenizers used only for meta token-count reporting (no model weights).
TOKENIZER_IDS = {
    "llava-1.5-7b-hf": "llava-hf/llava-1.5-7b-hf",
    "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
}

SUBSET_FILES = {
    "amber25": DATA / "vti" / "qual_subset_chair5_amber25.json",
    "amber100": DATA / "amber" / "pinned_amber_disc_100.json",
    "amber450": DATA / "amber" / "pinned_amber_disc_450.json",
    "pope30": DATA / "pope" / "pinned_pope_existence_yes_30.json",
    "pope120": DATA / "pope" / "pinned_pope_existence_yes_120.json",  # alias → POPE-yes-120
    "pope_yes_120": DATA / "pope" / "pinned_pope_existence_yes_120.json",
    "pope_no_120": DATA / "pope" / "pinned_pope_existence_no_120.json",
    "pope600": DATA / "pope" / "pinned_eval_ids.json",
}


def _load_templates() -> dict:
    return json.loads(TEMPLATES_PATH.read_text())


def _load_filler_templates() -> dict:
    return json.loads(FILLER_TEMPLATES_PATH.read_text())


def _merge_template_conditions(leading: dict, filler: dict) -> dict:
    """Leading conditions first; append filler conditions by condition_id."""
    by_id = {c["condition_id"]: c for c in leading["conditions"]}
    for c in filler["conditions"]:
        by_id[c["condition_id"]] = c
    # Preserve leading order, then filler order for any new ids
    ordered = []
    seen = set()
    for c in leading["conditions"] + filler["conditions"]:
        cid = c["condition_id"]
        if cid in seen:
            continue
        ordered.append(by_id[cid])
        seen.add(cid)
    return {
        "template_set_id": f"{leading['template_set_id']}+{filler['template_set_id']}",
        "leading_template_set_id": leading["template_set_id"],
        "filler_template_set_id": filler["template_set_id"],
        "version": max(int(leading.get("version", 1)), int(filler.get("version", 1))),
        "conditions": ordered,
    }


def _apply_prefix(question: str, cond: dict) -> str:
    prefix = cond.get("prefix") or ""
    join = cond.get("join") or ""
    if not prefix:
        return question
    return f"{prefix}{join}{question}"


def _content_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _token_counts_by_model(conditions: list[dict]) -> dict:
    """Record prefix token counts per template per model tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return {"error": "transformers not importable"}
    out = {}
    for short, model_id in TOKENIZER_IDS.items():
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        counts = {}
        for cond in conditions:
            prefix = cond.get("prefix") or ""
            join = cond.get("join") or ""
            text = f"{prefix}{join}"
            n = len(encode_no_special(tok, text)) if text else 0
            counts[cond["template_id"]] = {
                "condition_id": cond["condition_id"],
                "prefix_token_count": n,
                "prefix": prefix,
            }
        out[short] = {"tokenizer_id": model_id, "templates": counts}
    return out


def _ids_from_subset(name: str, path: Path) -> list[str]:
    obj = json.loads(path.read_text())
    if name.startswith("amber"):
        return list(obj["amber"])
    if name in ("pope30", "pope120", "pope_yes_120", "pope_no_120"):
        if obj.get("ids"):
            return list(obj["ids"])
        ids: list[str] = []
        for split_name in ("random", "popular", "adversarial"):
            ids.extend((obj.get("splits") or {}).get(split_name) or [])
        return ids
    splits = obj.get("splits") or {}
    ids = []
    for split_name in ("random", "popular", "adversarial"):
        ids.extend(splits.get(split_name) or [])
    return ids


def _filter_conditions(templates: dict, condition_ids: list[str] | None) -> dict:
    """Keep only requested condition_ids (order preserved). None = keep all."""
    if not condition_ids:
        return templates
    by_id = {c["condition_id"]: c for c in templates["conditions"]}
    missing = [c for c in condition_ids if c not in by_id]
    if missing:
        raise SystemExit(f"Unknown --condition_ids: {missing}")
    ordered = [by_id[c] for c in condition_ids]
    out = dict(templates)
    out["conditions"] = ordered
    out["condition_filter"] = list(condition_ids)
    return out


def _load_amber_items(ids: list[str]) -> list[dict]:
    samples = load_amber(task="discriminative", subset_ids=set(ids))
    by_id = {s["id"]: s for s in samples}
    out = []
    for i in ids:
        s = by_id.get(i)
        if s is None:
            continue
        raw = s.get("raw") or {}
        out.append({
            "item_id": s["id"],
            "image_id": raw.get("image") or s.get("image_path"),
            "gold": s.get("label"),
            "qtype": s.get("category"),
            "question": s["text"],
            "image_path": s.get("image_path"),
        })
    return out


def _load_pope_items(ids: list[str]) -> list[dict]:
    by_id = {}
    for split in ("random", "popular", "adversarial"):
        for s in load_pope(split=split):
            by_id[s["id"]] = {**s, "_split": split}
    out = []
    for i in ids:
        s = by_id.get(i)
        if s is None:
            continue
        raw = s.get("raw") or {}
        out.append({
            "item_id": s["id"],
            "image_id": raw.get("image") or s.get("image_path"),
            "gold": s.get("label"),
            "qtype": s.get("category"),
            "question": s["text"],
            "split": s.get("_split"),
            "image_path": s.get("image_path"),
        })
    return out


def _augment(items: list[dict], templates: dict) -> list[dict]:
    rows = []
    for item in items:
        variants = []
        for cond in templates["conditions"]:
            variants.append({
                "condition_id": cond["condition_id"],
                "template_id": cond["template_id"],
                "tone": cond.get("tone"),
                "direction": cond.get("direction"),
                "prompt": _apply_prefix(item["question"], cond),
            })
        rows.append({
            **{k: v for k, v in item.items() if k != "question"},
            "question_neutral": item["question"],
            "variants": variants,
            "template_set_id": templates["template_set_id"],
        })
    return rows


def _write_gallery(rows: list[dict], path: Path, n: int = 30) -> None:
    show = rows[:n]
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>A6 leading-clause gallery</title>",
        "<style>body{font-family:sans-serif;max-width:1000px;margin:1rem auto}"
        ".card{border:1px solid #bbb;margin:1rem 0;padding:0.75rem}"
        ".cond{margin:0.4rem 0;padding:0.4rem;background:#f7f7f7}"
        ".meta{color:#555;font-size:0.9rem}</style></head><body>",
        f"<h1>A6 leading-clause gallery</h1>"
        f"<p>template_set={html.escape(show[0]['template_set_id'] if show else '')} · "
        f"showing {len(show)} items</p>",
    ]
    for r in show:
        parts.append("<div class='card'>")
        parts.append(
            f"<div class='meta'>{html.escape(r['item_id'])} · gold={html.escape(str(r.get('gold')))} · "
            f"qtype={html.escape(str(r.get('qtype')))}</div>"
        )
        for v in r["variants"]:
            parts.append(
                f"<div class='cond'><b>{html.escape(v['condition_id'])}</b> "
                f"<span class='meta'>({html.escape(v['template_id'])})</span><br>"
                f"{html.escape(v['prompt'])}</div>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--subsets", nargs="+", default=["amber100", "pope30"],
        choices=list(SUBSET_FILES.keys()),
    )
    p.add_argument(
        "--skip_fillers", action="store_true",
        help="Emit leading-clause conditions only (legacy).",
    )
    p.add_argument(
        "--condition_ids",
        nargs="+",
        default=None,
        help=(
            "Optional subset of condition_ids to emit (order preserved). "
            "Example: neutral assertive_toward_no filler_b"
        ),
    )
    p.add_argument(
        "--out_name",
        default=None,
        help=(
            "Optional output stem override for a single --subsets entry "
            "(writes augmented_{out_name}.jsonl). Use when filtering "
            "conditions so the default augmented_{subset}.jsonl is not overwritten."
        ),
    )
    p.add_argument(
        "--skip_token_counts", action="store_true",
        help="Skip per-model tokenizer length recording in meta.",
    )
    args = p.parse_args()

    leading = _load_templates()
    if args.skip_fillers:
        templates = leading
        filler = None
    else:
        filler = _load_filler_templates()
        templates = _merge_template_conditions(leading, filler)
    templates = _filter_conditions(templates, args.condition_ids)
    if args.out_name and len(args.subsets) != 1:
        raise SystemExit("--out_name requires exactly one --subsets entry")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    amber_data_dir().mkdir(parents=True, exist_ok=True)
    pope_data_dir().mkdir(parents=True, exist_ok=True)

    meta = {
        "template_set_id": templates["template_set_id"],
        "template_path": str(TEMPLATES_PATH),
        "template_hash": _content_hash(leading),
        "filler_template_path": str(FILLER_TEMPLATES_PATH) if filler else None,
        "filler_template_hash": _content_hash(filler) if filler else None,
        "condition_ids": [c["condition_id"] for c in templates["conditions"]],
        "condition_filter": args.condition_ids,
        "subsets": {},
    }
    if not args.skip_token_counts:
        meta["prefix_token_counts_by_model"] = _token_counts_by_model(
            templates["conditions"]
        )

    gallery_rows: list[dict] = []
    for name in args.subsets:
        path = SUBSET_FILES[name]
        ids = _ids_from_subset(name, path)
        if name.startswith("amber"):
            items = _load_amber_items(ids)
            bench = "amber"
        else:
            items = _load_pope_items(ids)
            bench = "pope"
        rows = _augment(items, templates)
        stem = args.out_name or name
        out_path = augmented_jsonl_path(bench, stem)
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        h = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
        meta_key = stem
        meta["subsets"][meta_key] = {
            "benchmark": bench,
            "subset_name": name,
            "subset_file": str(path),
            "n_ids_requested": len(ids),
            "n_rows": len(rows),
            "n_conditions": len(templates["conditions"]),
            "condition_ids": [c["condition_id"] for c in templates["conditions"]],
            "jsonl": str(out_path),
            "content_hash_sha256_16": h,
        }
        print(f"[{meta_key}] wrote {len(rows)} rows → {out_path} hash={h}")
        if name in ("amber100", "amber25", "pope120") or not gallery_rows:
            gallery_rows = rows

    gallery_path = OUT_DIR / "leading_clause_gallery.html"
    _write_gallery(gallery_rows, gallery_path, n=30)
    meta_path = OUT_DIR / "augmentation_meta.json"
    # merge with existing meta if present
    if meta_path.exists():
        prev = json.loads(meta_path.read_text())
        prev_sub = prev.get("subsets") or {}
        prev_sub.update(meta["subsets"])
        meta["subsets"] = prev_sub
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {gallery_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
