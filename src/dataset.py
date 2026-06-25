"""
Hallucination benchmark loaders and path helpers.

Uniform diagnostic sample dict:
  {
    "id": str,
    "image_path": str | None,
    "image_pil": PIL.Image | None,
    "text": str,
    "label": str,
    "label_idx": int | None,
    "benchmark": str,
    "task": str | None,
    "category": str | None,
    "raw": dict,
  }
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Benchmark registry ────────────────────────────────────────────────────────

BENCHMARK_REGISTRY: Dict[str, dict] = {}


def _register(name: str, data_dir: str, loader: Callable, **extra):
    BENCHMARK_REGISTRY[name] = {"data_dir": data_dir, "loader": loader, **extra}


DATASET_DATA_DIRS: Dict[str, str] = {}


def _sync_data_dirs():
    DATASET_DATA_DIRS.clear()
    DATASET_DATA_DIRS.update(
        {k: v["data_dir"] for k, v in BENCHMARK_REGISTRY.items()}
    )


# ── Schema helpers ────────────────────────────────────────────────────────────

_TYPE_FIELD_CANDIDATES = [
    "type", "category_type", "subset", "task", "question_type",
]
_TEXT_FIELD_CANDIDATES = [
    "query", "question", "instruction", "text", "prompt", "caption",
]
_IMAGE_FIELD_CANDIDATES = [
    "image", "image_path", "img", "img_path", "image_file", "image_id",
]
_CATEGORY_FIELD_CANDIDATES = [
    "category", "topic", "subcategory", "hallucination_type",
]


def _detect_field(entry: dict, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in entry:
            return c
    return None


def inspect_schema(data, title: str = "Dataset") -> dict:
    entries = data if isinstance(data, list) else list(data.values())
    print(f"\n{'='*60}")
    print(f"  {title} Schema Inspection")
    print(f"{'='*60}")
    print(f"  Total entries : {len(entries)}")
    if not entries:
        return {}
    sample = entries[0]
    print(f"  Keys          : {sorted(sample.keys())}")
    type_field = _detect_field(sample, _TYPE_FIELD_CANDIDATES)
    if type_field:
        counts = Counter(e.get(type_field) for e in entries)
        print(f"  Type field    : '{type_field}'")
        print(f"  Unique values : {dict(counts)}")
    return {"n": len(entries), "keys": list(sample.keys())}


def benchmark_data_dir(benchmark: str, project_root: Optional[Path] = None) -> Path:
    root = project_root or _PROJECT_ROOT
    data_dir = DATASET_DATA_DIRS.get(benchmark, benchmark)
    return root / "data" / data_dir


def combined_json_path(benchmark: str, project_root: Optional[Path] = None) -> Path:
    return benchmark_data_dir(benchmark, project_root) / "combined.json"


def load_combined(benchmark: str, project_root: Optional[Path] = None) -> List[dict]:
    path = combined_json_path(benchmark, project_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python data_scripts/download_{benchmark}.py"
        )
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else list(data.values())


def _coco_image_path(coco_root: Path, image_id_or_name: str) -> Optional[Path]:
    """Resolve COCO val2014 image by numeric id or filename."""
    s = str(image_id_or_name)
    if s.endswith(".jpg"):
        candidates = [coco_root / s]
    else:
        num = s.zfill(12)
        candidates = [
            coco_root / f"COCO_val2014_{num}.jpg",
            coco_root / f"{num}.jpg",
        ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_pil(path: Optional[Path]) -> Optional[Image.Image]:
    if path is None or not path.exists():
        return None
    return Image.open(path).convert("RGB")


def _yes_no_label_idx(label: str) -> Optional[int]:
    l = label.strip().lower()
    if l in ("yes", "y", "true", "1"):
        return 1
    if l in ("no", "n", "false", "0"):
        return 0
    return None


# ── POPE ──────────────────────────────────────────────────────────────────────

def load_pope(
    data_dir: Optional[Path] = None,
    split: str = "random",
    limit: Optional[int] = None,
) -> List[dict]:
    """Load POPE object-presence yes/no questions (COCO val2014)."""
    root = data_dir or benchmark_data_dir("pope")
    manifest = root / "combined.json"
    if manifest.exists():
        entries = load_combined("pope", root.parent.parent)
        entries = [
            e for e in entries
            if e.get("category") == split or e.get("task") == split
        ]
    else:
        pope_dir = root / "output" / "coco" if (root / "output" / "coco").exists() else root
        json_files = sorted(pope_dir.glob(f"*{split}*.json"))
        if not json_files:
            json_files = sorted(pope_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No POPE JSON in {root}. Run: python data_scripts/download_pope.py"
            )
        entries = []
        coco_root = _PROJECT_ROOT / "data" / "coco" / "val2014"
        for jf in json_files:
            raw = []
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw.append(json.loads(line))
            for i, row in enumerate(raw):
                q = row.get("text") or row.get("question", "")
                label = row.get("label", "")
                img_key = row.get("image") or row.get("image_id", "")
                if str(img_key).endswith(".jpg"):
                    img_path = coco_root / str(img_key)
                else:
                    img_path = _coco_image_path(coco_root, img_key)
                entries.append({
                    "id": f"pope_{jf.stem}_{i:05d}",
                    "image_path": str(img_path) if img_path else None,
                    "text": q,
                    "label": label,
                    "label_idx": _yes_no_label_idx(label),
                    "benchmark": "pope",
                    "task": split,
                    "category": jf.stem,
                    "raw": row,
                })
        _save_combined(entries, root)

    samples = []
    coco_root = _PROJECT_ROOT / "data" / "coco" / "val2014"
    for row in entries:
        img_path = row.get("image_path")
        if img_path and not Path(img_path).exists():
            img_path = str(_coco_image_path(coco_root, row.get("raw", {}).get("image", "")) or "")
        pil = _load_pil(Path(img_path)) if img_path else None
        samples.append({
            "id": row["id"],
            "image_path": img_path,
            "image_pil": pil,
            "text": row["text"],
            "label": row["label"],
            "label_idx": row.get("label_idx"),
            "benchmark": "pope",
            "task": row.get("task", split),
            "category": row.get("category"),
            "raw": row.get("raw", row),
        })
    return samples[:limit] if limit else samples


# ── AMBER ─────────────────────────────────────────────────────────────────────

# AMBER discriminative dimensions (paper / official `inference.py` `de/da/dr`).
# The annotation `type` strings map to the three top-level dimensions; existence
# questions are tagged `discriminative-hallucination` upstream.
def _amber_discriminative_qtype(ann_type: str) -> str:
    t = (ann_type or "").lower()
    if "hallucination" in t:
        return "existence"
    if "attribute" in t:
        return "attribute"
    if "relation" in t:
        return "relation"
    return ann_type or "unknown"


def _load_amber_annotations(root: Path) -> Dict[int, dict]:
    """Map AMBER question id -> annotation entry (gold `truth`, `type`).

    Gold answers for the discriminative split are NOT in `combined.json` (the
    query files carry only id/image/query); they live in the AMBER source
    `data/annotations.json`, joined by question id. Returns {} if the file is
    absent (callers then leave label/category unenriched).
    """
    ann_path = root / "data" / "annotations.json"
    if not ann_path.exists():
        return {}
    with open(ann_path) as f:
        anns = json.load(f)
    return {a["id"]: a for a in anns if "id" in a}


def load_amber(
    data_dir: Optional[Path] = None,
    task: Optional[str] = None,
    limit: Optional[int] = None,
    subset_ids: Optional[set] = None,
) -> List[dict]:
    root = data_dir or benchmark_data_dir("amber")
    entries = load_combined("amber") if combined_json_path("amber").exists() else []
    if not entries:
        raise FileNotFoundError(
            f"Missing AMBER data in {root}. Run: python data_scripts/download_amber.py"
        )
    annotations = _load_amber_annotations(root)
    # Filter by task (and optional pinned subset / `limit`) BEFORE loading images,
    # so limited or subset runs don't pay to open the full ~14k-image set.
    # `subset_ids` (a pinned id set) takes precedence over `limit` when given.
    filtered = [r for r in entries
                if not task or r.get("task", "discriminative") == task]
    if subset_ids is not None:
        ids = set(subset_ids)
        filtered = [r for r in filtered if r["id"] in ids]
    elif limit:
        filtered = filtered[:limit]
    samples = []
    img_root = root / "images"
    for row in filtered:
        t = row.get("task", "discriminative")
        rel = row.get("image_path") or row.get("image", "")
        img_path = img_root / rel if rel and not Path(rel).is_absolute() else Path(rel or "")
        pil = _load_pil(img_path) if img_path else None
        raw = row.get("raw", row)
        label = row.get("label", "")
        category = row.get("category")
        # Enrich discriminative items with gold (yes/no) + question-type dimension
        # from the AMBER annotation file. Generative items keep their existing
        # (out-of-scope) fields untouched.
        if t == "discriminative":
            ann = annotations.get(raw.get("id")) if isinstance(raw, dict) else None
            if ann is not None:
                label = ann.get("truth", label)
                category = _amber_discriminative_qtype(ann.get("type", ""))
        samples.append({
            "id": row["id"],
            "image_path": str(img_path) if img_path else None,
            "image_pil": pil,
            "text": row["text"],
            "label": label,
            "label_idx": _yes_no_label_idx(label) if t == "discriminative" else row.get("label_idx"),
            "benchmark": "amber",
            "task": t,
            "category": category,
            "raw": raw,
        })
    return samples


# ── CHAIR ─────────────────────────────────────────────────────────────────────

_CHAIR_PROMPT = "Please describe this image in detail."


def load_chair(
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    subset_ids: Optional[set] = None,
    prompt_override: Optional[str] = None,
) -> List[dict]:
    """Load CHAIR eval images (COCO val2014) with a caption-generation prompt.

    `subset_ids` pins a fixed sample-id subset (filtered BEFORE image load, so a
    500-id subset does not open all ~40k COCO images); it takes precedence over
    `limit`. `prompt_override` replaces the stored caption prompt verbatim for
    every sample (used to pin the exact VTI prompt "Please Describe this image
    in detail." across the diagnostics).
    """
    root = data_dir or benchmark_data_dir("chair")
    entries = load_combined("chair") if combined_json_path("chair").exists() else []
    if not entries:
        raise FileNotFoundError(
            f"Missing CHAIR manifest in {root}. Run: python data_scripts/download_chair.py"
        )
    if subset_ids is not None:
        ids = set(subset_ids)
        entries = [r for r in entries if r["id"] in ids]
    elif limit:
        entries = entries[:limit]
    samples = []
    for row in entries:
        img_path = Path(row["image_path"])
        samples.append({
            "id": row["id"],
            "image_path": str(img_path),
            "image_pil": _load_pil(img_path),
            "text": prompt_override or row.get("text", _CHAIR_PROMPT),
            "label": row.get("label", "caption"),
            "label_idx": None,
            "benchmark": "chair",
            "task": "generative",
            "category": row.get("category", "coco"),
            "raw": row.get("raw", row),
        })
    return samples


# ── HallusionBench ────────────────────────────────────────────────────────────

def load_hallusionbench(
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    root = data_dir or benchmark_data_dir("hallusionbench")
    entries = load_combined("hallusionbench") if combined_json_path("hallusionbench").exists() else []
    if not entries:
        raise FileNotFoundError(
            f"Missing HallusionBench data in {root}. "
            "Run: python data_scripts/download_hallusionbench.py"
        )
    img_root = root / "images"
    samples = []
    for row in entries:
        rel = row.get("image_path") or row.get("filename", "")
        img_path = img_root / rel if rel else None
        pil = _load_pil(img_path) if img_path else None
        samples.append({
            "id": row["id"],
            "image_path": str(img_path) if img_path else None,
            "image_pil": pil,
            "text": row["text"],
            "label": row.get("label", ""),
            "label_idx": row.get("label_idx"),
            "benchmark": "hallusionbench",
            "task": row.get("task"),
            "category": row.get("category"),
            "raw": row.get("raw", row),
        })
    return samples[:limit] if limit else samples


# ── MMHal-Bench ───────────────────────────────────────────────────────────────

def load_mmhal_bench(
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    root = data_dir or benchmark_data_dir("mmhal_bench")
    entries = load_combined("mmhal_bench") if combined_json_path("mmhal_bench").exists() else []
    if not entries:
        raise FileNotFoundError(
            f"Missing MMHal-Bench data in {root}. "
            "Run: python data_scripts/download_mmhal_bench.py"
        )
    img_root = root / "images"
    samples = []
    for row in entries:
        rel = row.get("image_path") or row.get("image", "")
        img_path = img_root / rel if rel and not Path(rel).is_absolute() else Path(rel or "")
        pil = _load_pil(img_path) if img_path else None
        samples.append({
            "id": row["id"],
            "image_path": str(img_path) if img_path else None,
            "image_pil": pil,
            "text": row["text"],
            "label": row.get("label", ""),
            "label_idx": row.get("label_idx"),
            "benchmark": "mmhal_bench",
            "task": row.get("task"),
            "category": row.get("category"),
            "raw": row.get("raw", row),
        })
    return samples[:limit] if limit else samples


def _save_combined(entries: List[dict], benchmark_dir: Path) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    with open(benchmark_dir / "combined.json", "w") as f:
        json.dump(entries, f, indent=2)


def load_benchmark(
    name: str,
    limit: Optional[int] = None,
    **kwargs,
) -> List[dict]:
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available: {sorted(BENCHMARK_REGISTRY)}"
        )
    loader = BENCHMARK_REGISTRY[name]["loader"]
    return loader(limit=limit, **kwargs)


def load_image_for_sample(sample: dict) -> Optional[Image.Image]:
    if sample.get("image_pil") is not None:
        return sample["image_pil"]
    path = sample.get("image_path")
    return _load_pil(Path(path)) if path else None


# ── Path helpers ──────────────────────────────────────────────────────────────

def model_data_root(benchmark: str, model_short: str,
                    project_root: Optional[Path] = None) -> Path:
    root = project_root or _PROJECT_ROOT
    return root / "data" / DATASET_DATA_DIRS.get(benchmark, benchmark) / model_short


def model_activations_dir(benchmark: str, model_short: str,
                          project_root: Optional[Path] = None) -> Path:
    return model_data_root(benchmark, model_short, project_root) / "activations"


def model_responses_dir(benchmark: str, model_short: str,
                        intervention: str = "no_intervention",
                        project_root: Optional[Path] = None) -> Path:
    return model_data_root(benchmark, model_short, project_root) / "responses" / intervention


# Register benchmarks
_register("pope", "pope", load_pope)
_register("amber", "amber", load_amber)
_register("chair", "chair", load_chair)
_register("hallusionbench", "hallusionbench", load_hallusionbench)
_register("mmhal_bench", "mmhal-bench", load_mmhal_bench)
_sync_data_dirs()

ALL_BENCHMARKS = list(BENCHMARK_REGISTRY.keys())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Inspect hallucination benchmark loaders.")
    p.add_argument("--benchmark", required=True, choices=ALL_BENCHMARKS)
    p.add_argument("--limit", type=int, default=3)
    args = p.parse_args()
    samples = load_benchmark(args.benchmark, limit=args.limit)
    inspect_schema(samples, title=args.benchmark)
    for s in samples:
        print(f"  {s['id']}: label={s['label']!r} task={s.get('task')}")
