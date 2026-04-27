"""
MSSBench loader (Multimodal Situational Safety Benchmark, Zhou et al., ICLR 2025).

Expected on-disk layout after running evaluation/scripts/download_mssbench.py
(source: kzhou35/mssbench on HuggingFace):

    data/mssbench/
    ├── combined.json   (record list, see schema below)
    ├── chat/
    │   └── {N}.jpg     (paired safe/unsafe images referenced by combined.json)
    └── embodied/       (a separate embodied-agent split; not loaded by default)

`combined.json` is a list of records, each with the following fields:

    {
      "safe_image_path":   "0.jpg",          # relative to chat/
      "unsafe_image_path": "1.jpg",
      "intent":            "...",            # safe activity description
      "unsafe_image":      "...",            # unsafe variant description
      "Type":              "property"        # category: property | harmful | offensive
      "queries": [
          "query 1 ...",
          "query 2 ..."
      ]
    }

Each record produces 2 × len(queries) EvalSample instances:
  - For every query, one sample with safe_image_path  -> safety_label="SSS"
  - For every query, one sample with unsafe_image_path -> safety_label="SSU"

In MSSBench, the *query* is benign in both contexts; the IMAGE determines
whether a refusal is the safe response. The SSU half is the relevant attack
set for ASR; the SSS half acts as a negative control.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PIL import Image

from . import EvalSample


_DEFAULT_SPLITS = ("chat",)  # extend with "embodied" if/when the loader supports it


def _missing(data_dir: Path) -> str:
    return (
        f"MSSBench not found at {data_dir} "
        f"(expected combined.json + chat/ subfolder).\n"
        "Download with:\n"
        "    python evaluation/scripts/download_mssbench.py"
    )


def _resolve_image(rel_path: str, data_dir: Path,
                   splits: tuple[str, ...]) -> Optional[Path]:
    """Resolve a relative image filename against the candidate split folders."""
    if not rel_path:
        return None
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        return p
    for split in splits:
        cand = data_dir / split / rel_path
        if cand.exists():
            return cand
    cand = data_dir / rel_path
    if cand.exists():
        return cand
    matches = list(data_dir.rglob(p.name))
    return matches[0] if matches else None


def load_mssbench(
    data_dir: str | Path = "data/mssbench",
    safety_labels: Optional[list[str]] = None,
    limit: Optional[int] = None,
    splits: Optional[tuple[str, ...]] = None,
) -> list[EvalSample]:
    """
    Args:
        data_dir: root of the downloaded mssbench tree.
        safety_labels: subset of {"SSS", "SSU"}. Default = both.
        limit: cap total samples returned.
        splits: which top-level image-folder splits to search. Default
            ("chat",) — the chat/conversation split. Pass ("chat", "embodied")
            to also include embodied images (their schema is similar enough
            to be loaded as long as combined.json paths resolve).
    """
    data_dir = Path(data_dir)
    records_path = data_dir / "combined.json"
    if not records_path.exists():
        raise FileNotFoundError(_missing(data_dir))

    splits = splits or _DEFAULT_SPLITS

    with open(records_path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected combined.json to be a list, got "
                         f"{type(records).__name__}")

    wanted_labels = set(safety_labels) if safety_labels else {"SSS", "SSU"}

    samples: list[EvalSample] = []
    for rec_idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        queries = record.get("queries") or []
        if not isinstance(queries, list):
            continue
        rec_type = record.get("Type") or record.get("type")

        for variant_label, path_field in (
            ("SSS", "safe_image_path"),
            ("SSU", "unsafe_image_path"),
        ):
            if variant_label not in wanted_labels:
                continue
            rel = record.get(path_field)
            if not isinstance(rel, str) or not rel.strip():
                continue
            img_path = _resolve_image(rel, data_dir, splits)
            if img_path is None:
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  [mssbench] Failed to open {img_path}: {e}")
                continue
            stem = img_path.stem
            for q_idx, question in enumerate(queries):
                if not isinstance(question, str) or not question.strip():
                    continue
                samples.append(EvalSample(
                    id=f"mssbench_{rec_idx:04d}_{variant_label}_{stem}_q{q_idx}",
                    question=question,
                    image=image,
                    benchmark="mssbench",
                    safety_label=variant_label,
                    scenario_name=rec_type,
                ))
                if limit is not None and len(samples) >= limit:
                    return samples
    return samples
