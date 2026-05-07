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
    eval_only: bool = False,
    train_eval_split_path: Optional[str | Path] = None,
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
        eval_only: if True, restrict samples to those listed in the eval
            split of {data_dir}/train_eval_split.json (or
            train_eval_split_path if provided). The split is produced by
            `python -m src.dataset --mssbench_split` on the diagnostic side
            and is shared with safety_probes.py. Used for refusal eval to
            avoid contamination with samples that trained the
            comp_safety_shift direction.
        train_eval_split_path: explicit override for the split JSON. Defaults
            to {data_dir}/train_eval_split.json.
    """
    data_dir = Path(data_dir)
    records_path = data_dir / "combined.json"
    if not records_path.exists():
        raise FileNotFoundError(_missing(data_dir))

    splits = splits or _DEFAULT_SPLITS

    eval_ids: Optional[set[str]] = None
    if eval_only:
        split_path = (Path(train_eval_split_path) if train_eval_split_path
                      else data_dir / "train_eval_split.json")
        if not split_path.exists():
            raise FileNotFoundError(
                f"--mssbench_eval_only requires {split_path}.\n"
                "Run: python -m src.dataset --mssbench_split"
            )
        with open(split_path) as f:
            split = json.load(f)
        eval_ids = set(split.get("eval_sample_ids", []))
        if not eval_ids:
            raise ValueError(
                f"{split_path} has no eval_sample_ids; regenerate the split."
            )

    with open(records_path) as f:
        raw = json.load(f)
    # combined.json is {"chat": [...], "embodied": [...]}.
    # Flatten the requested splits into one record list.
    if isinstance(raw, dict):
        records: list[dict] = []
        for split in splits:
            if split in raw and isinstance(raw[split], list):
                records.extend(raw[split])
        if not records:
            raise ValueError(
                f"No records found in combined.json for splits {splits}. "
                f"Available keys: {list(raw.keys())}"
            )
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(f"Unexpected combined.json type: {type(raw).__name__}")

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
                sid = f"mssbench_{rec_idx:04d}_{variant_label}_{stem}_q{q_idx}"
                if eval_ids is not None and sid not in eval_ids:
                    continue
                samples.append(EvalSample(
                    id=sid,
                    question=question,
                    image=image,
                    benchmark="mssbench",
                    safety_label=variant_label,
                    scenario_name=rec_type,
                ))
                if limit is not None and len(samples) >= limit:
                    return samples
    return samples
