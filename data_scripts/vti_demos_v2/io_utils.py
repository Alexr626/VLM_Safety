"""JSONL I/O, resume-by-id, call logging, stage summaries."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _iter_json_objects(text: str) -> Iterator[dict]:
    """Parse JSONL or IDE-beautified consecutive JSON objects from a file body.

    Prefer one object per line. If a non-empty line fails ``json.loads`` (common
    when an editor pretty-prints JSONL with tabs/newlines), fall back to
    streaming ``JSONDecoder.raw_decode`` over the full text.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return
    try:
        for line in lines:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError("jsonl record is not an object")
            yield obj
        return
    except (json.JSONDecodeError, TypeError):
        pass

    dec = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = dec.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise TypeError(f"expected JSON object, got {type(obj).__name__}")
        yield obj
        idx = end


def read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    try:
        return list(_iter_json_objects(text))
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"failed to parse JSONL at {path}: {e}") from e


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.is_file():
        return
    yield from read_jsonl(path)


def append_jsonl(path: Path, record: dict) -> None:
    ensure_parent(path)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    ensure_parent(path)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_ids(path: Path, id_key: str = "id") -> Set[str]:
    return {str(r[id_key]) for r in read_jsonl(path) if id_key in r}


def load_exclude_ids_from_demos(demos_path: Path) -> Set[str]:
    return load_ids(demos_path)


def collect_seen_ids(artifact_paths: Iterable[Path]) -> Set[str]:
    seen: Set[str] = set()
    for p in artifact_paths:
        seen |= load_ids(p)
    return seen


def write_summary(path: Path, summary: Dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def append_call_log(path: Path, entry: Dict[str, Any]) -> None:
    append_jsonl(path, entry)


def rejection_histogram(records: Iterable[dict],
                        reason_key: str = "reject_reason") -> Dict[str, int]:
    c: Counter = Counter()
    for r in records:
        reason = r.get(reason_key) or r.get("reason") or "unknown"
        if isinstance(reason, list):
            reason = ";".join(str(x) for x in reason)
        c[str(reason)] += 1
    return dict(c.most_common())


def today_iso() -> str:
    return date.today().isoformat()
