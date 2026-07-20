"""Dump writer (N4): JSONL manifest + per-key npy acts/norms with resume-by-id."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np


class DumpWriter:
    def __init__(self, out_dir: Path, cell_id: str, metadata: dict):
        self.out_dir = Path(out_dir)
        self.cell_id = cell_id
        self.cell_dir = self.out_dir / cell_id
        self.acts_dir = self.cell_dir / "acts"
        self.norms_dir = self.cell_dir / "norms"
        self.cell_dir.mkdir(parents=True, exist_ok=True)
        self.acts_dir.mkdir(exist_ok=True)
        self.norms_dir.mkdir(exist_ok=True)
        self.manifest_path = self.cell_dir / "manifest.jsonl"
        self.meta_path = self.cell_dir / "metadata.json"
        if not self.meta_path.exists():
            self.meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        self._done: Set[str] = set()
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self._done.add(self._key(row["item_id"], row["condition_id"]))

    @staticmethod
    def _key(item_id: str, condition_id: str) -> str:
        return f"{item_id}::{condition_id}"

    @staticmethod
    def _file_stem(item_id: str, condition_id: str) -> str:
        return f"{item_id}__{condition_id}"

    def is_done(self, item_id: str, condition_id: str) -> bool:
        return self._key(item_id, condition_id) in self._done

    def write(
        self,
        *,
        item_id: str,
        condition_id: str,
        template_id: str,
        gold: Any,
        record: Dict[str, Any],
        extra: Optional[dict] = None,
    ) -> None:
        key = self._key(item_id, condition_id)
        if key in self._done:
            return
        scores = record["scores"]
        row = {
            "item_id": item_id,
            "condition_id": condition_id,
            "template_id": template_id,
            "gold": gold,
            "response": record["response"],
            "parsed_outcome": record["parsed_outcome"],
            "first_vs_parsed_agree": record["first_vs_parsed_agree"],
            "degeneracy_flag": bool(record.get("degeneracy_flag", False)),
            "truncated": bool(record.get("truncated", False)),
            "status": record.get("status", "ok"),
            **{f"score_{k}": v for k, v in scores.items()},
        }
        if extra:
            row.update(extra)
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        stem = self._file_stem(item_id, condition_id)
        np.save(self.acts_dir / f"{stem}.npy", record["last_token_acts_fp16"])
        np.save(self.norms_dir / f"{stem}.npy", record["prefill_pos_norms_fp16"])
        self._done.add(key)


def load_acts_dir(cell_dir: Path) -> Dict[str, np.ndarray]:
    """Compatibility helper: map ``item__condition`` → array from npy or legacy npz."""
    acts_dir = Path(cell_dir) / "acts"
    out: Dict[str, np.ndarray] = {}
    if acts_dir.is_dir():
        for p in acts_dir.glob("*.npy"):
            out[p.stem] = np.load(p)
        return out
    npz = Path(cell_dir) / "last_token_acts.npz"
    if npz.exists():
        with np.load(npz) as z:
            return {k: z[k] for k in z.files}
    return out


def load_norms_dir(cell_dir: Path) -> Dict[str, np.ndarray]:
    """Map ``item__condition`` → prefill pos-norm array (L+1, T) from npy or legacy npz."""
    norms_dir = Path(cell_dir) / "norms"
    out: Dict[str, np.ndarray] = {}
    if norms_dir.is_dir():
        for p in norms_dir.glob("*.npy"):
            out[p.stem] = np.load(p)
        return out
    npz = Path(cell_dir) / "prefill_pos_norms.npz"
    if npz.exists():
        with np.load(npz) as z:
            return {k: z[k] for k in z.files}
    return out
