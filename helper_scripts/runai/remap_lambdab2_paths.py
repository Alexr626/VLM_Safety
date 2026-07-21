#!/usr/bin/env python3
"""Rewrite lambdab2 absolute paths to the local (NFS) repo root.

Lambdab2 builds (augment JSONLs, data/*/combined.json, some metadata) bake in:

  /home/romanus/dev/vlm_hallucination_mitigation_summer_2026/...

On RunAI the same files live under:

  /home/datalake/romanus/vlm_hallucination/...

This script does a text-level prefix replace in JSON / JSONL (and optional
extra extensions). Dry-run by default; pass --apply to write.

Examples (inside the NFS pod, from repo root or any cwd):

  python helper_scripts/runai/remap_lambdab2_paths.py
  python helper_scripts/runai/remap_lambdab2_paths.py --apply
  python helper_scripts/runai/remap_lambdab2_paths.py --apply \\
      --old /home/romanus/dev/vlm_hallucination_mitigation_summer_2026 \\
      --new /home/datalake/romanus/vlm_hallucination
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import project_root  # noqa: E402

DEFAULT_OLD = "/home/romanus/dev/vlm_hallucination_mitigation_summer_2026"

# Prefer these trees; they hold the baked absolute paths that break RunAI.
DEFAULT_ROOTS = (
    "data",
    # augmented JSONLs now live under data/{amber,pope}/
    # (covered by the "data" root above)
    "experiment_artifacts",
)

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "images",
    "val2014",
    "train2014",
    "hf_cache",
    "node_modules",
}

DEFAULT_SUFFIXES = {".json", ".jsonl"}


def _iter_files(roots: list[Path], suffixes: set[str]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in suffixes:
                out.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            if path.suffix.lower() in suffixes:
                out.append(path)
    return sorted(out)


def _remap_text(text: str, old: str, new: str) -> tuple[str, int]:
    if old not in text:
        return text, 0
    count = text.count(old)
    return text.replace(old, new), count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--old",
        default=DEFAULT_OLD,
        help=f"Absolute path prefix to replace (default: {DEFAULT_OLD})",
    )
    ap.add_argument(
        "--new",
        default=None,
        help="Replacement prefix (default: project_root() of this checkout)",
    )
    ap.add_argument(
        "--root",
        action="append",
        dest="roots",
        default=None,
        help="Directory or file to scan (repeatable). Default: data/, "
             "augment/outputs/, experiment_artifacts/",
    )
    ap.add_argument(
        "--ext",
        action="append",
        dest="exts",
        default=None,
        help="Extra file extension to include (e.g. --ext .txt). "
             ".json and .jsonl are always included.",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write changes in place (default is dry-run)",
    )
    args = ap.parse_args()

    old = args.old.rstrip("/")
    new = (args.new or str(project_root())).rstrip("/")
    if old == new:
        print(f"ERROR: --old and --new are identical: {old}", file=sys.stderr)
        return 2

    root = project_root()
    if args.roots:
        roots = [(root / r).resolve() if not Path(r).is_absolute() else Path(r)
                 for r in args.roots]
    else:
        roots = [(root / r).resolve() for r in DEFAULT_ROOTS]

    suffixes = set(DEFAULT_SUFFIXES)
    if args.exts:
        for e in args.exts:
            suffixes.add(e if e.startswith(".") else f".{e}")

    files = _iter_files(roots, suffixes)
    print(f"old prefix : {old}")
    print(f"new prefix : {new}")
    print(f"mode       : {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"roots      : {', '.join(str(r) for r in roots)}")
    print(f"files scan : {len(files)}")

    n_files = 0
    n_repl = 0
    for path in files:
        raw = path.read_bytes()
        # Skip obvious binaries
        if b"\x00" in raw[:8192]:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        new_text, count = _remap_text(text, old, new)
        if count == 0:
            continue
        n_files += 1
        n_repl += count
        rel = path.relative_to(root) if path.is_relative_to(root) else path
        print(f"  {rel}: {count} replacement(s)")
        if args.apply:
            path.write_text(new_text, encoding="utf-8")

    print(f"done: {n_files} file(s), {n_repl} replacement(s)"
          + ("" if args.apply else " (dry-run; pass --apply to write)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
