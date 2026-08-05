"""Raw mean-difference textual VTI directions over demos_850 partitions.

CPU-only. Reads cached last-token stacks from ``textual_v2/_act_cache`` and
never constructs a model wrapper or executes a forward pass. Raises on any
cache miss.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from src.extraction import ActivationCache
from src.paths import (
    experiment_artifacts_dir,
    project_root,
    vti_demos_850_partition_path,
    vti_demos_850_path,
)

from .directions_partition import (
    PARTITION_SIZES,
    SELECTION_POLICY,
    build_or_load_partition,
    check_partition_integrity,
    resolve_act_cache_dir,
    select_block_demos,
)
from .directions_v2 import (
    DIFF_POLARITY,
    DIMENSIONS,
    TOKEN_POLICY,
    _git_commit,
    demos_content_hash,
    load_demos_v2_rows,
    load_textual_v2_directions,
    save_textual_v2_directions,
    variant_suffix,
)

STEER_RECONSTRUCTION = "raw_mean_difference"
SIGN_CONVENTION = (
    "mean over demos of (value - h_value); no PCA, no component selection, no sign flip"
)
SLUG_STEM = "demos850"


def meandiff_partition_slug(
    demos_hash: str,
    dimension: str,
    num_demos: int,
    seed: int = 42,
) -> str:
    return (
        f"{SLUG_STEM}_{demos_hash[:8]}_{dimension}_nd{num_demos}_"
        f"s{seed}_meandiff_partition"
    )


def meandiff_cache_dir(model_short: str, slug: str) -> Path:
    return experiment_artifacts_dir("vti", model_short) / "textual_v2" / slug


def load_cached_stack(
    cache: ActivationCache,
    demo_id: str,
    variant: str,
) -> np.ndarray:
    """Load ``(num_layers+1, hidden_dim)`` float32 stack; raise on miss."""
    suffix = variant_suffix(variant)
    cached = cache.load_or_none(demo_id, suffix=suffix)
    if cached is None:
        raise FileNotFoundError(
            f"Activation cache miss for demo_id={demo_id!r} variant={variant!r} "
            f"(suffix={suffix}) under {cache.cache_dir}"
        )
    n = max(cached.keys()) + 1
    rows = [np.asarray(cached[i], dtype=np.float32) for i in range(n)]
    return np.stack(rows, axis=0).astype(np.float32)


def obtain_textual_meandiff_from_stacks(
    h_stacks: Sequence[np.ndarray],
    value_stacks: Sequence[np.ndarray],
) -> Tuple[np.ndarray, dict]:
    """Mean over demos of ``value_stack - h_stack``.

    Returns full direction ``(num_layers+1, hidden_dim)`` float32 plus
    diagnostics. Diff polarity is ``value_minus_h_value``.
    """
    if len(h_stacks) != len(value_stacks) or not h_stacks:
        raise ValueError("Need non-empty paired h/value stacks")
    n_layers_plus, hidden = h_stacks[0].shape
    diffs = []
    mean_diff_norms = []
    for h_act, v_act in zip(h_stacks, value_stacks):
        h_act = np.asarray(h_act, dtype=np.float32)
        v_act = np.asarray(v_act, dtype=np.float32)
        if h_act.shape != (n_layers_plus, hidden) or v_act.shape != (n_layers_plus, hidden):
            raise ValueError(
                f"Stack shape mismatch: h={h_act.shape} v={v_act.shape} "
                f"expected=({n_layers_plus}, {hidden})"
            )
        per_layer = v_act - h_act
        diffs.append(per_layer)
        mean_diff_norms.append(np.linalg.norm(per_layer, axis=-1))
    full = np.mean(np.stack(diffs, axis=0), axis=0).astype(np.float32)
    mean_diff_norms_arr = np.stack(mean_diff_norms, axis=0).astype(np.float32)
    diag = {
        "n_pairs": len(diffs),
        "direction_layer_norms": np.linalg.norm(full, axis=-1).tolist(),
        "mean_diff_layer_norms_mean_over_demos": mean_diff_norms_arr.mean(axis=0).tolist(),
        "flat_dim": int(n_layers_plus * hidden),
    }
    return full, diag


def compute_or_load_meandiff_directions(
    model_short: str,
    *,
    dimension: str = "all",
    num_demos: int,
    seed: int = 42,
    demos_path: Optional[Path] = None,
    partition_path: Optional[Path] = None,
    act_cache_override: Optional[Path] = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """Return decoder-layer directions ``(num_layers, hidden_dim)`` float32.

    Zero forward passes. Raises ``FileNotFoundError`` on any activation-cache miss.
    """
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}")
    if num_demos not in PARTITION_SIZES:
        raise ValueError(f"num_demos must be one of {PARTITION_SIZES}")

    demos_path = Path(demos_path) if demos_path is not None else vti_demos_850_path()
    partition_path = (
        Path(partition_path)
        if partition_path is not None
        else vti_demos_850_partition_path()
    )
    demos_hash = demos_content_hash(demos_path)
    slug = meandiff_partition_slug(demos_hash, dimension, num_demos, seed=seed)
    cache_dir = meandiff_cache_dir(model_short, slug)
    if (cache_dir / "directions.npz").exists() and not force_recompute:
        directions, meta = load_textual_v2_directions(cache_dir)
        print(
            f"  [meandiff] loaded {slug}  n_pairs={meta.get('n_pairs')}  "
            f"shape={directions.shape}"
        )
        return directions

    part = build_or_load_partition(demos_path, partition_path, seed=seed)
    check_partition_integrity(demos_path, part)
    block_ids = part["blocks"][str(num_demos)]
    rows = load_demos_v2_rows(demos_path)
    rows_by_id = {r["id"]: r for r in rows}
    flat_demos = select_block_demos(rows_by_id, block_ids, dimension)

    act_dir = resolve_act_cache_dir(model_short, act_cache_override)
    cache = ActivationCache(str(act_dir))

    h_stacks, v_stacks = [], []
    for demo in flat_demos:
        h_stacks.append(load_cached_stack(cache, demo["id"], dimension))
        v_stacks.append(load_cached_stack(cache, demo["id"], "value"))

    full, diag = obtain_textual_meandiff_from_stacks(h_stacks, v_stacks)
    directions = full[1:].astype(np.float32)

    try:
        demos_rel = str(demos_path.relative_to(project_root()))
    except ValueError:
        demos_rel = str(demos_path)
    try:
        part_rel = str(partition_path.relative_to(project_root()))
    except ValueError:
        part_rel = str(partition_path)

    question = flat_demos[0].get("question", "Describe this image in detail.")
    meta = {
        "demos_path": str(demos_path),
        "demos_file": demos_path.name,
        "content_hash_sha256_16": demos_hash,
        "partition_file": part_rel,
        "partition_content_hash": part["_meta"]["content_hash_sha256_16"],
        "block_size": num_demos,
        "dimension": dimension,
        "num_demos_requested": num_demos,
        "n_pairs": len(flat_demos),
        "ids_used": list(block_ids),
        "skipped_ids": [],
        "question": question,
        "token_policy": TOKEN_POLICY,
        "diff_polarity": DIFF_POLARITY,
        "sign_convention": SIGN_CONVENTION,
        "selection_policy": SELECTION_POLICY,
        "seed": seed,
        "steer_reconstruction": STEER_RECONSTRUCTION,
        "rank": None,
        "steer_component": None,
        "direction_layer_norms": diag["direction_layer_norms"],
        "mean_diff_layer_norms_mean_over_demos": diag[
            "mean_diff_layer_norms_mean_over_demos"
        ],
        "model_short": model_short,
        "slug": slug,
        "act_cache_dir": str(act_dir),
        "forwards_executed": 0,
        "act_cache_read_only": True,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "git_commit": _git_commit(),
        "demos_rel": demos_rel,
    }
    save_textual_v2_directions(directions, meta, cache_dir, components=None)
    print(
        f"  [meandiff] saved {cache_dir}  shape={directions.shape}  "
        f"n_pairs={len(flat_demos)}  forwards_executed=0"
    )
    return directions


__all__ = [
    "SELECTION_POLICY",
    "STEER_RECONSTRUCTION",
    "SIGN_CONVENTION",
    "meandiff_partition_slug",
    "meandiff_cache_dir",
    "load_cached_stack",
    "obtain_textual_meandiff_from_stacks",
    "compute_or_load_meandiff_directions",
]
