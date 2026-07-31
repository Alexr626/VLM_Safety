"""demos_850 disjoint-partition textual VTI directions.

Reuses extractor math and activation-cache helpers from ``directions_v2``
without editing that module. Selection policy is ``disjoint_partition`` over
``data/vti/demos_850.jsonl`` blocks of 50 / 100 / 200 / 500.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.extraction import ActivationCache
from src.paths import (
    experiment_artifacts_dir,
    project_root,
    vti_demos_850_partition_path,
    vti_demos_850_path,
)

from .directions_v2 import (
    DIFF_POLARITY,
    DIMENSIONS,
    SIGN_CONVENTION,
    STEER_COMPONENT,
    TOKEN_POLICY,
    _git_commit,
    act_cache_dir,
    demos_content_hash,
    ensure_variant_activation,
    load_demos_v2_rows,
    load_textual_v2_directions,
    obtain_textual_vti_v2_from_stacks,
    prefetch_activations,
    save_textual_v2_directions,
    variant_suffix,
)

PARTITION_SIZES: Tuple[int, ...] = (50, 100, 200, 500)
SELECTION_POLICY = "disjoint_partition"
SLUG_STEM = "demos850"
PARTITION_SEED = 42
QWEN_ACT_CACHE_MAX_PIXELS = 1003520
FIT_LOCUS_GLOBAL = "global"
FIT_LOCUS_PERLAYER = "perlayer"


def build_or_load_partition(
    demos_path: Path,
    partition_path: Path,
    seed: int = PARTITION_SEED,
    sizes: Sequence[int] = PARTITION_SIZES,
) -> dict:
    """Return partition JSON; write if missing. Raise on hash mismatch."""
    demos_path = Path(demos_path)
    partition_path = Path(partition_path)
    h = demos_content_hash(demos_path)
    if sum(sizes) != 850:
        raise ValueError(f"sizes must sum to 850; got {sizes} sum={sum(sizes)}")

    if partition_path.exists():
        obj = json.loads(partition_path.read_text())
        meta = obj.get("_meta") or {}
        if meta.get("content_hash_sha256_16") != h:
            raise RuntimeError(
                f"Partition hash {meta.get('content_hash_sha256_16')} != "
                f"demos hash {h}; rebuild {partition_path}"
            )
        return obj

    rows = load_demos_v2_rows(demos_path)
    if len(rows) != 850:
        raise RuntimeError(f"Expected 850 demos rows; got {len(rows)}")
    ids = sorted(r["id"] for r in rows)
    rng = random.Random(seed)
    rng.shuffle(ids)
    blocks: Dict[str, List[str]] = {}
    cursor = 0
    for n in sizes:
        blocks[str(n)] = ids[cursor : cursor + n]
        cursor += n
    if cursor != len(ids):
        raise RuntimeError(f"Partition did not consume all ids: {cursor}/{len(ids)}")

    try:
        demos_rel = str(demos_path.relative_to(project_root()))
    except ValueError:
        demos_rel = str(demos_path)

    obj = {
        "_meta": {
            "demos_file": demos_rel,
            "content_hash_sha256_16": h,
            "seed": seed,
            "selection_policy": SELECTION_POLICY,
            "sizes": list(sizes),
            "n_ids": len(ids),
            "created": datetime.now().strftime("%Y-%m-%d"),
        },
        "blocks": blocks,
    }
    partition_path.parent.mkdir(parents=True, exist_ok=True)
    partition_path.write_text(json.dumps(obj, indent=2) + "\n")
    return obj


def select_block_demos(
    rows_by_id: Dict[str, dict],
    block_ids: Sequence[str],
    dimension: str,
) -> List[dict]:
    """Return flat demos for every id in ``block_ids``; raise on missing caption."""
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}")
    used: List[dict] = []
    for rid in block_ids:
        row = rows_by_id.get(rid)
        if row is None:
            raise RuntimeError(f"Partition id {rid} missing from demos file")
        value = row.get("value")
        hv = (row.get("h_values") or {}).get(dimension)
        if not (value and str(value).strip() and hv and str(hv).strip()):
            raise RuntimeError(
                f"Missing value or h_values[{dimension}] for id={rid}"
            )
        used.append({
            "id": row["id"],
            "image": row["image"],
            "question": row.get("question", "Describe this image in detail."),
            "value": value,
            "h_value": hv,
        })
    return used


def partition_slug(
    demos_hash: str,
    dimension: str,
    num_demos: int,
    seed: int = 42,
    rank: int = 2,
    fit_locus: str = FIT_LOCUS_GLOBAL,
) -> str:
    base = (
        f"{SLUG_STEM}_{demos_hash[:8]}_{dimension}_nd{num_demos}_"
        f"s{seed}_r{rank}_partition"
    )
    if fit_locus == FIT_LOCUS_GLOBAL:
        return base
    if fit_locus == FIT_LOCUS_PERLAYER:
        return f"{base}_perlayer"
    raise ValueError(
        f"fit_locus must be one of ('global', 'perlayer'); got {fit_locus!r}"
    )


def partition_cache_dir(model_short: str, slug: str) -> Path:
    return experiment_artifacts_dir("vti", model_short) / "textual_v2" / slug


def resolve_act_cache_dir(
    model_short: str,
    act_cache_override: Optional[Path] = None,
) -> Path:
    if act_cache_override is not None:
        return Path(act_cache_override)
    return act_cache_dir(model_short)


def check_partition_integrity(
    demos_path: Path,
    partition_obj: dict,
    sizes: Sequence[int] = PARTITION_SIZES,
) -> None:
    """Check 0.5 — raise on any failure."""
    rows = load_demos_v2_rows(demos_path)
    all_ids = [r["id"] for r in rows]
    if len(all_ids) != 850 or len(set(all_ids)) != 850:
        raise RuntimeError(
            f"demos_850 integrity: n={len(all_ids)} unique={len(set(all_ids))}"
        )
    h = demos_content_hash(demos_path)
    meta = partition_obj.get("_meta") or {}
    if meta.get("content_hash_sha256_16") != h:
        raise RuntimeError("partition content_hash_sha256_16 mismatch")
    blocks = partition_obj["blocks"]
    seen = set()
    for n in sizes:
        key = str(n)
        ids = blocks[key]
        if len(ids) != n:
            raise RuntimeError(f"block {n} has len {len(ids)}")
        if len(set(ids)) != n:
            raise RuntimeError(f"block {n} has duplicate ids")
        overlap = seen.intersection(ids)
        if overlap:
            raise RuntimeError(f"block {n} overlaps prior blocks: {list(overlap)[:5]}")
        seen.update(ids)
    if seen != set(all_ids):
        raise RuntimeError(
            f"union != demos ids: missing={len(set(all_ids)-seen)} "
            f"extra={len(seen-set(all_ids))}"
        )


def check_cache_fidelity(
    wrapper,
    model_short: str,
    demos_path: Path,
    *,
    act_cache_override: Optional[Path] = None,
    n_ids: int = 3,
    variants: Sequence[str] = ("value", "all"),
    min_cosine: float = 0.9999,
    max_rel_delta: float = 1e-3,
) -> dict:
    """Re-forward already-cached (id, variant) pairs; return pass/fail report.

    Check 0.4. Does not write to the cache.
    """
    cache_path = resolve_act_cache_dir(model_short, act_cache_override)
    cache = ActivationCache(str(cache_path))
    rows = load_demos_v2_rows(demos_path)
    rows_by_id = {r["id"]: r for r in rows}

    # Prefer ids that already have all requested variants cached.
    candidates = []
    for r in rows:
        rid = r["id"]
        if all(cache.exists(rid, suffix=variant_suffix(v)) for v in variants):
            candidates.append(rid)
        if len(candidates) >= n_ids:
            break
    if len(candidates) < n_ids:
        return {
            "pass": False,
            "reason": f"only {len(candidates)} cached ids with variants {list(variants)}",
            "cache_dir": str(cache_path),
            "ids": candidates,
        }

    from .directions import _demo_prompt, load_demo_image
    from src.extraction import get_last_token_activations
    import torch

    per_check = []
    all_ok = True
    for rid in candidates[:n_ids]:
        row = rows_by_id[rid]
        demo = {
            "id": rid,
            "image": row["image"],
            "question": row.get("question", "Describe this image in detail."),
            "value": row["value"],
        }
        image = load_demo_image(demo)
        for variant in variants:
            if variant == "value":
                caption = row["value"]
            else:
                caption = row["h_values"][variant]
            cached = cache.load(rid, suffix=variant_suffix(variant))
            cached_stack = np.stack(
                [np.asarray(cached[i], dtype=np.float32) for i in range(max(cached) + 1)],
                axis=0,
            )
            text = _demo_prompt(demo, caption)
            with torch.no_grad():
                hidden_states, _, _ = wrapper.forward_vl(image, text)
            fresh = get_last_token_activations(hidden_states)
            fresh_stack = np.stack(
                [np.asarray(fresh[i], dtype=np.float32) for i in range(max(fresh) + 1)],
                axis=0,
            )
            if fresh_stack.shape != cached_stack.shape:
                all_ok = False
                per_check.append({
                    "id": rid, "variant": variant, "pass": False,
                    "reason": f"shape {fresh_stack.shape} != {cached_stack.shape}",
                })
                continue
            layer_ok = True
            layer_stats = []
            for li in range(fresh_stack.shape[0]):
                a = cached_stack[li]
                b = fresh_stack[li]
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                cos = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0
                rel = float(np.max(np.abs(a - b)) / na) if na > 0 else float("inf")
                ok = cos >= min_cosine and rel <= max_rel_delta
                if not ok:
                    layer_ok = False
                layer_stats.append({"layer": li, "cosine": cos, "rel_max_abs": rel, "ok": ok})
            if not layer_ok:
                all_ok = False
            per_check.append({
                "id": rid,
                "variant": variant,
                "pass": layer_ok,
                "layers": layer_stats,
            })

    return {
        "pass": all_ok,
        "cache_dir": str(cache_path),
        "ids": candidates[:n_ids],
        "variants": list(variants),
        "min_cosine": min_cosine,
        "max_rel_delta": max_rel_delta,
        "checks": per_check,
    }


def compute_or_load_partition_directions(
    wrapper,
    model_short: str,
    *,
    dimension: str,
    num_demos: int,
    rank: int = 2,
    seed: int = 42,
    demos_path: Optional[Path] = None,
    partition_path: Optional[Path] = None,
    act_cache_override: Optional[Path] = None,
    max_pixels: Optional[int] = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """Return decoder-layer directions ``(num_layers, hidden_dim)`` float32."""
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
    slug = partition_slug(demos_hash, dimension, num_demos, seed=seed, rank=rank)
    cache_dir = partition_cache_dir(model_short, slug)
    if (cache_dir / "directions.npz").exists() and not force_recompute:
        directions, meta = load_textual_v2_directions(cache_dir)
        if directions.shape != (wrapper.num_layers, wrapper.hidden_dim):
            raise RuntimeError(
                f"Cached partition directions shape {directions.shape} != "
                f"({wrapper.num_layers}, {wrapper.hidden_dim})"
            )
        print(f"  [partition] loaded {slug}  n_pairs={meta.get('n_pairs')}")
        return directions

    part = build_or_load_partition(demos_path, partition_path, seed=seed)
    check_partition_integrity(demos_path, part)
    block_ids = part["blocks"][str(num_demos)]
    rows = load_demos_v2_rows(demos_path)
    rows_by_id = {r["id"]: r for r in rows}
    flat_demos = select_block_demos(rows_by_id, block_ids, dimension)

    act_dir = resolve_act_cache_dir(model_short, act_cache_override)
    # Prefetch into the shared (or override) cache by temporarily pointing
    # ensure_variant_activation via a local ActivationCache — reuse
    # prefetch_activations only when using the default act_cache_dir.
    if act_cache_override is None:
        prefetch_activations(
            wrapper, flat_demos, ["value", dimension], model_short,
            rows_by_id=rows_by_id,
        )
        cache = ActivationCache(str(act_dir))
    else:
        cache = ActivationCache(str(act_dir))
        for demo in flat_demos:
            row = rows_by_id[demo["id"]]
            ensure_variant_activation(
                wrapper, cache, demo, "value", demo["value"],
            )
            ensure_variant_activation(
                wrapper, cache, demo, dimension, demo["h_value"],
            )

    h_stacks, v_stacks = [], []
    for demo in flat_demos:
        h_stacks.append(
            ensure_variant_activation(
                wrapper, cache, demo, dimension, demo["h_value"],
            )
        )
        v_stacks.append(
            ensure_variant_activation(
                wrapper, cache, demo, "value", demo["value"],
            )
        )

    full, diag = obtain_textual_vti_v2_from_stacks(h_stacks, v_stacks, rank=rank)
    directions = full[1:].detach().cpu().float().numpy().astype(np.float32)
    if directions.shape != (wrapper.num_layers, wrapper.hidden_dim):
        raise RuntimeError(
            f"Direction shape {directions.shape} != "
            f"({wrapper.num_layers}, {wrapper.hidden_dim})"
        )

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
        "rank": rank,
        "steer_component": STEER_COMPONENT,
        "steer_reconstruction": "live_pc1_plus_mean",
        "explained_variance_ratio": diag["explained_variance_ratio"],
        "pc1_explained_variance": diag["pc1_explained_variance"],
        "pc2_explained_variance": diag["pc2_explained_variance"],
        "pc1_layer_norms": diag["pc1_layer_norms"],
        "pc2_layer_norms": diag["pc2_layer_norms"],
        "direction_layer_norms": diag["direction_layer_norms"],
        "mean_diff_layer_norms_mean_over_demos": diag[
            "mean_diff_layer_norms_mean_over_demos"
        ],
        "model_short": model_short,
        "slug": slug,
        "max_pixels": max_pixels,
        "act_cache_dir": str(act_dir),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "git_commit": _git_commit(),
    }
    save_textual_v2_directions(
        directions,
        meta,
        cache_dir,
        components=diag["components"],
        pca_mean=diag["pca_mean"],
        n_layers_plus=full.shape[0],
        hidden_dim=full.shape[1],
    )
    print(
        f"  [partition] saved {cache_dir}  "
        f"PC1_evr={diag['pc1_explained_variance']:.4f}  "
        f"n_pairs={len(flat_demos)}"
    )
    return directions


def extract_partition_grid(
    wrapper,
    model_short: str,
    dimension: str,
    sizes: Sequence[int] = PARTITION_SIZES,
    **kw,
) -> Dict[int, Path]:
    """Prefetch (value, dimension) across all block ids, then fit each block."""
    demos_path = Path(kw.get("demos_path") or vti_demos_850_path())
    partition_path = Path(kw.get("partition_path") or vti_demos_850_partition_path())
    seed = int(kw.get("seed", PARTITION_SEED))
    rank = int(kw.get("rank", 2))
    act_cache_override = kw.get("act_cache_override")
    max_pixels = kw.get("max_pixels")
    force_recompute = bool(kw.get("force_recompute", False))

    part = build_or_load_partition(demos_path, partition_path, seed=seed, sizes=sizes)
    check_partition_integrity(demos_path, part, sizes=sizes)
    rows = load_demos_v2_rows(demos_path)
    rows_by_id = {r["id"]: r for r in rows}

    # Prefetch all ids that appear in any requested block for this dimension.
    all_block_ids: List[str] = []
    for n in sizes:
        all_block_ids.extend(part["blocks"][str(n)])
    flat_all = select_block_demos(rows_by_id, all_block_ids, dimension)

    act_dir = resolve_act_cache_dir(model_short, act_cache_override)
    if act_cache_override is None:
        prefetch_activations(
            wrapper, flat_all, ["value", dimension], model_short,
            rows_by_id=rows_by_id,
        )
    else:
        cache = ActivationCache(str(act_dir))
        for demo in flat_all:
            ensure_variant_activation(
                wrapper, cache, demo, "value", demo["value"],
            )
            ensure_variant_activation(
                wrapper, cache, demo, dimension, demo["h_value"],
            )

    out: Dict[int, Path] = {}
    for n in sizes:
        compute_or_load_partition_directions(
            wrapper,
            model_short,
            dimension=dimension,
            num_demos=n,
            rank=rank,
            seed=seed,
            demos_path=demos_path,
            partition_path=partition_path,
            act_cache_override=act_cache_override,
            max_pixels=max_pixels,
            force_recompute=force_recompute,
        )
        h = demos_content_hash(demos_path)
        out[n] = partition_cache_dir(
            model_short, partition_slug(h, dimension, n, seed=seed, rank=rank),
        )
    return out
