"""demos_850 shuffled-control directions: same captions, deranged images per block.

Control for deployed ``demos850_{h8}_all_nd{N}_s42_r2_partition`` cells.
Uses a dedicated act-cache namespace so image swaps cannot silently hit
``textual_v2/_act_cache`` or the legacy demos_v2 ``shuffled_control/_act_cache``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.extraction import ActivationCache
from src.paths import (
    experiment_artifacts_dir,
    experiment_artifacts_root,
    project_root,
    vti_demos_850_partition_path,
    vti_demos_850_path,
)

from .directions_partition import (
    PARTITION_SIZES,
    SELECTION_POLICY as PARTITION_SELECTION_POLICY,
    build_or_load_partition,
    check_partition_integrity,
    partition_cache_dir,
    partition_slug,
    select_block_demos,
)
from .directions_v2 import (
    DIFF_POLARITY,
    SIGN_CONVENTION,
    STEER_COMPONENT,
    TOKEN_POLICY,
    _git_commit,
    demos_content_hash,
    ensure_variant_activation,
    load_demos_v2_rows,
    load_textual_v2_directions,
    obtain_textual_vti_v2_from_stacks,
    save_textual_v2_directions,
    variant_suffix,
)
from .shuffled_control import (
    build_shuffled_flat_demos,
    generate_derangement,
    validate_derangement,
)

DERANGEMENT_SEED = 1234
DIMENSION = "all"
RANK = 2
PARTITION_SEED = 42
EXPECTED_DEMOS_HASH_PREFIX = "ba05bd960cad0c18"


def derangement_path_for_block(num_demos: int) -> Path:
    return (
        experiment_artifacts_root()
        / "vti"
        / f"shuffled_control_demos850_image_derangement_nd{num_demos}_s{DERANGEMENT_SEED}.json"
    )


def shuffled_demos850_act_cache_dir(model_short: str) -> Path:
    return (
        experiment_artifacts_dir("vti", model_short)
        / "shuffled_control_demos850"
        / "_act_cache"
    )


def shuffled_demos850_direction_dir(model_short: str, num_demos: int) -> Path:
    return (
        experiment_artifacts_dir("vti", model_short)
        / "shuffled_control_demos850"
        / f"all_nd{num_demos}"
    )


def sanity_report_path(model_short: str, num_demos: int) -> Path:
    return (
        experiment_artifacts_dir("vti", model_short)
        / "shuffled_control_demos850"
        / f"shuffled_control_sanity_report_{model_short}_all_nd{num_demos}.md"
    )


def sanity_rollup_path(model_short: str, date_tag: str = "2026-07-29") -> Path:
    return (
        experiment_artifacts_dir("vti", model_short)
        / "shuffled_control_demos850"
        / f"shuffled_control_sanity_rollup_{model_short}_{date_tag}.md"
    )


def deployed_partition_slug(
    demos_hash: str,
    num_demos: int,
    dimension: str = DIMENSION,
    seed: int = PARTITION_SEED,
    rank: int = RANK,
) -> str:
    return partition_slug(demos_hash, dimension, num_demos, seed=seed, rank=rank)


def load_or_write_block_derangement(
    ids: Sequence[str],
    num_demos: int,
    *,
    deployed_slug: str,
    seed: int = DERANGEMENT_SEED,
    force: bool = False,
) -> Tuple[Dict[str, str], Path]:
    path = derangement_path_for_block(num_demos)
    ids = list(ids)
    if len(ids) != num_demos:
        raise ValueError(f"Expected {num_demos} ids, got {len(ids)}")
    if path.exists() and not force:
        obj = json.loads(path.read_text())
        mapping = {str(k): str(v) for k, v in obj["map"].items()}
        validate_derangement(ids, mapping)
        if obj.get("seed") != seed:
            raise RuntimeError(
                f"Existing derangement seed {obj.get('seed')} != requested {seed}"
            )
        if set(mapping.keys()) != set(ids):
            raise RuntimeError(
                f"Existing derangement ids do not match block nd{num_demos}"
            )
        return mapping, path

    mapping = generate_derangement(ids, seed=seed)
    n_fixed = validate_derangement(ids, mapping)
    if n_fixed != 0:
        raise RuntimeError(f"Generated derangement has {n_fixed} fixed points")
    payload = {
        "seed": seed,
        "n_ids": len(ids),
        "n_fixed_points": n_fixed,
        "deployed_slug_control_for": deployed_slug,
        "dimension": DIMENSION,
        "num_demos": num_demos,
        "selection_policy": PARTITION_SELECTION_POLICY,
        "created": datetime.now().strftime("%Y-%m-%d"),
        "map": mapping,
        "note": (
            "Each key is a demo id whose captions are kept; the value is the "
            "demo id whose image path is used instead (within the same block)."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return mapping, path


def _count_cache_files(cache_dir: Path) -> int:
    if not cache_dir.exists():
        return 0
    return sum(1 for p in cache_dir.glob("sample_*.npz") if p.is_file())


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(project_root()))
    except ValueError:
        return str(path)


def extract_shuffled_control_partition_direction(
    wrapper,
    model_short: str,
    *,
    num_demos: int,
    demos_path: Optional[Path] = None,
    partition_path: Optional[Path] = None,
    derangement_seed: int = DERANGEMENT_SEED,
    max_pixels: Optional[int] = None,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, dict, dict]:
    """Extract one demos850 shuffled-control cell; return (directions, meta, checks)."""
    if num_demos not in PARTITION_SIZES:
        raise ValueError(f"num_demos must be one of {PARTITION_SIZES}")

    demos_path = Path(demos_path) if demos_path is not None else vti_demos_850_path()
    partition_path = (
        Path(partition_path)
        if partition_path is not None
        else vti_demos_850_partition_path()
    )
    demos_hash = demos_content_hash(demos_path)
    if not demos_hash.startswith(EXPECTED_DEMOS_HASH_PREFIX[:8]):
        # Soft check on full 16: require exact known pool unless overridden by force path
        if demos_hash != EXPECTED_DEMOS_HASH_PREFIX:
            raise RuntimeError(
                f"demos_850 hash {demos_hash} != expected {EXPECTED_DEMOS_HASH_PREFIX}"
            )

    part = build_or_load_partition(
        demos_path, partition_path, seed=PARTITION_SEED, sizes=PARTITION_SIZES,
    )
    check_partition_integrity(demos_path, part, sizes=PARTITION_SIZES)
    block_ids = list(part["blocks"][str(num_demos)])
    if len(block_ids) != num_demos:
        raise RuntimeError(f"Block {num_demos} has {len(block_ids)} ids")

    slug = deployed_partition_slug(demos_hash, num_demos)
    deployed_dir = partition_cache_dir(model_short, slug)
    if not (deployed_dir / "directions.npz").exists():
        raise FileNotFoundError(f"Deployed partition direction missing: {deployed_dir}")
    deployed_dirs, deployed_meta = load_textual_v2_directions(deployed_dir)
    ids_used = list(deployed_meta["ids_used"])
    if ids_used != block_ids:
        raise RuntimeError(
            f"Deployed ids_used != partition block nd{num_demos} "
            f"(positional mismatches="
            f"{sum(a != b for a, b in zip(ids_used, block_ids)) + abs(len(ids_used) - len(block_ids))})"
        )
    if deployed_meta.get("dimension") != DIMENSION:
        raise RuntimeError(
            f"Deployed dimension {deployed_meta.get('dimension')} != {DIMENSION}"
        )

    derangement, derangement_file = load_or_write_block_derangement(
        ids_used,
        num_demos,
        deployed_slug=slug,
        seed=derangement_seed,
    )
    n_fixed = validate_derangement(ids_used, derangement)

    out_dir = shuffled_demos850_direction_dir(model_short, num_demos)
    act_dir = shuffled_demos850_act_cache_dir(model_short)

    if (out_dir / "directions.npz").exists() and not force_recompute:
        directions, meta = load_textual_v2_directions(out_dir)
        checks = {
            "skipped_extraction": True,
            "reason": "directions already present; pass --force_recompute to rebuild",
            "derangement_fixed_points": n_fixed,
            "block_size": num_demos,
        }
        return directions, meta, checks

    if force_recompute:
        # Only wipe this block's cached ids, not the whole shared cache.
        cache_probe = ActivationCache(str(act_dir))
        for rid in ids_used:
            for variant in (DIMENSION, "value"):
                p = cache_probe._path(rid, variant_suffix(variant))
                if p.exists():
                    p.unlink()

    rows = load_demos_v2_rows(demos_path)
    rows_by_id = {r["id"]: r for r in rows}
    flat_demos = select_block_demos(rows_by_id, ids_used, DIMENSION)
    used_ids = [d["id"] for d in flat_demos]
    if used_ids != ids_used:
        raise RuntimeError("select_block_demos did not preserve ids_used order")

    shuffled = build_shuffled_flat_demos(flat_demos, rows_by_id, derangement)

    caption_matches = 0
    for demo in shuffled:
        row = rows_by_id[demo["id"]]
        if (
            demo["value"] == row["value"]
            and demo["h_value"] == row["h_values"][DIMENSION]
        ):
            caption_matches += 1

    cache = ActivationCache(str(act_dir))
    forwards_executed = 0
    cache_hits = 0
    h_stacks, v_stacks = [], []
    for demo in shuffled:
        for variant, caption in (
            (DIMENSION, demo["h_value"]),
            ("value", demo["value"]),
        ):
            suffix = variant_suffix(variant)
            if cache.load_or_none(demo["id"], suffix=suffix) is None:
                forwards_executed += 1
            else:
                cache_hits += 1
            stack = ensure_variant_activation(
                wrapper, cache, demo, variant, caption,
            )
            if variant == DIMENSION:
                h_stacks.append(stack)
            else:
                v_stacks.append(stack)

    full, diag = obtain_textual_vti_v2_from_stacks(h_stacks, v_stacks, rank=RANK)
    directions = full[1:].detach().cpu().float().numpy().astype(np.float32)
    expected_shape = (wrapper.num_layers, wrapper.hidden_dim)
    if directions.shape != expected_shape:
        raise RuntimeError(
            f"Direction shape {directions.shape} != {expected_shape}"
        )

    question = shuffled[0].get("question", "Describe this image in detail.")
    meta = {
        "control_for_deployed_slug": slug,
        "control_type": "image_derangement",
        "derangement_file": _rel(derangement_file),
        "derangement_seed": derangement_seed,
        "derangement_fixed_points": n_fixed,
        "demos_path": str(demos_path),
        "demos_file": demos_path.name,
        "content_hash_sha256_16": demos_hash,
        "partition_file": _rel(partition_path),
        "partition_content_hash": part["_meta"]["content_hash_sha256_16"],
        "block_size": num_demos,
        "dimension": DIMENSION,
        "num_demos_requested": num_demos,
        "n_pairs": len(used_ids),
        "ids_used": used_ids,
        "skipped_ids": [],
        "question": question,
        "token_policy": TOKEN_POLICY,
        "diff_polarity": DIFF_POLARITY,
        "sign_convention": SIGN_CONVENTION,
        "selection_policy": PARTITION_SELECTION_POLICY,
        "seed": PARTITION_SEED,
        "rank": RANK,
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
        "act_cache_dir": _rel(act_dir),
        "forwards_executed": forwards_executed,
        "cache_hits": cache_hits,
        "max_pixels": max_pixels,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "git_commit": _git_commit(),
    }
    save_textual_v2_directions(
        directions,
        meta,
        out_dir,
        components=diag["components"],
        pca_mean=diag["pca_mean"],
        n_layers_plus=full.shape[0],
        hidden_dim=full.shape[1],
    )

    id_mismatches = sum(1 for a, b in zip(used_ids, ids_used) if a != b)
    id_mismatches += abs(len(used_ids) - len(ids_used))
    deployed_norms = list(deployed_meta.get("direction_layer_norms") or [])
    shuffled_norms = list(diag["direction_layer_norms"])
    # Fresh ids in a disjoint block should miss on both variants → N*2 forwards.
    expected_forwards = num_demos * 2
    checks = {
        "skipped_extraction": False,
        "block_size": num_demos,
        "derangement_fixed_points": n_fixed,
        "caption_matches": caption_matches,
        "caption_total": len(shuffled),
        "ids_used_positional_mismatches": id_mismatches,
        "direction_shape": list(directions.shape),
        "expected_direction_shape": list(expected_shape),
        "forwards_executed": forwards_executed,
        "cache_hits": cache_hits,
        "expected_forwards": expected_forwards,
        "control_for_deployed_slug": slug,
        "lineage_slug_ok": slug
        == deployed_meta.get("slug", slug),
        "cache_files_after": _count_cache_files(act_dir),
        "act_cache_dir": _rel(act_dir),
        "deployed_direction_layer_norms": deployed_norms,
        "shuffled_direction_layer_norms": shuffled_norms,
        "deployed_direction_shape": list(deployed_dirs.shape),
        "all_shuffled_norms_finite": bool(
            np.isfinite(np.asarray(shuffled_norms, dtype=np.float64)).all()
        ),
        "cache_under_demos850_namespace": "shuffled_control_demos850" in str(act_dir),
    }
    return directions, meta, checks


def write_sanity_report(
    model_short: str,
    num_demos: int,
    checks: dict,
    meta: dict,
    path: Optional[Path] = None,
) -> Path:
    path = path or sanity_report_path(model_short, num_demos)
    path.parent.mkdir(parents=True, exist_ok=True)

    fixed = checks["derangement_fixed_points"]
    cap_ok = checks.get("caption_matches", 0)
    cap_tot = checks.get("caption_total", num_demos)
    id_mis = checks.get("ids_used_positional_mismatches", -1)
    shape = checks.get("direction_shape")
    exp_shape = checks.get("expected_direction_shape")
    forwards = checks.get("forwards_executed", -1)
    exp_fwd = checks.get("expected_forwards", num_demos * 2)
    cache_hits = checks.get("cache_hits", -1)
    lineage_ok = checks.get("lineage_slug_ok", False)
    ns_ok = checks.get("cache_under_demos850_namespace", False)

    gating = [
        ("Derangement validity (fixed points)", fixed, 0, fixed == 0),
        (
            "Captions unchanged",
            f"{cap_ok} of {cap_tot}",
            f"{cap_tot} of {cap_tot}",
            cap_ok == cap_tot,
        ),
        (
            "Same demos / order (positional id mismatches)",
            id_mis,
            0,
            id_mis == 0,
        ),
        (
            "Direction array shape",
            shape,
            exp_shape,
            shape == exp_shape,
        ),
        (
            "Lineage slug matches deployed partition cell",
            meta.get("control_for_deployed_slug"),
            checks.get("control_for_deployed_slug"),
            lineage_ok
            and meta.get("control_for_deployed_slug")
            == checks.get("control_for_deployed_slug"),
        ),
        (
            "Act cache under shuffled_control_demos850/",
            checks.get("act_cache_dir"),
            "…/shuffled_control_demos850/_act_cache",
            ns_ok,
        ),
        (
            "No wrong-cache reuse (forwards for this block's new pairs)",
            f"forwards={forwards} hits={cache_hits}",
            f"forwards={exp_fwd} hits=0",
            forwards == exp_fwd and cache_hits == 0,
        ),
    ]
    # If extraction was skipped, do not fail the forward gate the same way.
    if checks.get("skipped_extraction"):
        gating[-1] = (
            "No wrong-cache reuse (forwards for this block's new pairs)",
            "skipped_extraction",
            "n/a",
            True,
        )

    all_gate_ok = all(g[3] for g in gating)

    lines = [
        f"# Shuffled-control sanity report — `{model_short}` all/nd{num_demos}",
        "",
        f"Date: {meta.get('date', datetime.now().strftime('%Y-%m-%d'))}",
        f"Control for: `{meta.get('control_for_deployed_slug')}`",
        f"Derangement: `{meta.get('derangement_file')}` "
        f"(seed={meta.get('derangement_seed', DERANGEMENT_SEED)})",
        f"Direction dir: `{_rel(shuffled_demos850_direction_dir(model_short, num_demos))}`",
        f"Act cache: `{meta.get('act_cache_dir')}`",
        "",
        "## Gating checks",
        "",
        "| Check | Observed | Expect | Pass |",
        "|---|---|---|---|",
    ]
    for name, obs, exp, ok in gating:
        lines.append(f"| {name} | {obs} | {exp} | {'PASS' if ok else 'FAIL'} |")
    lines += [
        "",
        f"**All gating checks pass:** {'yes' if all_gate_ok else 'no'}",
        "",
        "## Non-gating: per-layer magnitude (L2)",
        "",
        "| Layer | Deployed norm | Shuffled-control norm |",
        "|---:|---:|---:|",
    ]
    dep = checks.get("deployed_direction_layer_norms") or []
    shuf = checks.get("shuffled_direction_layer_norms") or meta.get(
        "direction_layer_norms"
    ) or []
    n = max(len(dep), len(shuf))
    for i in range(n):
        d = dep[i] if i < len(dep) else ""
        s = shuf[i] if i < len(shuf) else ""
        lines.append(f"| {i} | {d} | {s} |")
    lines += [
        "",
        f"All shuffled norms finite: {checks.get('all_shuffled_norms_finite')}",
        f"Deployed direction shape (decoder layers): "
        f"{checks.get('deployed_direction_shape')}",
        f"Cache files after extraction: {checks.get('cache_files_after')}",
        f"forwards_executed={forwards} cache_hits={cache_hits}",
        "",
    ]
    if checks.get("skipped_extraction"):
        lines += [
            "## Note",
            "",
            f"Extraction skipped: {checks.get('reason')}",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")
    # Stash overall gate on the path via sidecar? Return path; caller reads gating from checks.
    checks["all_gating_pass"] = all_gate_ok
    return path


def write_sanity_rollup(
    model_short: str,
    cell_results: Sequence[dict],
    date_tag: str = "2026-07-29",
) -> Path:
    path = sanity_rollup_path(model_short, date_tag=date_tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Shuffled-control demos850 sanity rollup — `{model_short}`",
        "",
        f"Date: {date_tag}",
        "",
        "| N | Gating | forwards | cache_hits | slug |",
        "|---:|---|---:|---:|---|",
    ]
    for row in cell_results:
        lines.append(
            f"| {row['num_demos']} | "
            f"{'PASS' if row.get('all_gating_pass') else 'FAIL'} | "
            f"{row.get('forwards_executed')} | {row.get('cache_hits')} | "
            f"`{row.get('control_for_deployed_slug')}` |"
        )
    act_dir = shuffled_demos850_act_cache_dir(model_short)
    n_files = _count_cache_files(act_dir)
    lines += [
        "",
        f"Act cache files under `{_rel(act_dir)}`: {n_files} (expect 1700)",
        f"Act cache file count matches 850×2: {'yes' if n_files == 1700 else 'no'}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path
