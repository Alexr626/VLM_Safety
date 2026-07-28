"""Shuffled-control textual direction: same captions, deranged images.

Builds a control for the deployed demos_v2 ``all``/nd200 direction by keeping
demo ids, order, and caption pairs fixed while swapping each demo's image via
a seed-1234 derangement. Activations must use a separate cache namespace —
``ActivationCache`` keys omit the image, so pointing at ``textual_v2/_act_cache``
would silently reuse original-image stacks.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.extraction import ActivationCache
from src.paths import (
    experiment_artifacts_dir,
    experiment_artifacts_root,
    project_root,
    vti_demos_v2_path,
)

from .directions_v2 import (
    DEFAULT_ORDER_PATH,
    DIFF_POLARITY,
    SELECTION_POLICY,
    SIGN_CONVENTION,
    STEER_COMPONENT,
    TOKEN_POLICY,
    _git_commit,
    demos_content_hash,
    ensure_variant_activation,
    load_demos_v2_rows,
    load_or_build_master_order,
    load_textual_v2_directions,
    obtain_textual_vti_v2_from_stacks,
    save_textual_v2_directions,
    select_prefix_demos,
    textual_v2_cache_dir,
    variant_suffix,
)

DEPLOYED_SLUG = "demosv2_9a44f4af_all_nd200_s42_r2_prefix"
DERANGEMENT_SEED = 1234
DERANGEMENT_FILENAME = "shuffled_control_image_derangement_nd200_s1234.json"
DIMENSION = "all"
NUM_DEMOS = 200
RANK = 2
ORDER_SEED = 42


def derangement_path() -> Path:
    return experiment_artifacts_root() / "vti" / DERANGEMENT_FILENAME


def shuffled_act_cache_dir(model_short: str) -> Path:
    return experiment_artifacts_dir("vti", model_short) / "shuffled_control" / "_act_cache"


def shuffled_direction_dir(model_short: str) -> Path:
    return experiment_artifacts_dir("vti", model_short) / "shuffled_control" / "all_nd200"


def sanity_report_path(model_short: str) -> Path:
    return (
        experiment_artifacts_dir("vti", model_short)
        / "shuffled_control"
        / f"shuffled_control_sanity_report_{model_short}.md"
    )


def deployed_direction_dir(model_short: str, slug: str = DEPLOYED_SLUG) -> Path:
    return textual_v2_cache_dir(model_short, slug)


def generate_derangement(ids: Sequence[str], seed: int = DERANGEMENT_SEED) -> Dict[str, str]:
    """Map each demo id → source-image demo id with no fixed points."""
    ids = list(ids)
    if len(ids) < 2:
        raise ValueError("Need at least 2 ids for a derangement")
    rng = random.Random(seed)
    # Rejection sampling on Fisher–Yates; expected trials ≈ e for large n.
    while True:
        perm = ids[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(ids, perm)):
            return {a: b for a, b in zip(ids, perm)}


def validate_derangement(ids: Sequence[str], mapping: Dict[str, str]) -> int:
    """Return fixed-point count (expect 0). Raises on incomplete / unknown keys."""
    ids = list(ids)
    if set(mapping.keys()) != set(ids):
        raise ValueError("Derangement keys do not match ids_used")
    if set(mapping.values()) != set(ids):
        raise ValueError("Derangement values are not a permutation of ids_used")
    return sum(1 for i in ids if mapping[i] == i)


def load_or_write_derangement(
    ids: Sequence[str],
    path: Optional[Path] = None,
    seed: int = DERANGEMENT_SEED,
    force: bool = False,
) -> Tuple[Dict[str, str], Path]:
    path = path or derangement_path()
    if path.exists() and not force:
        obj = json.loads(path.read_text())
        mapping = {str(k): str(v) for k, v in obj["map"].items()}
        validate_derangement(ids, mapping)
        if obj.get("seed") != seed:
            raise RuntimeError(
                f"Existing derangement seed {obj.get('seed')} != requested {seed}"
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
        "deployed_slug_control_for": DEPLOYED_SLUG,
        "dimension": DIMENSION,
        "num_demos": NUM_DEMOS,
        "created": datetime.now().strftime("%Y-%m-%d"),
        "map": mapping,
        "note": (
            "Each key is a demo id whose captions are kept; the value is the "
            "demo id whose image path is used instead."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return mapping, path


def build_shuffled_flat_demos(
    flat_demos: Sequence[dict],
    rows_by_id: Dict[str, dict],
    derangement: Dict[str, str],
) -> List[dict]:
    """Same ids/order/captions; override ``image`` from the derangement source demo."""
    out: List[dict] = []
    for demo in flat_demos:
        src_id = derangement[demo["id"]]
        src_row = rows_by_id[src_id]
        out.append({
            "id": demo["id"],
            "image": src_row["image"],
            "question": demo.get("question", "Describe this image in detail."),
            "value": demo["value"],
            "h_value": demo["h_value"],
            "image_source_demo_id": src_id,
        })
    return out


def _count_cache_files(cache_dir: Path) -> int:
    if not cache_dir.exists():
        return 0
    return sum(1 for p in cache_dir.glob("sample_*.npz") if p.is_file())


def extract_shuffled_control_direction(
    wrapper,
    model_short: str,
    *,
    derangement: Dict[str, str],
    derangement_file: Path,
    demos_path: Optional[Path] = None,
    order_path: Path = DEFAULT_ORDER_PATH,
    deployed_slug: str = DEPLOYED_SLUG,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, dict, dict]:
    """Extract shuffled-control direction; return (directions, meta, checks)."""
    demos_path = Path(demos_path) if demos_path is not None else vti_demos_v2_path()
    deployed_dir = deployed_direction_dir(model_short, deployed_slug)
    if not (deployed_dir / "directions.npz").exists():
        raise FileNotFoundError(f"Deployed direction missing: {deployed_dir}")
    deployed_dirs, deployed_meta = load_textual_v2_directions(deployed_dir)
    ids_used = list(deployed_meta["ids_used"])
    if len(ids_used) != NUM_DEMOS:
        raise RuntimeError(f"Expected {NUM_DEMOS} ids_used, got {len(ids_used)}")

    n_fixed = validate_derangement(ids_used, derangement)

    out_dir = shuffled_direction_dir(model_short)
    act_dir = shuffled_act_cache_dir(model_short)
    if (out_dir / "directions.npz").exists() and not force_recompute:
        directions, meta = load_textual_v2_directions(out_dir)
        checks = {
            "skipped_extraction": True,
            "reason": "directions already present; pass force_recompute=True to rebuild",
            "derangement_fixed_points": n_fixed,
        }
        return directions, meta, checks

    if act_dir.exists() and _count_cache_files(act_dir) > 0 and not force_recompute:
        raise RuntimeError(
            f"Shuffled act cache is not empty ({act_dir}); "
            "refuse to risk partial reuse. Pass force_recompute=True to wipe."
        )
    if force_recompute and act_dir.exists():
        for p in act_dir.glob("sample_*.npz"):
            p.unlink()

    order_obj = load_or_build_master_order(
        demos_path, order_path=order_path, seed=ORDER_SEED,
    )
    rows = load_demos_v2_rows(demos_path)
    rows_by_id = {r["id"]: r for r in rows}
    flat_demos, used_ids, skipped_ids = select_prefix_demos(
        rows_by_id, order_obj["ids"], DIMENSION, NUM_DEMOS,
    )
    if used_ids != ids_used:
        raise RuntimeError(
            "Rebuilt prefix ids_used != deployed metadata ids_used "
            f"({sum(a != b for a, b in zip(used_ids, ids_used))} positional mismatches; "
            f"len {len(used_ids)} vs {len(ids_used)})"
        )

    shuffled = build_shuffled_flat_demos(flat_demos, rows_by_id, derangement)

    # Caption-unchanged check against demos file rows.
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
    h_stacks, v_stacks = [], []
    for demo in shuffled:
        for variant, caption in (
            (DIMENSION, demo["h_value"]),
            ("value", demo["value"]),
        ):
            suffix = variant_suffix(variant)
            if cache.load_or_none(demo["id"], suffix=suffix) is None:
                forwards_executed += 1
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

    demos_hash = demos_content_hash(demos_path)
    question = shuffled[0].get("question", "Describe this image in detail.")
    meta = {
        "control_for_deployed_slug": deployed_slug,
        "control_type": "shuffled_image_derangement",
        "derangement_file": str(
            derangement_file.relative_to(project_root())
            if derangement_file.is_relative_to(project_root())
            else derangement_file
        ),
        "derangement_seed": DERANGEMENT_SEED,
        "derangement_fixed_points": n_fixed,
        "demos_path": str(demos_path),
        "demos_file": demos_path.name,
        "content_hash_sha256_16": demos_hash,
        "dimension": DIMENSION,
        "num_demos_requested": NUM_DEMOS,
        "n_pairs": len(used_ids),
        "ids_used": used_ids,
        "skipped_ids": skipped_ids,
        "question": question,
        "token_policy": TOKEN_POLICY,
        "diff_polarity": DIFF_POLARITY,
        "sign_convention": SIGN_CONVENTION,
        "selection_policy": SELECTION_POLICY,
        "seed": ORDER_SEED,
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
        "act_cache_dir": str(
            act_dir.relative_to(project_root())
            if act_dir.is_relative_to(project_root())
            else act_dir
        ),
        "forwards_executed": forwards_executed,
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
    checks = {
        "skipped_extraction": False,
        "derangement_fixed_points": n_fixed,
        "caption_matches": caption_matches,
        "caption_total": len(shuffled),
        "ids_used_positional_mismatches": id_mismatches,
        "direction_shape": list(directions.shape),
        "expected_direction_shape": list(expected_shape),
        "forwards_executed": forwards_executed,
        "expected_forwards": NUM_DEMOS * 2,
        "cache_files_after": _count_cache_files(act_dir),
        "deployed_direction_layer_norms": deployed_norms,
        "shuffled_direction_layer_norms": shuffled_norms,
        "deployed_direction_shape": list(deployed_dirs.shape),
        "all_shuffled_norms_finite": bool(
            np.isfinite(np.asarray(shuffled_norms, dtype=np.float64)).all()
        ),
    }
    return directions, meta, checks


def write_sanity_report(
    model_short: str,
    checks: dict,
    meta: dict,
    path: Optional[Path] = None,
) -> Path:
    path = path or sanity_report_path(model_short)
    path.parent.mkdir(parents=True, exist_ok=True)

    fixed = checks["derangement_fixed_points"]
    cap_ok = checks.get("caption_matches", 0)
    cap_tot = checks.get("caption_total", NUM_DEMOS)
    id_mis = checks.get("ids_used_positional_mismatches", -1)
    shape = checks.get("direction_shape")
    exp_shape = checks.get("expected_direction_shape")
    forwards = checks.get("forwards_executed", -1)
    exp_fwd = checks.get("expected_forwards", NUM_DEMOS * 2)

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
            "No cache reuse (forward passes executed)",
            forwards,
            exp_fwd,
            forwards == exp_fwd,
        ),
    ]
    all_gate_ok = all(g[3] for g in gating)

    lines = [
        f"# Shuffled-control sanity report — `{model_short}`",
        "",
        f"Date: {meta.get('date', datetime.now().strftime('%Y-%m-%d'))}",
        f"Control for: `{meta.get('control_for_deployed_slug', DEPLOYED_SLUG)}`",
        f"Derangement: `{meta.get('derangement_file', DERANGEMENT_FILENAME)}` "
        f"(seed={meta.get('derangement_seed', DERANGEMENT_SEED)})",
        f"Direction dir: `experiment_artifacts/vti/{model_short}/shuffled_control/all_nd200/`",
        f"Act cache: `experiment_artifacts/vti/{model_short}/shuffled_control/_act_cache/`",
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
        "Shuffled-control `direction_layer_norms` alongside the deployed "
        "direction's norms (includes embedding row at index 0; same layout as "
        "deployed `metadata.json`).",
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
    return path
