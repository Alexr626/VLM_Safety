"""Per-layer PCA textual VTI directions for demos_850 (deployed + shuffled control).

Fits one centered PCA per (num_layers+1) activation row, reconstructs as
PC1 + mean, drops the embedding row. CPU-only; reads activation caches with
``ActivationCache.load`` (raises on miss; never forwards).
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.extraction import ActivationCache
from src.paths import (
    experiment_artifacts_root,
    project_root,
    vti_demos_850_partition_path,
    vti_demos_850_path,
)

from .directions_partition import (
    FIT_LOCUS_PERLAYER,
    PARTITION_SEED,
    PARTITION_SIZES,
    SELECTION_POLICY,
    build_or_load_partition,
    check_partition_integrity,
    partition_cache_dir,
    partition_slug,
)
from .directions_v2 import (
    DIFF_POLARITY,
    SIGN_CONVENTION,
    STEER_COMPONENT,
    TOKEN_POLICY,
    _git_commit,
    _stack_from_act_dict,
    act_cache_dir,
    demos_content_hash,
    load_textual_v2_directions,
    save_textual_v2_directions,
    variant_suffix,
)
from .pca import PCA
from .shuffled_control import validate_derangement
from .shuffled_control_partition import (
    DERANGEMENT_SEED,
    DIMENSION,
    RANK,
    load_or_write_block_derangement,
    shuffled_demos850_act_cache_dir,
    shuffled_demos850_direction_dir,
)

FIT_LOCUS_DESCRIPTION = {
    "global": "one PCA over the flattened (num_layers+1)*hidden diff vector",
    "perlayer": (
        "one centered PCA per (num_layers+1) row, on that row's hidden-dim diffs"
    ),
}

EXPECTED_DECODER_SHAPES = {
    "llava-1.5-7b-hf": (32, 4096),
    "qwen2.5-vl-7b-instruct": (28, 3584),
}

EXPECTED_DEMOS_HASH = "ba05bd960cad0c18"


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(project_root()))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perlayer_partition_cache_dir(
    model_short: str,
    slug: str,
    *,
    artifacts_root: Optional[Path] = None,
) -> Path:
    """``experiment_artifacts/vti/{model_short}/textual_v2_perlayer/{slug}``."""
    root = Path(artifacts_root) if artifacts_root is not None else experiment_artifacts_root()
    return root / "vti" / model_short / "textual_v2_perlayer" / slug


def perlayer_shuffled_control_dir(
    model_short: str,
    num_demos: int,
    *,
    artifacts_root: Optional[Path] = None,
) -> Path:
    """``…/shuffled_control_demos850_perlayer/all_nd{N}_perlayer``."""
    root = Path(artifacts_root) if artifacts_root is not None else experiment_artifacts_root()
    return (
        root
        / "vti"
        / model_short
        / "shuffled_control_demos850_perlayer"
        / f"all_nd{num_demos}_perlayer"
    )


def load_stack_readonly(
    cache: ActivationCache,
    demo_id: str,
    variant: str,
) -> torch.Tensor:
    """``(num_layers+1, hidden)`` float32. Raises FileNotFoundError on miss."""
    act = cache.load(demo_id, suffix=variant_suffix(variant))
    return _stack_from_act_dict(act)


def perlayer_pca_fit(
    diffs_3d: torch.Tensor,
    rank: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[float]]]:
    """Fit one centered PCA per layer row.

    Args:
        diffs_3d: ``(num_layers+1, n_pairs, hidden)`` float32.

    Returns:
        components ``(L+1, rank, hidden)``, mean ``(L+1, hidden)``,
        direction ``(L+1, hidden)``, evr_per_layer ``L+1`` lists of length rank.
    """
    if diffs_3d.ndim != 3:
        raise ValueError(f"diffs_3d must be 3-D; got shape {tuple(diffs_3d.shape)}")
    x = diffs_3d.float()
    pca = PCA(n_components=rank).fit(x)
    components = pca.components_  # (L+1, rank, hidden)
    mean = pca.mean_[:, 0, :]  # (L+1, hidden)
    direction = components[:, STEER_COMPONENT, :] + mean

    z = x - pca.mean_
    _, s, _ = torch.linalg.svd(z, full_matrices=False)
    evr_per_layer: List[List[float]] = []
    for li in range(s.shape[0]):
        s2 = (s[li] ** 2).clamp(min=0)
        total = float(s2.sum().item()) or 1.0
        evr_per_layer.append(
            [float(s2[j].item() / total) for j in range(min(rank, int(s2.numel())))]
        )
    return components.cpu(), mean.cpu(), direction.cpu(), evr_per_layer


def obtain_textual_vti_perlayer_from_stacks(
    h_stacks: Sequence[torch.Tensor],
    value_stacks: Sequence[torch.Tensor],
    rank: int = 2,
) -> Tuple[torch.Tensor, dict]:
    """Same return contract as ``obtain_textual_vti_v2_from_stacks``, per-layer fit."""
    if len(h_stacks) != len(value_stacks) or not h_stacks:
        raise ValueError("Need non-empty paired h/value stacks")
    n_layers_plus, hidden = h_stacks[0].shape
    n_pairs = len(h_stacks)
    diffs = torch.empty(
        (n_layers_plus, n_pairs, hidden), dtype=torch.float32,
    )
    mean_diff_norms = []
    for i, (h_act, v_act) in enumerate(zip(h_stacks, value_stacks)):
        if tuple(h_act.shape) != (n_layers_plus, hidden):
            raise ValueError(f"h_stack shape {tuple(h_act.shape)} != {(n_layers_plus, hidden)}")
        if tuple(v_act.shape) != (n_layers_plus, hidden):
            raise ValueError(f"value_stack shape {tuple(v_act.shape)} != {(n_layers_plus, hidden)}")
        per_layer = v_act.float() - h_act.float()
        diffs[:, i, :] = per_layer
        mean_diff_norms.append(per_layer.norm(dim=-1).tolist())

    components, mean, direction, evr_per_layer = perlayer_pca_fit(diffs, rank=rank)
    # Flatten to the global on-disk schema for save_textual_v2_directions.
    components_flat = torch.stack(
        [components[:, j, :].reshape(-1) for j in range(rank)], dim=0,
    )
    pca_mean_flat = mean.reshape(-1)

    pc1 = components[:, 0, :]
    pc2 = components[:, 1, :] if rank >= 2 else None
    diag = {
        "explained_variance_ratio": None,
        "pc1_explained_variance": None,
        "pc2_explained_variance": None,
        "explained_variance_ratio_per_layer": evr_per_layer,
        "pc1_explained_variance_per_layer": [row[0] for row in evr_per_layer],
        "pc1_layer_norms": pc1.norm(dim=-1).tolist(),
        "pc2_layer_norms": pc2.norm(dim=-1).tolist() if pc2 is not None else None,
        "direction_layer_norms": direction.norm(dim=-1).tolist(),
        "mean_diff_layer_norms_mean_over_demos": (
            torch.tensor(mean_diff_norms).float().mean(dim=0).tolist()
        ),
        "n_pairs": n_pairs,
        "flat_dim": int(n_layers_plus * hidden),
        "components": components_flat.numpy().astype(np.float32),
        "pca_mean": pca_mean_flat.numpy().astype(np.float32),
        "components_3d": components.numpy().astype(np.float32),
        "pca_mean_3d": mean.numpy().astype(np.float32),
        "fit_locus": FIT_LOCUS_PERLAYER,
    }
    return direction, diag


def _global_cell_dirs(
    model_short: str,
    arm: str,
    num_demos: int,
    demos_hash: str,
) -> Tuple[Path, str]:
    """Return (global_fit_dir, deployed_slug)."""
    slug = partition_slug(
        demos_hash, DIMENSION, num_demos, seed=PARTITION_SEED, rank=RANK,
    )
    if arm == "deployed":
        return partition_cache_dir(model_short, slug), slug
    if arm == "control":
        return shuffled_demos850_direction_dir(model_short, num_demos), slug
    raise ValueError(f"arm must be 'deployed' or 'control'; got {arm!r}")


def _perlayer_out_dir(
    model_short: str,
    arm: str,
    num_demos: int,
    demos_hash: str,
    *,
    artifacts_root: Optional[Path] = None,
) -> Tuple[Path, str]:
    slug = partition_slug(
        demos_hash,
        DIMENSION,
        num_demos,
        seed=PARTITION_SEED,
        rank=RANK,
        fit_locus=FIT_LOCUS_PERLAYER,
    )
    if arm == "deployed":
        return (
            perlayer_partition_cache_dir(
                model_short, slug, artifacts_root=artifacts_root,
            ),
            slug,
        )
    if arm == "control":
        return (
            perlayer_shuffled_control_dir(
                model_short, num_demos, artifacts_root=artifacts_root,
            ),
            slug,
        )
    raise ValueError(f"arm must be 'deployed' or 'control'; got {arm!r}")


def _act_cache_for_arm(model_short: str, arm: str) -> Path:
    if arm == "deployed":
        return act_cache_dir(model_short)
    if arm == "control":
        return shuffled_demos850_act_cache_dir(model_short)
    raise ValueError(f"arm must be 'deployed' or 'control'; got {arm!r}")


def compute_perlayer_direction_for_cell(
    model_short: str,
    arm: str,
    num_demos: int,
    *,
    demos_path: Optional[Path] = None,
    partition_path: Optional[Path] = None,
    rank: int = RANK,
    force_recompute: bool = False,
    artifacts_root: Optional[Path] = None,
) -> Tuple[np.ndarray, dict, dict]:
    """Fit one per-layer PCA cell from cached stacks. Returns (dirs, meta, checks)."""
    if arm not in ("deployed", "control"):
        raise ValueError(f"arm must be 'deployed' or 'control'; got {arm!r}")
    if num_demos not in PARTITION_SIZES:
        raise ValueError(f"num_demos must be one of {PARTITION_SIZES}")
    if model_short not in EXPECTED_DECODER_SHAPES:
        raise ValueError(f"unsupported model_short: {model_short}")

    demos_path = Path(demos_path) if demos_path is not None else vti_demos_850_path()
    partition_path = (
        Path(partition_path)
        if partition_path is not None
        else vti_demos_850_partition_path()
    )
    demos_hash = demos_content_hash(demos_path)
    if demos_hash != EXPECTED_DEMOS_HASH:
        raise RuntimeError(
            f"demos_850 hash {demos_hash} != expected {EXPECTED_DEMOS_HASH}"
        )

    part = build_or_load_partition(
        demos_path, partition_path, seed=PARTITION_SEED, sizes=PARTITION_SIZES,
    )
    check_partition_integrity(demos_path, part, sizes=PARTITION_SIZES)
    block_ids = list(part["blocks"][str(num_demos)])

    global_dir, deployed_slug = _global_cell_dirs(
        model_short, arm, num_demos, demos_hash,
    )
    if not (global_dir / "directions.npz").exists():
        raise FileNotFoundError(f"Global-fit cell missing: {global_dir}")
    global_dirs, global_meta = load_textual_v2_directions(global_dir)
    ids_used = list(global_meta["ids_used"])
    if ids_used != block_ids and arm == "deployed":
        raise RuntimeError(
            f"Deployed ids_used != partition block nd{num_demos}"
        )
    if arm == "control":
        # Control ids_used must match the parallel global control cell and the
        # deployed partition block (same demos, deranged images).
        dep_global, _ = _global_cell_dirs(
            model_short, "deployed", num_demos, demos_hash,
        )
        _, dep_meta = load_textual_v2_directions(dep_global)
        dep_ids = list(dep_meta["ids_used"])
        if ids_used != dep_ids:
            raise RuntimeError(
                f"Control ids_used != deployed ids_used for nd{num_demos}"
            )
        if dep_ids != block_ids:
            raise RuntimeError(
                f"Deployed ids_used != partition block nd{num_demos}"
            )

    # Expected shape from parallel global cell, cross-checked against table.
    expected = EXPECTED_DECODER_SHAPES[model_short]
    if tuple(global_dirs.shape) != expected:
        raise RuntimeError(
            f"Global cell shape {global_dirs.shape} != expected {expected}"
        )

    out_dir, slug = _perlayer_out_dir(
        model_short,
        arm,
        num_demos,
        demos_hash,
        artifacts_root=artifacts_root,
    )
    if (out_dir / "directions.npz").exists() and not force_recompute:
        directions, meta = load_textual_v2_directions(out_dir)
        checks = {
            "skipped_extraction": True,
            "reason": "directions already present; pass --force_recompute to rebuild",
            "out_dir": _rel(out_dir),
            "n_pairs": int(meta.get("n_pairs", 0)),
            "forwards_executed": int(meta.get("forwards_executed", -1)),
            "shape": list(directions.shape),
        }
        return directions, meta, checks

    act_dir = _act_cache_for_arm(model_short, arm)
    cache = ActivationCache(str(act_dir))
    # ActivationCache.__init__ mkdir's; both caches already exist. Load only.
    h_stacks: List[torch.Tensor] = []
    v_stacks: List[torch.Tensor] = []
    cache_hits = 0
    for rid in ids_used:
        h_stacks.append(load_stack_readonly(cache, rid, DIMENSION))
        v_stacks.append(load_stack_readonly(cache, rid, "value"))
        cache_hits += 2

    full, diag = obtain_textual_vti_perlayer_from_stacks(
        h_stacks, v_stacks, rank=rank,
    )
    directions = full[1:].detach().cpu().float().numpy().astype(np.float32)
    if tuple(directions.shape) != expected:
        raise RuntimeError(
            f"Per-layer direction shape {directions.shape} != {expected}"
        )
    if not np.isfinite(directions).all():
        raise RuntimeError("Per-layer directions contain non-finite entries")
    layer_norms = np.linalg.norm(directions.astype(np.float64), axis=1)
    if not np.all(layer_norms > 0):
        raise RuntimeError("Per-layer directions have a zero-norm decoder row")

    # Check 2 fragment: mean identity vs global pca_mean_flat reshape.
    global_comp = np.load(global_dir / "components.npz")
    global_mean = np.asarray(global_comp["pca_mean_flat"], dtype=np.float64)
    n_plus = int(global_comp["n_layers_plus"])
    hidden = int(global_comp["hidden_dim"])
    global_mean_3d = global_mean.reshape(n_plus, hidden)
    perlayer_mean = np.asarray(diag["pca_mean_3d"], dtype=np.float64)
    mean_abs = float(np.max(np.abs(global_mean_3d - perlayer_mean)))
    scale = float(np.max(np.abs(global_mean))) or 1.0
    mean_gate = 1e-6 * scale
    if mean_abs > mean_gate:
        raise RuntimeError(
            f"Check 2 mean identity failed: max_abs_diff={mean_abs} > {mean_gate}"
        )
    # Reconstruction identity: components[l,0] + mean[l] == direction[l]
    comps_3d = np.asarray(diag["components_3d"], dtype=np.float64)
    recon = comps_3d[:, 0, :] + perlayer_mean
    recon_diff = float(np.max(np.abs(recon - full.numpy().astype(np.float64))))
    if recon_diff > 1e-5:
        raise RuntimeError(
            f"Per-layer recon identity failed: max_abs_diff={recon_diff}"
        )
    if directions.shape != (n_plus - 1, hidden):
        raise RuntimeError(
            f"Decoder slice shape {directions.shape} != ({n_plus - 1}, {hidden})"
        )

    derangement_meta: Dict = {}
    if arm == "control":
        mapping, derangement_file = load_or_write_block_derangement(
            ids_used,
            num_demos,
            deployed_slug=deployed_slug,
            seed=DERANGEMENT_SEED,
            force=False,
        )
        n_fixed = validate_derangement(ids_used, mapping)
        if n_fixed != 0:
            raise RuntimeError(f"Derangement has {n_fixed} fixed points")
        if set(mapping.keys()) != set(ids_used):
            raise RuntimeError("Derangement key set != ids_used")
        derangement_meta = {
            "control_for_deployed_slug": deployed_slug,
            "control_type": "image_derangement",
            "derangement_file": _rel(derangement_file),
            "derangement_seed": DERANGEMENT_SEED,
            "derangement_fixed_points": n_fixed,
        }

    try:
        part_rel = str(partition_path.relative_to(project_root()))
    except ValueError:
        part_rel = str(partition_path)

    meta = {
        "fit_locus": FIT_LOCUS_PERLAYER,
        "fit_locus_description": FIT_LOCUS_DESCRIPTION["perlayer"],
        "arm": "deployed" if arm == "deployed" else "shuffled_image_control",
        "demos_path": str(demos_path),
        "demos_file": demos_path.name,
        "content_hash_sha256_16": demos_hash,
        "partition_file": part_rel,
        "partition_content_hash": part["_meta"]["content_hash_sha256_16"],
        "block_size": num_demos,
        "dimension": DIMENSION,
        "num_demos_requested": num_demos,
        "n_pairs": len(ids_used),
        "ids_used": list(ids_used),
        "skipped_ids": [],
        "token_policy": TOKEN_POLICY,
        "diff_polarity": DIFF_POLARITY,
        "sign_convention": SIGN_CONVENTION,
        "selection_policy": SELECTION_POLICY,
        "seed": PARTITION_SEED,
        "rank": rank,
        "steer_component": STEER_COMPONENT,
        "steer_reconstruction": "live_pc1_plus_mean",
        "explained_variance_ratio": None,
        "pc1_explained_variance": None,
        "pc2_explained_variance": None,
        "explained_variance_ratio_per_layer": diag["explained_variance_ratio_per_layer"],
        "pc1_explained_variance_per_layer": diag["pc1_explained_variance_per_layer"],
        "pc1_layer_norms": diag["pc1_layer_norms"],
        "pc2_layer_norms": diag["pc2_layer_norms"],
        "direction_layer_norms": diag["direction_layer_norms"],
        "mean_diff_layer_norms_mean_over_demos": diag[
            "mean_diff_layer_norms_mean_over_demos"
        ],
        "model_short": model_short,
        "slug": slug,
        "global_fit_source_dir": _rel(global_dir),
        "global_fit_metadata_sha256": _sha256_file(global_dir / "metadata.json"),
        "global_fit_directions_sha256": _sha256_file(global_dir / "directions.npz"),
        "forwards_executed": 0,
        "cache_hits": cache_hits,
        "act_cache_dir": _rel(act_dir),
        "act_cache_read_only": True,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "git_commit": _git_commit(),
        **derangement_meta,
    }
    # Drop bulky 3-D arrays from metadata (stored flattened in components.npz).
    save_textual_v2_directions(
        directions,
        meta,
        out_dir,
        components=diag["components"],
        pca_mean=diag["pca_mean"],
        n_layers_plus=full.shape[0],
        hidden_dim=full.shape[1],
    )

    pc1_evr = diag["pc1_explained_variance_per_layer"]
    checks = {
        "skipped_extraction": False,
        "out_dir": _rel(out_dir),
        "n_pairs": len(ids_used),
        "shape": list(directions.shape),
        "expected_shape": list(expected),
        "forwards_executed": 0,
        "cache_hits": cache_hits,
        "act_cache_read_only": True,
        "mean_identity_max_abs_diff": mean_abs,
        "mean_identity_gate": mean_gate,
        "mean_identity_pass": mean_abs <= mean_gate,
        "recon_identity_max_abs_diff": recon_diff,
        "pc1_evr_min": float(min(pc1_evr)),
        "pc1_evr_max": float(max(pc1_evr)),
        "direction_layer_norms_decoder_min": float(layer_norms.min()),
        "direction_layer_norms_decoder_max": float(layer_norms.max()),
        "all_finite": True,
        "no_zero_layer": bool(np.all(layer_norms > 0)),
    }
    return directions, meta, checks
