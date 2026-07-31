"""demos_v2 textual VTI directions — live extractor math + shuffled_prefix selection.

Reuses the author-demo textual path geometry from ``directions.py`` /
``obtain_textual_vti`` (global flatten PCA, ``value − h_value`` diffs,
``PC1 + mean`` reconstruction) but:

- reads ``data/vti/demos_v2.jsonl`` with a dimension selector
- selects demos via nested ``shuffled_prefix`` (seed 42 master order)
- caches VL last-token stacks once per (image, caption-variant)
- writes identity-bearing caches under ``textual_v2/{slug}/``
- fits rank-2 PCA so PC2 is available for later diagnostics; **steering
  uses the live rank-1 reconstruction** (component 0 + mean only)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.extraction import ActivationCache, cleanup_gpu, get_last_token_activations
from src.paths import experiment_artifacts_dir, project_root, vti_demos_v2_path

from .directions import _demo_prompt, load_demo_image
from .pca import PCA

DIMENSIONS = ("existence", "attribute", "counting", "relation", "all")
NUM_DEMOS_GRID = (50, 100, 200, 500)
DEFAULT_ORDER_PATH = project_root() / "data" / "vti" / "demos_v2_order_s42.json"
SELECTION_POLICY = "shuffled_prefix"
TOKEN_POLICY = (
    "last_token_of_full_sequence: forward_vl(image, question + ' ' + caption); "
    "take hidden_states[layer][0, -1, :] for embedding row + every decoder layer"
)
DIFF_POLARITY = "value_minus_h_value"  # live textual path (clean − hallucinated)
SIGN_CONVENTION = (
    "live_textual: PCA.svd_flip on U/V (max-abs column of U); "
    "steering direction = (PC1 + mean) reshaped — no mean-diff sign-align"
)
STEER_COMPONENT = 0  # PC1 only (live recon with n_components effective 1)
FIT_LOCUS_GLOBAL = "global"
FIT_LOCUS_PERLAYER = "perlayer"


def demos_content_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root()),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def load_demos_v2_rows(demos_path: Optional[Path] = None) -> List[dict]:
    path = Path(demos_path) if demos_path is not None else vti_demos_v2_path()
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_or_build_master_order(
    demos_path: Optional[Path] = None,
    order_path: Path = DEFAULT_ORDER_PATH,
    seed: int = 42,
) -> dict:
    """Return order JSON; write ``demos_v2_order_s42.json`` if missing."""
    demos_path = Path(demos_path) if demos_path is not None else vti_demos_v2_path()
    h = demos_content_hash(demos_path)
    if order_path.exists():
        obj = json.loads(order_path.read_text())
        meta = obj.get("_meta") or {}
        if meta.get("content_hash_sha256_16") != h:
            raise RuntimeError(
                f"Order file hash {meta.get('content_hash_sha256_16')} != "
                f"demos hash {h}; rebuild {order_path}"
            )
        return obj

    import random

    rows = load_demos_v2_rows(demos_path)
    ids = sorted(r["id"] for r in rows)
    rng = random.Random(seed)
    rng.shuffle(ids)
    obj = {
        "_meta": {
            "demos_file": str(demos_path.relative_to(project_root()))
            if demos_path.is_relative_to(project_root())
            else str(demos_path),
            "content_hash_sha256_16": h,
            "seed": seed,
            "selection_policy": SELECTION_POLICY,
            "n_ids": len(ids),
            "created": datetime.now().strftime("%Y-%m-%d"),
        },
        "ids": ids,
    }
    order_path.parent.mkdir(parents=True, exist_ok=True)
    order_path.write_text(json.dumps(obj, indent=2) + "\n")
    return obj


def _row_has_dimension(row: dict, dimension: str) -> bool:
    hv = (row.get("h_values") or {}).get(dimension)
    return bool(hv and str(hv).strip()) and bool(row.get("value") and str(row["value"]).strip())


def select_prefix_demos(
    rows_by_id: Dict[str, dict],
    master_ids: Sequence[str],
    dimension: str,
    num_demos: int,
) -> Tuple[List[dict], List[str], List[str]]:
    """First ``num_demos`` master-order ids valid for ``dimension``.

    Returns ``(flat_demos, used_ids, skipped_ids)`` where each flat demo has
    ``value`` / ``h_value`` (dimension caption) for the live extractor.
    """
    used: List[dict] = []
    used_ids: List[str] = []
    skipped: List[str] = []
    for rid in master_ids:
        row = rows_by_id.get(rid)
        if row is None:
            skipped.append(rid)
            continue
        if not _row_has_dimension(row, dimension):
            skipped.append(rid)
            continue
        flat = {
            "id": row["id"],
            "image": row["image"],
            "question": row.get("question", "Describe this image in detail."),
            "value": row["value"],
            "h_value": row["h_values"][dimension],
        }
        used.append(flat)
        used_ids.append(rid)
        if len(used) >= num_demos:
            break
    return used, used_ids, skipped


def textual_v2_slug(
    demos_hash: str,
    dimension: str,
    num_demos: int,
    seed: int = 42,
    rank: int = 2,
    selection_policy: str = SELECTION_POLICY,
    fit_locus: str = FIT_LOCUS_GLOBAL,
) -> str:
    policy = "prefix" if selection_policy == SELECTION_POLICY else selection_policy
    base = (
        f"demosv2_{demos_hash[:8]}_{dimension}_nd{num_demos}_"
        f"s{seed}_r{rank}_{policy}"
    )
    if fit_locus == FIT_LOCUS_GLOBAL:
        return base
    if fit_locus == FIT_LOCUS_PERLAYER:
        return f"{base}_perlayer"
    raise ValueError(
        f"fit_locus must be one of ('global', 'perlayer'); got {fit_locus!r}"
    )


def textual_v2_cache_dir(model_short: str, slug: str) -> Path:
    return experiment_artifacts_dir("vti", model_short) / "textual_v2" / slug


def act_cache_dir(model_short: str) -> Path:
    return experiment_artifacts_dir("vti", model_short) / "textual_v2" / "_act_cache"


def variant_suffix(variant: str) -> str:
    """ActivationCache suffix for a caption variant (``value`` or dimension)."""
    return f"vl_v2_{variant}"


def _stack_from_act_dict(act: Dict[int, np.ndarray]) -> torch.Tensor:
    """``(num_layers+1, hidden_dim)`` float32 tensor from ActivationCache dict."""
    n = max(act.keys()) + 1
    rows = [torch.from_numpy(np.asarray(act[i], dtype=np.float32)) for i in range(n)]
    return torch.stack(rows, dim=0)


def ensure_variant_activation(
    wrapper,
    cache: ActivationCache,
    demo: dict,
    variant: str,
    caption: str,
) -> torch.Tensor:
    """Return last-token stack for one (image, caption); cache on miss."""
    sid = demo["id"]
    suffix = variant_suffix(variant)
    cached = cache.load_or_none(sid, suffix=suffix)
    if cached is not None:
        return _stack_from_act_dict(cached)

    image = load_demo_image(demo)
    text = _demo_prompt(demo, caption)
    with torch.no_grad():
        hidden_states, _, _ = wrapper.forward_vl(image, text)
    n_expected = wrapper.num_layers + 1
    if len(hidden_states) != n_expected:
        raise RuntimeError(
            f"hidden_states length {len(hidden_states)} != "
            f"num_layers+1 ({n_expected}) for {wrapper.model_name}"
        )
    act = get_last_token_activations(hidden_states)
    cache.save(sid, act, suffix=suffix)
    cleanup_gpu()
    return _stack_from_act_dict(act)


def prefetch_activations(
    wrapper,
    demos: Sequence[dict],
    variants: Sequence[str],
    model_short: str,
    rows_by_id: Optional[Dict[str, dict]] = None,
) -> None:
    """Forward missing (demo, variant) pairs. ``variants`` uses ``value`` or dim names."""
    cache = ActivationCache(str(act_cache_dir(model_short)))
    for demo in demos:
        row = (rows_by_id or {}).get(demo["id"], demo)
        for variant in variants:
            if variant == "value":
                caption = demo.get("value") or row.get("value")
            else:
                caption = (row.get("h_values") or {}).get(variant) or demo.get("h_value")
            if not caption:
                raise ValueError(f"Missing caption for {demo['id']} variant={variant}")
            # Build a demo dict with the right fields for image/question lookup.
            ensure_variant_activation(wrapper, cache, demo, variant, caption)


def _live_pca_fit(
    diffs_flat: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
    """Fit live textual PCA on ``(N, flat_dim)`` diffs.

    Returns ``(components (rank, flat), mean (1, flat), direction_flat (flat,),
    explained_variance_ratio[:rank])``.
    """
    pca = PCA(n_components=rank).to(diffs_flat.device).fit(diffs_flat.float())
    # components_: (1, rank, flat); mean_: (1, 1, flat)
    components = pca.components_.squeeze(0)  # (rank, flat)
    mean = pca.mean_.reshape(-1)  # (flat,)
    # Live steering recon with PC1 only (= rank-1 obtain_textual_vti).
    direction_flat = (components[STEER_COMPONENT] + mean).float()

    X = diffs_flat.float().unsqueeze(0)  # (1, N, flat)
    Z = X - pca.mean_
    _, S, _ = torch.linalg.svd(Z, full_matrices=False)
    s2 = (S[0] ** 2).clamp(min=0)
    total = float(s2.sum().item()) or 1.0
    evr = [float(s2[i].item() / total) for i in range(min(rank, s2.numel()))]
    return components.cpu(), mean.cpu(), direction_flat.cpu(), evr


def obtain_textual_vti_v2_from_stacks(
    h_stacks: Sequence[torch.Tensor],
    value_stacks: Sequence[torch.Tensor],
    rank: int = 2,
) -> Tuple[torch.Tensor, dict]:
    """Live-path PCA on cached stacks.

    ``h_stacks`` / ``value_stacks``: each ``(num_layers+1, hidden_dim)``.
    Diff polarity matches ``obtain_textual_vti``: ``value − h_value``.

    Returns decoder-aligned full direction ``(num_layers+1, hidden_dim)``
    (caller slices ``[1:]`` for hooks) plus PCA diagnostics.
    """
    if len(h_stacks) != len(value_stacks) or not h_stacks:
        raise ValueError("Need non-empty paired h/value stacks")
    n_layers_plus, hidden = h_stacks[0].shape
    diffs = []
    mean_diff_norms = []
    for h_act, v_act in zip(h_stacks, value_stacks):
        # Live path: pos=value, neg=h_value → value - h_value
        d = (v_act.reshape(-1) - h_act.reshape(-1)).float()
        diffs.append(d)
        per_layer = v_act.float() - h_act.float()
        mean_diff_norms.append(per_layer.norm(dim=-1).tolist())
    fit_data = torch.stack(diffs, dim=0)
    components, mean, direction_flat, evr = _live_pca_fit(fit_data, rank=rank)
    direction = direction_flat.view(n_layers_plus, hidden)

    # Per-layer PC norms (pre-recon unit-ish components reshaped).
    pc1 = components[0].view(n_layers_plus, hidden)
    pc2 = components[1].view(n_layers_plus, hidden) if rank >= 2 else None
    diag = {
        "explained_variance_ratio": evr,
        "pc1_explained_variance": evr[0] if evr else None,
        "pc2_explained_variance": evr[1] if len(evr) > 1 else None,
        "pc1_layer_norms": pc1.norm(dim=-1).tolist(),
        "pc2_layer_norms": pc2.norm(dim=-1).tolist() if pc2 is not None else None,
        "direction_layer_norms": direction.norm(dim=-1).tolist(),
        "mean_diff_layer_norms_mean_over_demos": (
            torch.tensor(mean_diff_norms).float().mean(dim=0).tolist()
        ),
        "n_pairs": len(diffs),
        "flat_dim": int(fit_data.shape[1]),
        "components": components.numpy().astype(np.float32),
        "pca_mean": mean.numpy().astype(np.float32),
    }
    return direction, diag


def save_textual_v2_directions(
    directions: np.ndarray,
    meta: dict,
    cache_dir: Path,
    components: Optional[np.ndarray] = None,
    pca_mean: Optional[np.ndarray] = None,
    n_layers_plus: Optional[int] = None,
    hidden_dim: Optional[int] = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "directions.npz"
    arrays = {f"layer_{i}": directions[i] for i in range(directions.shape[0])}
    np.savez(
        path,
        **arrays,
        num_layers=directions.shape[0],
        hidden_dim=directions.shape[1],
    )
    if components is not None and pca_mean is not None and n_layers_plus and hidden_dim:
        # Store reshaped PC1/PC2 (+ embedding row) for tomorrow's projection diagnostic.
        payload = {
            "pca_mean_flat": pca_mean,
            "n_layers_plus": n_layers_plus,
            "hidden_dim": hidden_dim,
            "rank": components.shape[0],
        }
        for i in range(components.shape[0]):
            payload[f"pc{i}"] = components[i].reshape(n_layers_plus, hidden_dim)
        np.savez(cache_dir / "components.npz", **payload)
    (cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_textual_v2_directions(cache_dir: Path) -> Tuple[np.ndarray, dict]:
    data = np.load(cache_dir / "directions.npz")
    n = int(data["num_layers"])
    directions = np.stack([data[f"layer_{i}"] for i in range(n)], axis=0).astype(np.float32)
    meta = json.loads((cache_dir / "metadata.json").read_text())
    return directions, meta


def compute_or_load_textual_directions_v2(
    wrapper,
    model_short: str,
    *,
    dimension: str = "all",
    num_demos: int = 500,
    rank: int = 2,
    seed: int = 42,
    demos_path: Optional[Path] = None,
    order_path: Path = DEFAULT_ORDER_PATH,
    force_recompute: bool = False,
    prefetch_variants: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Return decoder-layer directions ``(num_layers, hidden_dim)`` float32."""
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}")
    demos_path = Path(demos_path) if demos_path is not None else vti_demos_v2_path()
    demos_hash = demos_content_hash(demos_path)
    slug = textual_v2_slug(demos_hash, dimension, num_demos, seed=seed, rank=rank)
    cache_dir = textual_v2_cache_dir(model_short, slug)
    if (cache_dir / "directions.npz").exists() and not force_recompute:
        directions, meta = load_textual_v2_directions(cache_dir)
        if directions.shape != (wrapper.num_layers, wrapper.hidden_dim):
            raise RuntimeError(
                f"Cached v2 directions shape {directions.shape} != "
                f"({wrapper.num_layers}, {wrapper.hidden_dim})"
            )
        print(f"  [textual_v2] loaded {slug}  n_pairs={meta.get('n_pairs')}")
        return directions

    order_obj = load_or_build_master_order(demos_path, order_path=order_path, seed=seed)
    rows = load_demos_v2_rows(demos_path)
    rows_by_id = {r["id"]: r for r in rows}
    flat_demos, used_ids, skipped_ids = select_prefix_demos(
        rows_by_id, order_obj["ids"], dimension, num_demos,
    )
    if not flat_demos:
        raise RuntimeError(f"No valid demos for dimension={dimension}")

    # Prefetch value + this dimension (and any extra variants requested).
    variants = list(dict.fromkeys(
        list(prefetch_variants or []) + ["value", dimension]
    ))
    print(
        f"  [textual_v2] extracting {slug}: n_pairs={len(flat_demos)} "
        f"variants={variants} (skipped_so_far={len(skipped_ids)})"
    )
    prefetch_activations(
        wrapper, flat_demos, variants, model_short, rows_by_id=rows_by_id,
    )

    cache = ActivationCache(str(act_cache_dir(model_short)))
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

    question = flat_demos[0].get("question", "Describe this image in detail.")
    meta = {
        "demos_path": str(demos_path),
        "demos_file": demos_path.name,
        "content_hash_sha256_16": demos_hash,
        "dimension": dimension,
        "num_demos_requested": num_demos,
        "n_pairs": len(used_ids),
        "ids_used": used_ids,
        "skipped_ids": skipped_ids,
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
        "date": datetime.now().strftime("%Y-%m-%d"),
        "git_commit": _git_commit(),
    }
    # Drop bulky arrays from meta JSON (stored in components.npz).
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
        f"  [textual_v2] saved {cache_dir}  "
        f"PC1_evr={diag['pc1_explained_variance']:.4f}  "
        f"PC2_evr={diag.get('pc2_explained_variance')}"
    )
    return directions


def extract_dimension_grid(
    wrapper,
    model_short: str,
    dimension: str,
    num_demos_list: Sequence[int] = NUM_DEMOS_GRID,
    rank: int = 2,
    seed: int = 42,
    demos_path: Optional[Path] = None,
    order_path: Path = DEFAULT_ORDER_PATH,
) -> Dict[int, Path]:
    """Prefetch the max-prefix once, then PCA for each nested ``num_demos``."""
    demos_path = Path(demos_path) if demos_path is not None else vti_demos_v2_path()
    order_obj = load_or_build_master_order(demos_path, order_path=order_path, seed=seed)
    rows = load_demos_v2_rows(demos_path)
    rows_by_id = {r["id"]: r for r in rows}
    max_n = max(num_demos_list)
    flat_max, _, _ = select_prefix_demos(
        rows_by_id, order_obj["ids"], dimension, max_n,
    )
    prefetch_activations(
        wrapper,
        flat_max,
        ["value", dimension],
        model_short,
        rows_by_id=rows_by_id,
    )
    out: Dict[int, Path] = {}
    for n in num_demos_list:
        compute_or_load_textual_directions_v2(
            wrapper,
            model_short,
            dimension=dimension,
            num_demos=n,
            rank=rank,
            seed=seed,
            demos_path=demos_path,
            order_path=order_path,
        )
        h = demos_content_hash(demos_path)
        out[n] = textual_v2_cache_dir(
            model_short, textual_v2_slug(h, dimension, n, seed=seed, rank=rank),
        )
    return out
