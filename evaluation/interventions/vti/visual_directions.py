"""Visual VTI direction extraction, caching, and perturbed-image dumps."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.paths import coco_train2014_dir, experiment_artifacts_dir, vti_demos_path
from src.vision_dispatch import (
    VisionDispatch,
    denorm_pixel_tensor,
    get_vision_dispatch,
    prepare_pixel_tensor,
)

from .directions import load_demo_image, load_vti_demos
from .pca import PCA
from .visual_capture import capture_corrupted_mean, capture_visual_hiddenstates
from .visual_perturb import MaskFill, PerturbType, perturb

DirectionRecon = Literal["top_pc", "legacy_pc_plus_mean"]


def _config_slug(params: dict) -> str:
    digest = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode(),
    ).hexdigest()[:12]
    return (
        f"{params['perturb_type']}_r{params['mask_ratio']}_{params['mask_fill']}"
        f"_{params['direction_recon']}_nd{params['num_demos']}"
        f"_nt{params['num_trials']}_s{params['seed']}_{digest}"
    )


def visual_direction_cache_dir(
    model_short: str,
    params: dict,
    cache_root: Optional[Path] = None,
) -> Path:
    root = cache_root or experiment_artifacts_dir("vti", model_short)
    return root / "visual" / _config_slug(params)


def _sign_align(pc: torch.Tensor, mean_diff: torch.Tensor) -> torch.Tensor:
    if torch.dot(pc.flatten(), mean_diff.flatten()) < 0:
        pc = -pc
    return pc


def _reconstruct_top_pc(demo_diffs: List[torch.Tensor]) -> torch.Tensor:
    """Per-(layer, token) rank-1 PCA on demo diffs (encoder rows only).

    ``demo_diffs[i]`` shape ``(n_layers+1, n_tokens, D)``; embedding row dropped.
    Returns ``(n_layers, n_tokens, D)``.
    """
    stacked = torch.stack(demo_diffs, dim=0)  # (N, L+1, T, D)
    enc = stacked[:, 1:, :, :]  # drop embedding
    n_layers, n_tokens, dim = enc.shape[1], enc.shape[2], enc.shape[3]
    out = torch.zeros(n_layers, n_tokens, dim)
    for layer in range(n_layers):
        for tok in range(n_tokens):
            vecs = enc[:, layer, tok, :].unsqueeze(0)  # (1, N, D)
            mean_diff = vecs.mean(dim=1, keepdim=True)
            pca = PCA(n_components=1).to(vecs.device).fit(vecs.float())
            pc = pca.components_[0, 0]
            pc = _sign_align(pc, mean_diff[0, 0])
            out[layer, tok] = pc
    return out


def _reconstruct_legacy_pc_plus_mean(demo_diffs: List[torch.Tensor]) -> torch.Tensor:
    """Port reference ``obtain_visual_vti`` reconstruction (layer-major reshape)."""
    n_layers, n_tokens, feat_dim = demo_diffs[0].shape
    hidden_states_all = []
    for diff in demo_diffs:
        h = diff.reshape(n_tokens, -1)
        hidden_states_all.append(h)
    fit_data = torch.stack(hidden_states_all, dim=1)
    pca = PCA(n_components=1).to(fit_data.device).fit(fit_data.float())
    direction = (
        pca.components_.sum(dim=1, keepdim=True) + pca.mean_
    ).mean(1).view(n_layers, n_tokens, feat_dim)
    return direction[1:]  # drop embedding row from final directions


def _maybe_drop_cls(directions: torch.Tensor, include_cls: bool) -> torch.Tensor:
    if include_cls:
        return directions
    return directions[:, 1:, :]


def _save_perturbed_examples(
    wrapper,
    out_dir: Path,
    demos: List[dict],
    *,
    perturb_type: PerturbType,
    mask_ratio: float,
    mask_fill: MaskFill,
    noise_sigma: float,
    patch_size: int,
    seed: int,
    n_demos: int = 3,
    n_trials: int = 5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    from torchvision.transforms.functional import to_pil_image

    for demo_idx, demo in enumerate(demos[:n_demos]):
        image = load_demo_image(demo)
        clean = prepare_pixel_tensor(wrapper, image)
        clean_rgb = denorm_pixel_tensor(wrapper, clean)
        to_pil_image(clean_rgb).save(out_dir / f"{demo_idx}_clean.png")
        rng = random.Random(seed + demo_idx)
        for trial in range(n_trials):
            pert = perturb(
                clean,
                perturb_type,
                mask_ratio=mask_ratio,
                mask_fill=mask_fill,
                noise_sigma=noise_sigma,
                patch_size=patch_size,
                rng=rng,
            )
            rgb = denorm_pixel_tensor(wrapper, pert)
            fname = (
                f"{demo_idx}_trial{trial}_{perturb_type}"
                f"_r{mask_ratio}_{mask_fill}.png"
            )
            to_pil_image(rgb).save(out_dir / fname)


def obtain_visual_vti(
    wrapper,
    demos: List[dict],
    *,
    perturb_type: PerturbType = "patch_mask",
    mask_ratio: float = 0.99,
    mask_fill: MaskFill = "zero",
    noise_sigma: float = 0.1,
    num_trials: int = 50,
    direction_recon: DirectionRecon = "top_pc",
    include_cls: bool = True,
    patch_size: int = 14,
    base_seed: int = 42,
    dispatch: Optional[VisionDispatch] = None,
    save_perturbed_pngs: bool = True,
    cache_dir: Optional[Path] = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute visual directions ``(n_layers, n_tokens, D)`` and metadata."""
    dispatch = dispatch or get_vision_dispatch(wrapper)
    demo_diffs: List[torch.Tensor] = []
    demo_ids: List[str] = []

    for i, demo in enumerate(demos):
        image = load_demo_image(demo)
        pixel = prepare_pixel_tensor(wrapper, image)
        clean = capture_visual_hiddenstates(wrapper, pixel, dispatch=dispatch)
        corrupted = capture_corrupted_mean(
            wrapper,
            pixel,
            perturb_type=perturb_type,
            mask_ratio=mask_ratio,
            mask_fill=mask_fill,
            noise_sigma=noise_sigma,
            num_trials=num_trials,
            patch_size=patch_size,
            seed=base_seed + i * 10007,
            dispatch=dispatch,
        )
        demo_diffs.append(corrupted - clean)
        demo_ids.append(demo.get("id", str(i)))

    if direction_recon == "top_pc":
        directions = _reconstruct_top_pc(demo_diffs)
    else:
        directions = _reconstruct_legacy_pc_plus_mean(demo_diffs)

    directions = _maybe_drop_cls(directions, include_cls)

    meta = {
        "perturb_type": perturb_type,
        "mask_ratio": mask_ratio,
        "mask_fill": mask_fill,
        "noise_sigma": noise_sigma,
        "num_trials": num_trials,
        "num_demos": len(demos),
        "direction_recon": direction_recon,
        "include_cls": include_cls,
        "patch_size": patch_size,
        "seed": base_seed,
        "demo_ids": demo_ids,
        "shape": list(directions.shape),
    }

    if save_perturbed_pngs and cache_dir is not None:
        _save_perturbed_examples(
            wrapper,
            cache_dir / "perturbed_examples",
            demos,
            perturb_type=perturb_type,
            mask_ratio=mask_ratio,
            mask_fill=mask_fill,
            noise_sigma=noise_sigma,
            patch_size=patch_size,
            seed=base_seed,
        )

    return directions, meta


def save_visual_directions(
    directions: np.ndarray,
    meta: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        directions=directions,
        meta_json=json.dumps(meta),
    )


def load_visual_directions(path: Path) -> Tuple[np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    directions = data["directions"].astype(np.float32)
    meta = json.loads(str(data["meta_json"]))
    return directions, meta


def compute_or_load_visual_directions(
    wrapper,
    model_short: str,
    *,
    num_demos: int = 70,
    seed: int = 42,
    demos_path: Optional[Path] = None,
    perturb_type: PerturbType = "patch_mask",
    mask_ratio: float = 0.99,
    mask_fill: MaskFill = "zero",
    noise_sigma: float = 0.1,
    num_trials: int = 50,
    direction_recon: DirectionRecon = "top_pc",
    include_cls: bool = True,
    patch_size: int = 14,
    force_recompute: bool = False,
) -> np.ndarray:
    params = {
        "perturb_type": perturb_type,
        "mask_ratio": mask_ratio,
        "mask_fill": mask_fill,
        "noise_sigma": noise_sigma,
        "num_trials": num_trials,
        "num_demos": num_demos,
        "direction_recon": direction_recon,
        "include_cls": include_cls,
        "patch_size": patch_size,
        "seed": seed,
        "demos_file": str(demos_path or vti_demos_path()),
    }
    cache_dir = visual_direction_cache_dir(model_short, params)
    cache_path = cache_dir / "directions.npz"
    if cache_path.exists() and not force_recompute:
        directions, _ = load_visual_directions(cache_path)
        return directions

    demos = load_vti_demos(
        demos_path=demos_path, num_demos=num_demos, seed=seed,
    )
    directions_t, meta = obtain_visual_vti(
        wrapper,
        demos,
        perturb_type=perturb_type,
        mask_ratio=mask_ratio,
        mask_fill=mask_fill,
        noise_sigma=noise_sigma,
        num_trials=num_trials,
        direction_recon=direction_recon,
        include_cls=include_cls,
        patch_size=patch_size,
        base_seed=seed,
        save_perturbed_pngs=True,
        cache_dir=cache_dir,
    )
    meta["params"] = params
    meta["demos_file_hash"] = hashlib.sha256(
        (demos_path or vti_demos_path()).read_bytes(),
    ).hexdigest()[:16]
    directions = directions_t.detach().cpu().numpy().astype(np.float32)
    save_visual_directions(directions, meta, cache_path)
    (cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return directions


def reconstruct_direction_from_diffs(
    demo_diffs: List[torch.Tensor],
    direction_recon: DirectionRecon,
    include_cls: bool = True,
) -> torch.Tensor:
    """Math-only entry for parity tests (pre-captured diffs)."""
    if direction_recon == "top_pc":
        directions = _reconstruct_top_pc(demo_diffs)
    else:
        directions = _reconstruct_legacy_pc_plus_mean(demo_diffs)
    return _maybe_drop_cls(directions, include_cls)
