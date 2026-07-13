"""Image perturbations for visual VTI direction extraction."""

from __future__ import annotations

import random
from typing import List, Literal, Optional

import torch

PerturbType = Literal["patch_mask", "gaussian_noise"]
MaskFill = Literal["zero", "mean"]


def mask_patches(
    tensor: torch.Tensor,
    indices: List[int],
    patch_size: int = 14,
    fill: MaskFill = "zero",
) -> torch.Tensor:
    """Mask square patches on a ``(C, H, W)`` CLIP-normalized tensor."""
    new_tensor = tensor.clone()
    h, w = tensor.shape[1], tensor.shape[2]
    patches_per_row = w // patch_size
    if fill == "mean":
        fill_values = tensor.mean(dim=(1, 2), keepdim=True)
    else:
        fill_values = torch.zeros(tensor.shape[0], 1, 1, dtype=tensor.dtype)

    for index in indices:
        row = index // patches_per_row
        col = index % patches_per_row
        start_y = row * patch_size
        start_x = col * patch_size
        new_tensor[
            :, start_y:start_y + patch_size, start_x:start_x + patch_size
        ] = fill_values.expand(-1, patch_size, patch_size)
    return new_tensor


def perturb(
    image_tensor: torch.Tensor,
    perturb_type: PerturbType,
    *,
    mask_ratio: float = 0.99,
    mask_fill: MaskFill = "zero",
    noise_sigma: float = 0.1,
    patch_size: int = 14,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Return a perturbed copy of ``image_tensor`` ``(C, H, W)``."""
    if perturb_type == "gaussian_noise":
        noise = torch.randn_like(image_tensor) * noise_sigma
        return image_tensor + noise

    h, w = image_tensor.shape[1], image_tensor.shape[2]
    n_patches = (h // patch_size) * (w // patch_size)
    n_mask = int(mask_ratio * n_patches)
    n_mask = max(0, min(n_mask, n_patches))
    rng = rng or random.Random(0)
    indices = rng.sample(range(n_patches), n_mask) if n_mask else []
    return mask_patches(
        image_tensor, indices, patch_size=patch_size, fill=mask_fill,
    )
