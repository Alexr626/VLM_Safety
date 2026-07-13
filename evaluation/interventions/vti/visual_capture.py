"""Capture ViT hidden states for visual VTI."""

from __future__ import annotations

import random
from typing import List, Optional

import torch
from PIL import Image

from src.vision_dispatch import VisionDispatch, get_vision_dispatch, prepare_pixel_tensor

from .visual_perturb import MaskFill, PerturbType, perturb


@torch.no_grad()
def capture_visual_hiddenstates(
    wrapper,
    pixel_values: torch.Tensor,
    dispatch: Optional[VisionDispatch] = None,
) -> torch.Tensor:
    """One ViT forward; stack all ``hidden_states`` rows including embeddings.

    Returns ``(n_layers + 1, n_tokens, hidden_dim)`` on CPU float32.
    """
    dispatch = dispatch or get_vision_dispatch(wrapper)
    vit = dispatch.get_vision_tower(wrapper)
    if pixel_values.dim() == 3:
        pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.to(device=wrapper.device, dtype=wrapper.torch_dtype)
    out = vit(pixel_values, output_hidden_states=True, return_dict=True)
    stacked = torch.stack(
        [h[0].detach().cpu().float() for h in out.hidden_states],
        dim=0,
    )
    return stacked


@torch.no_grad()
def capture_corrupted_mean(
    wrapper,
    pixel_values: torch.Tensor,
    *,
    perturb_type: PerturbType,
    mask_ratio: float,
    mask_fill: MaskFill,
    noise_sigma: float,
    num_trials: int,
    patch_size: int,
    seed: int,
    dispatch: Optional[VisionDispatch] = None,
) -> torch.Tensor:
    """Average ``num_trials`` perturbed captures (running mean on CPU)."""
    dispatch = dispatch or get_vision_dispatch(wrapper)
    rng = random.Random(seed)
    running: Optional[torch.Tensor] = None
    for _ in range(num_trials):
        perturbed = perturb(
            pixel_values.cpu(),
            perturb_type,
            mask_ratio=mask_ratio,
            mask_fill=mask_fill,
            noise_sigma=noise_sigma,
            patch_size=patch_size,
            rng=rng,
        )
        h = capture_visual_hiddenstates(wrapper, perturbed, dispatch=dispatch)
        if running is None:
            running = h.clone()
        else:
            running += h
    assert running is not None
    return running / num_trials


def load_demo_pixels(wrapper, image: Image.Image) -> torch.Tensor:
    return prepare_pixel_tensor(wrapper, image)
