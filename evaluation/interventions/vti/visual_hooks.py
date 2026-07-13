"""Forward-hook context for visual VTI steering on the ViT encoder."""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import torch

from src.vision_dispatch import VisionDispatch, get_vision_dispatch

from .steer import HOOK_SITES, steer


def _output_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_tensor(output, new_tensor):
    if isinstance(output, tuple):
        return (new_tensor,) + output[1:]
    return new_tensor


def _resolve_vit_module(dispatch: VisionDispatch, wrapper, layer_idx: int, hook_site: str):
    if hook_site == "mlp":
        return dispatch.get_vit_mlp(wrapper, layer_idx)
    if hook_site == "layer":
        return dispatch.get_vit_layer(wrapper, layer_idx)
    raise ValueError(f"Unknown hook_site '{hook_site}'. Choose from {HOOK_SITES}.")


@contextmanager
def vti_visual_hook_ctx(
    wrapper,
    directions,  # np.ndarray (n_layers, n_tokens, hidden_dim)
    *,
    variant: str,
    alpha: float,
    hook_site: str,
    eps_coeff: float = 0.1,
    include_cls: bool = True,
):
    """Register per-layer ViT steer hooks; always removed on exit."""
    dispatch = get_vision_dispatch(wrapper)
    if hook_site not in HOOK_SITES:
        raise ValueError(f"Unknown hook_site '{hook_site}'")

    if isinstance(directions, np.ndarray):
        dir_t = torch.from_numpy(directions)
    else:
        dir_t = directions
    n_layers, n_tokens, hidden_dim = dir_t.shape
    if n_layers != dispatch.n_layers:
        raise ValueError(
            f"Expected {dispatch.n_layers} direction layers, got {n_layers}"
        )

    handles = []

    def _make_hook(layer_idx: int):
        layer_dirs = dir_t[layer_idx]  # (n_tokens, hidden_dim)

        def hook(_module, _inputs, output):
            t = _output_tensor(output)
            if t.dim() != 3:
                return output
            _batch, seq, dim = t.shape
            if dim != hidden_dim:
                raise RuntimeError(
                    f"ViT hook dim {dim} != direction dim {hidden_dim}"
                )
            if include_cls:
                if seq != layer_dirs.shape[0]:
                    raise RuntimeError(
                        f"ViT seq_len {seq} != direction n_tokens "
                        f"{layer_dirs.shape[0]}"
                    )
                pos_range = range(seq)
                dir_index = lambda p: p
            else:
                if seq != layer_dirs.shape[0] + 1:
                    raise RuntimeError(
                        f"ViT seq_len {seq} != direction n_tokens+1 "
                        f"({layer_dirs.shape[0] + 1})"
                    )
                pos_range = range(1, seq)
                dir_index = lambda p: p - 1

            steered = t.clone()
            for pos in pos_range:
                direction = layer_dirs[dir_index(pos)].to(
                    device=t.device, dtype=t.dtype,
                )
                steered[:, pos:pos + 1, :] = steer(
                    t[:, pos:pos + 1, :],
                    direction,
                    alpha,
                    variant,
                    eps_coeff,
                )
            return _replace_tensor(output, steered)

        return hook

    try:
        for layer_idx in range(n_layers):
            mod = _resolve_vit_module(dispatch, wrapper, layer_idx, hook_site)
            handles.append(mod.register_forward_hook(_make_hook(layer_idx)))
        yield
    finally:
        for h in handles:
            h.remove()
