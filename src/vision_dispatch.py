"""Vision-encoder dispatch for VTI visual steering (LLaVA-1.5 first)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from PIL import Image


@dataclass
class VisionDispatch:
    family: str
    n_layers: int
    hidden_dim: int
    n_tokens: int
    cls_index: int = 0
    patch_size: int = 14

    def get_vision_tower(self, wrapper):
        raise NotImplementedError

    def get_vit_layer(self, wrapper, idx: int):
        raise NotImplementedError

    def get_vit_mlp(self, wrapper, idx: int):
        raise NotImplementedError


class _LLaVAVisionDispatch(VisionDispatch):
    def get_vision_tower(self, wrapper):
        return wrapper.model.vision_tower

    def _encoder_layers(self, wrapper):
        return self.get_vision_tower(wrapper).vision_model.encoder.layers

    def get_vit_layer(self, wrapper, idx: int):
        return self._encoder_layers(wrapper)[idx]

    def get_vit_mlp(self, wrapper, idx: int):
        return self.get_vit_layer(wrapper, idx).mlp


def get_vision_dispatch(wrapper) -> VisionDispatch:
    cls_name = type(wrapper).__name__
    if cls_name == "LLaVAWrapper":
        return _LLaVAVisionDispatch(
            family="llava",
            n_layers=24,
            hidden_dim=1024,
            n_tokens=577,
            cls_index=0,
            patch_size=14,
        )
    raise NotImplementedError(
        f"Vision dispatch not implemented for {cls_name}. "
        "Visual VTI is LLaVA-1.5 only in this build."
    )


def prepare_pixel_tensor(wrapper, image: Image.Image) -> torch.Tensor:
    """CLIP-normalized pixel tensor ``(C, H, W)``."""
    proc = wrapper.processor.image_processor
    inputs = proc(images=image, return_tensors="pt")
    return inputs["pixel_values"][0].detach().cpu().float()


def clip_denorm_params(wrapper) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) as ``(C,)`` tensors for the vision image processor."""
    proc = wrapper.processor.image_processor
    mean = torch.tensor(proc.image_mean, dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(proc.image_std, dtype=torch.float32).view(-1, 1, 1)
    return mean, std


def denorm_pixel_tensor(wrapper, tensor: torch.Tensor) -> torch.Tensor:
    """Invert CLIP normalization → ``(C, H, W)`` in approx [0, 1]."""
    mean, std = clip_denorm_params(wrapper)
    t = tensor.detach().cpu().float()
    if t.dim() == 4:
        t = t[0]
    out = t * std + mean
    return out.clamp(0.0, 1.0)


@torch.no_grad()
def verify_vision_layout(wrapper, dispatch: VisionDispatch | None = None) -> dict:
    """Assert ViT hook paths and tensor shapes; return measured layout facts."""
    dispatch = dispatch or get_vision_dispatch(wrapper)
    dummy = Image.new("RGB", (336, 336), color=(128, 64, 32))
    pixel = prepare_pixel_tensor(wrapper, dummy).to(wrapper.device)
    vit = dispatch.get_vision_tower(wrapper)
    out = vit(
        pixel.unsqueeze(0),
        output_hidden_states=True,
        return_dict=True,
    )
    hs = out.hidden_states
    n_hs = len(hs)
    expected_layers = dispatch.n_layers + 1
    if n_hs != expected_layers:
        raise RuntimeError(
            f"ViT hidden_states length {n_hs} != n_layers+1 ({expected_layers})"
        )
    for i in range(dispatch.n_layers):
        dispatch.get_vit_layer(wrapper, i)
        dispatch.get_vit_mlp(wrapper, i)
    sample = hs[1]
    b, seq, dim = sample.shape
    if b != 1:
        raise RuntimeError(f"Expected batch=1, got {b}")
    if seq != dispatch.n_tokens:
        raise RuntimeError(
            f"ViT seq_len {seq} != expected {dispatch.n_tokens}"
        )
    if dim != dispatch.hidden_dim:
        raise RuntimeError(
            f"ViT hidden_dim {dim} != expected {dispatch.hidden_dim}"
        )
    return {
        "family": dispatch.family,
        "n_hidden_state_rows": n_hs,
        "n_encoder_layers": dispatch.n_layers,
        "n_tokens": seq,
        "hidden_dim": dim,
    }
