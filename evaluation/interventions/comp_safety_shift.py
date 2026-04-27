"""
CompSafetyShift intervention: subtract the projection of each last-token hidden
state onto a per-layer compositional safety direction `c^l`.

The compositional direction is loaded from
`experiment_artifacts/{model_short}/compositional_safety/compositional_safety_direction_vectors.npz`,
where key `layer_l` was estimated at the residual stream `hidden_states[l]`,
i.e. the OUTPUT of transformer layer (l-1). Therefore, to apply the correction
at the same point where the direction was estimated, we hook the module that
PRODUCES `hidden_states[l]` — that is `model.layers[l-1]`. Layer index 0
(embedding output) is not addressable via a transformer-layer hook and is
rejected if requested.

Correction (per layer, last token only):
    x_corrected = x - alpha * dot(x, c^l) * c^l        (c^l is unit-normalised)
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import InterventionBase


class CompSafetyShiftIntervention(InterventionBase):
    """Compositional-direction modality-shift correction."""

    # Defaults are expressed in npz-key (== hidden_states index) space.
    # Hooks attach to model.layers[l - 1] for npz key layer_l.
    _DEFAULT_LAYER_RANGES = {
        "llava-1.5-7b-hf":      (6, 14),
        "qwen2-vl-7b":          (5, 12),
        "qwen2-vl-7b-instruct": (5, 12),
        "sharegpt4v-7b":        (6, 14),
        "qwen-vl-chat":         (6, 14),
    }

    def __init__(
        self,
        model_id: str,
        alpha: float = 1.0,
        layer_start: Optional[int] = None,
        layer_end: Optional[int] = None,
        project_root: Optional[str] = None,
    ):
        from src.model import _normalize_model_name

        self._model_id = model_id
        self._model_short = _normalize_model_name(model_id)
        self.alpha = float(alpha)

        default_start, default_end = self._DEFAULT_LAYER_RANGES.get(
            self._model_short, (5, 14)
        )
        self.layer_start = layer_start if layer_start is not None else default_start
        self.layer_end   = layer_end   if layer_end   is not None else default_end

        if self.layer_start < 1:
            raise ValueError(
                f"layer_start={self.layer_start} is invalid. The npz key 'layer_0' "
                "corresponds to the embedding output (pre-transformer) and is not "
                "addressable as a transformer-layer hook. Use layer_start >= 1."
            )
        if self.layer_end < self.layer_start:
            raise ValueError(
                f"layer_end ({self.layer_end}) < layer_start ({self.layer_start})."
            )

        root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        npz_path = (
            root / "experiment_artifacts" / self._model_short
            / "compositional_safety"
            / "compositional_safety_direction_vectors.npz"
        )
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Compositional safety direction vectors not found: {npz_path}\n"
                "Run diagnostic_experiments/run_scripts/run_compositional_safety.sh "
                f"for {model_id} first (or run_overnight_comp_directions.sh)."
            )
        data = np.load(str(npz_path))
        self._directions: dict[int, torch.Tensor] = {}
        for l in range(self.layer_start, self.layer_end + 1):
            key = f"layer_{l}"
            if key not in data.files:
                raise KeyError(
                    f"Direction vector missing for {key} in {npz_path}. "
                    f"Available keys: {sorted(data.files)}"
                )
            vec = data[key].astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm < 0.999:
                vec = vec / max(norm, 1e-12)
            self._directions[l] = torch.tensor(vec)
        self._npz_path = npz_path

    @property
    def name(self) -> str:
        return "comp_safety_shift"

    @property
    def config(self) -> dict:
        return {
            "alpha": self.alpha,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "direction_npz": str(self._npz_path),
        }

    # ── Layer access per wrapper family ─────────────────────────────────────
    def _get_layer_module(self, wrapper, transformer_idx: int):
        """Return the nn.Module whose OUTPUT is hidden_states[transformer_idx + 1]."""
        from src.model import (
            LLaVAWrapper, ShareGPT4VWrapper, Qwen2VLWrapper, QwenVLWrapper,
        )
        if isinstance(wrapper, LLaVAWrapper):
            return wrapper.model.language_model.model.layers[transformer_idx]
        if isinstance(wrapper, ShareGPT4VWrapper):
            return wrapper.model.model.layers[transformer_idx]
        if isinstance(wrapper, Qwen2VLWrapper):
            # transformers >= 4.45 layout: model.model.layers
            return wrapper.model.model.layers[transformer_idx]
        if isinstance(wrapper, QwenVLWrapper):
            # Qwen-VL custom code: GPT-style transformer.h
            return wrapper.model.transformer.h[transformer_idx]
        raise NotImplementedError(
            f"CompSafetyShiftIntervention does not yet support "
            f"{type(wrapper).__name__}. Add layer access in _get_layer_module()."
        )

    def _validate_hidden_dim(self, wrapper):
        if any(int(d.shape[0]) != int(wrapper.hidden_dim) for d in self._directions.values()):
            raise ValueError(
                f"Direction vector shape mismatch: hidden_dim={wrapper.hidden_dim} "
                f"but directions have shape {next(iter(self._directions.values())).shape}."
            )

    @contextmanager
    def _apply_hooks(self, wrapper):
        self._validate_hidden_dim(wrapper)
        device = next(wrapper.model.parameters()).device
        handles = []

        try:
            for npz_layer, direction_cpu in self._directions.items():
                transformer_idx = npz_layer - 1  # see module docstring
                module = self._get_layer_module(wrapper, transformer_idx)
                # Move direction to the device of THIS module's output
                # (under accelerate device_map, layers can live on different devices).
                try:
                    module_device = next(module.parameters()).device
                except StopIteration:
                    module_device = device
                direction = direction_cpu.to(module_device)
                alpha = self.alpha

                def make_hook(d, a):
                    def hook(_module, _input, output):
                        if isinstance(output, tuple):
                            hidden = output[0]
                            rest = output[1:]
                        else:
                            hidden = output
                            rest = None
                        if hidden is None or hidden.dim() < 2:
                            return output
                        # Cast direction to the hidden's dtype/device for the math.
                        d_local = d.to(dtype=hidden.dtype, device=hidden.device)
                        x = hidden[:, -1, :]                            # (B, H)
                        coef = (x * d_local).sum(dim=-1, keepdim=True)  # (B, 1)
                        x_new = x - a * coef * d_local                  # (B, H)
                        # Some HF layer outputs are non-contiguous / non-leaf;
                        # clone to be safe before in-place assignment.
                        new_hidden = hidden.clone()
                        new_hidden[:, -1, :] = x_new
                        if rest is None:
                            return new_hidden
                        return (new_hidden,) + rest
                    return hook

                h = module.register_forward_hook(make_hook(direction, alpha))
                handles.append(h)
            yield
        finally:
            for h in handles:
                h.remove()

    def generate(self, wrapper, image: Optional[Image.Image], question: str,
                 max_new_tokens: int = 256) -> str:
        with self._apply_hooks(wrapper):
            if image is not None:
                return wrapper.generate_vl(image, question,
                                           max_new_tokens=max_new_tokens)
            return wrapper.generate_text(question, max_new_tokens=max_new_tokens)
