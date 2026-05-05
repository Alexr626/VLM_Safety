"""
CompSafetyShift intervention: subtract the projection of the **modality-induced
shift** onto a per-layer compositional safety direction `c^l`.

The modality-induced shift is:
    m^l = x_vl^l - x_tt^l
where x_vl is the VL (image + text) hidden state and x_tt is the TT (caption +
text) hidden state at the same layer. The correction removes the component of
this shift that lies along the compositional safety direction:

    x_corrected = x_vl - alpha * ((m^l · c^l) / ||c^l||^2) * c^l

To compute x_tt, the intervention performs a single text-only forward pass
(using the image caption) BEFORE generation begins, and caches the last-token
hidden state at each intervention layer.

Application schedule: PREFILL ONLY. The forward hook fires at most once per
layer per generate() call, on the first forward pass (the prompt prefill).
Subsequent autoregressive decoding steps are not intercepted; the corrected
representation propagates through the KV cache. This matches the upstream
ShiftDC reference implementation (see add_prefill_hooks / get_shiftdc_hook
in shiftdc/utils, which use an `applied` flag with identical semantics).

The compositional direction is loaded from
`experiment_artifacts/{model_short}/compositional_safety/{direction_source}/compositional_safety_direction_vectors.npz`,
where `direction_source` is one of "holisafe_tt" / "holisafe_vl" /
"mssbench_tt" / "mssbench_vl" (default: "mssbench_vl"). Key `layer_l` was
estimated at the residual stream `hidden_states[l]`, i.e. the OUTPUT of
transformer layer (l-1). Hooks attach to `model.layers[l-1]`.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import InterventionBase


class CompSafetyShiftIntervention(InterventionBase):
    """Compositional-direction modality-shift correction (prefill-only)."""

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
        direction_source: str = "mssbench_vl",
        direction_path: Optional[str] = None,
    ):
        from src.model import _normalize_model_name

        self._model_id = model_id
        self._model_short = _normalize_model_name(model_id)
        self.alpha = float(alpha)
        self._direction_source = direction_source

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
        if direction_path is not None:
            npz_path = Path(direction_path)
        else:
            npz_path = (
                root / "experiment_artifacts" / self._model_short
                / "compositional_safety" / direction_source
                / "compositional_safety_direction_vectors.npz"
            )
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Compositional safety direction vectors not found: {npz_path}\n"
                f"Run: python diagnostic_experiments/experiment_scripts/"
                f"compositional_safety_direction.py "
                f"--model {model_id} "
                f"--source {direction_source.split('_')[0]} "
                f"--representation {direction_source.split('_')[1]}"
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
            "direction_source": self._direction_source,
            "direction_npz": str(self._npz_path),
            "schedule": "prefill_only",
        }

    # ── Layer access per wrapper family ─────────────────────────────────────
    def _get_layer_module(self, wrapper, transformer_idx: int):
        """Return the nn.Module whose OUTPUT is hidden_states[transformer_idx + 1]."""
        from src.model import (
            LLaVAWrapper, ShareGPT4VWrapper, Qwen2VLWrapper, QwenVLWrapper,
        )
        if isinstance(wrapper, LLaVAWrapper):
            # transformers ≥ 4.45 flattened LlavaForConditionalGeneration so
            # `language_model` IS the LlamaModel (with `.layers` directly).
            # Older transformers had `language_model` = LlamaForCausalLM with
            # `.model.layers`. Support both layouts.
            lm = wrapper.model.language_model
            if hasattr(lm, "layers"):
                return lm.layers[transformer_idx]
            return lm.model.layers[transformer_idx]
        if isinstance(wrapper, ShareGPT4VWrapper):
            # Same dual-layout concern: model is LlamaForCausalLM in older
            # transformers (.model.layers), or LlamaModel in newer (.layers).
            inner = wrapper.model
            if hasattr(inner, "layers"):
                return inner.layers[transformer_idx]
            return inner.model.layers[transformer_idx]
        if isinstance(wrapper, Qwen2VLWrapper):
            inner = wrapper.model
            if hasattr(inner, "layers"):
                return inner.layers[transformer_idx]
            return inner.model.layers[transformer_idx]
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
    def _apply_hooks(self, wrapper, x_tt_cache: Optional[dict] = None):
        """Register forward hooks that apply the CompShift correction.

        The hooks fire AT MOST ONCE per layer per context-manager activation
        (i.e., one prefill per generate() call). An `applied` dict tracks
        per-layer state and is shared across all hook closures.

        Args:
            x_tt_cache: {npz_layer: Tensor(hidden_dim,)} — precomputed
                last-token TT hidden states. When provided, hooks compute
                m^l = x_vl - x_tt and project THAT onto c^l. When None,
                no correction is applied (hooks are no-ops).
        """
        self._validate_hidden_dim(wrapper)
        device = next(wrapper.model.parameters()).device
        handles = []
        applied: dict[int, bool] = {l: False for l in self._directions}

        try:
            for npz_layer, direction_cpu in self._directions.items():
                transformer_idx = npz_layer - 1  # see module docstring
                module = self._get_layer_module(wrapper, transformer_idx)
                try:
                    module_device = next(module.parameters()).device
                except StopIteration:
                    module_device = device
                direction = direction_cpu.to(module_device)
                alpha = self.alpha
                # x_tt for this layer (may be None if cache absent)
                x_tt_l = (x_tt_cache[npz_layer].to(module_device)
                          if x_tt_cache and npz_layer in x_tt_cache
                          else None)

                def make_hook(d, a, tt_l, layer_id):
                    def hook(_module, _input, output):
                        if applied[layer_id]:
                            return output  # prefill already corrected; skip
                        if tt_l is None:
                            return output  # no TT baseline → skip correction
                        if isinstance(output, tuple):
                            hidden = output[0]
                            rest = output[1:]
                        else:
                            hidden = output
                            rest = None
                        if hidden is None or hidden.dim() < 2:
                            return output
                        d_local = d.to(dtype=hidden.dtype, device=hidden.device)
                        tt_local = tt_l.to(dtype=hidden.dtype, device=hidden.device)
                        x_vl = hidden[:, -1, :]                         # (B, H)
                        m = x_vl - tt_local.unsqueeze(0)                # (B, H) modality shift
                        # Projection of m onto c^l: ((m · c) / ||c||^2) * c
                        denom = (d_local * d_local).sum().clamp(min=1e-12)  # ||c||^2
                        coef = (m * d_local).sum(dim=-1, keepdim=True) / denom  # (B, 1)
                        x_new = x_vl - a * coef * d_local               # (B, H)
                        new_hidden = hidden.clone()
                        new_hidden[:, -1, :] = x_new
                        applied[layer_id] = True
                        if rest is None:
                            return new_hidden
                        return (new_hidden,) + rest
                    return hook

                h = module.register_forward_hook(
                    make_hook(direction, alpha, x_tt_l, npz_layer)
                )
                handles.append(h)
            yield
        finally:
            for h in handles:
                h.remove()

    def generate(self, wrapper, image: Optional[Image.Image], question: str,
                 max_new_tokens: int = 256, caption: Optional[str] = None) -> str:
        x_tt_cache: Optional[dict] = None
        if image is not None and caption is not None:
            # Build TT prompt from caption and run a single forward pass
            # (not generation) to get x_tt at each intervention layer.
            tt_prompt = f"Image description: {caption}\n\n{question}"
            with torch.no_grad():
                hidden_states_tt, _, _ = wrapper.forward_text(tt_prompt)
            x_tt_cache = {}
            for l in self._directions:
                x_tt_cache[l] = hidden_states_tt[l][0, -1, :].detach()
        elif image is not None and caption is None:
            print(f"  [comp_safety_shift] WARNING: no caption for this sample; "
                  f"correction skipped (no TT baseline available).")

        with self._apply_hooks(wrapper, x_tt_cache=x_tt_cache):
            if image is not None:
                return wrapper.generate_vl(image, question,
                                           max_new_tokens=max_new_tokens)
            return wrapper.generate_text(question, max_new_tokens=max_new_tokens)
