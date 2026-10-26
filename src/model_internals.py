"""
Uniform access to the language-model internals of the supported VLMs.

Every signal and edit in this project reads the decoder's residual stream,
its RMSNorm gains, attention projections, and rotary embedding. This module
hides where those live for each wrapper family (and across transformers
versions, which moved the LLaVA / Qwen2.5-VL submodules after 4.50).

Layer-index convention (same as HF `output_hidden_states`):
    hidden_states[0]   = embedding output
    hidden_states[l]   = output of decoder layer l-1   (l = 1..num_layers)
`layer(i)` below takes a 0-based decoder-layer index i, whose output is
hidden_states[i + 1].

Usage
-----
    from src.model import create_wrapper
    from src.model_internals import get_internals
    wrapper = create_wrapper("llava-hf/llava-1.5-7b-hf").load()
    mi = get_internals(wrapper)
    gain, eps = mi.rmsnorm_params(15)
    q_proj, k_proj = mi.qk_proj(15)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


_DECODER_PATHS = {
    "llava": ("language_model.model", "model.language_model", "language_model"),
    "qwen2_vl": ("model.language_model", "model"),
}
_LM_HEAD_PATHS = ("lm_head", "language_model.lm_head")


def _resolve(root: nn.Module, dotted: str) -> Optional[nn.Module]:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def family_for(wrapper) -> str:
    cls_name = type(wrapper).__name__
    if cls_name == "LLaVAWrapper":
        return "llava"
    if cls_name == "Qwen2VLWrapper":
        return "qwen2_vl"
    raise NotImplementedError(
        f"No internals dispatch for wrapper {cls_name}. "
        "Supported: LLaVAWrapper, Qwen2VLWrapper."
    )


@dataclass
class ModelInternals:
    wrapper: object
    family: str
    decoder: nn.Module
    lm_head: nn.Module

    # ── Structure ──────────────────────────────────────────────────────────
    @property
    def layers(self) -> nn.ModuleList:
        return self.decoder.layers

    @property
    def num_layers(self) -> int:
        return len(self.decoder.layers)

    def layer(self, idx: int) -> nn.Module:
        return self.decoder.layers[idx]

    def attn(self, idx: int) -> nn.Module:
        return self.layer(idx).self_attn

    def mlp(self, idx: int) -> nn.Module:
        return self.layer(idx).mlp

    def input_layernorm(self, idx: int) -> nn.Module:
        """RMSNorm applied to the residual stream before attention."""
        return self.layer(idx).input_layernorm

    def post_attention_layernorm(self, idx: int) -> nn.Module:
        return self.layer(idx).post_attention_layernorm

    @property
    def final_norm(self) -> nn.Module:
        return self.decoder.norm

    @property
    def rotary_emb(self) -> Optional[nn.Module]:
        return getattr(self.decoder, "rotary_emb", None)

    # ── Parameters read by attention / probe signals ───────────────────────
    def rmsnorm_params(self, idx: int) -> Tuple[torch.Tensor, float]:
        """(gain g, eps) of layer idx's input RMSNorm."""
        norm = self.input_layernorm(idx)
        eps = getattr(norm, "variance_epsilon", None)
        if eps is None:
            eps = getattr(norm, "eps")
        return norm.weight.detach(), float(eps)

    def qk_proj(self, idx: int) -> Tuple[nn.Linear, nn.Linear]:
        """(q_proj, k_proj) of layer idx. Qwen's q_proj carries a bias."""
        a = self.attn(idx)
        return a.q_proj, a.k_proj

    @property
    def text_config(self):
        cfg = self.wrapper.model.config
        return getattr(cfg, "text_config", None) or cfg

    @property
    def num_heads(self) -> int:
        return self.text_config.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return getattr(self.text_config, "num_key_value_heads", None) or self.num_heads

    @property
    def head_dim(self) -> int:
        hd = getattr(self.text_config, "head_dim", None)
        return hd or self.text_config.hidden_size // self.num_heads

    # ── Token positions ────────────────────────────────────────────────────
    def image_token_span(self, input_ids: torch.Tensor) -> Tuple[Optional[int], Optional[int]]:
        return self.wrapper.get_image_token_span(input_ids)

    def bos_position(self, input_ids: torch.Tensor) -> Optional[int]:
        """0 if the sequence starts with BOS, else None (Qwen2.5-VL has no BOS;
        callers pick the first token or an empirically measured sink)."""
        bos_id = getattr(self.wrapper.tokenizer, "bos_token_id", None)
        row = input_ids[0] if input_ids.dim() == 2 else input_ids
        if bos_id is not None and int(row[0].item()) == bos_id:
            return 0
        return None


def get_internals(wrapper) -> ModelInternals:
    if wrapper.model is None:
        raise RuntimeError("Call wrapper.load() before get_internals().")
    fam = family_for(wrapper)
    decoder = None
    for path in _DECODER_PATHS[fam]:
        cand = _resolve(wrapper.model, path)
        if cand is not None and hasattr(cand, "layers") and hasattr(cand, "norm"):
            decoder = cand
            break
    if decoder is None:
        raise RuntimeError(
            f"Could not locate the language-model decoder for {fam}; "
            f"tried {_DECODER_PATHS[fam]}.")
    lm_head = None
    for path in _LM_HEAD_PATHS:
        cand = _resolve(wrapper.model, path)
        if isinstance(cand, nn.Module):
            lm_head = cand
            break
    if lm_head is None:
        raise RuntimeError(f"Could not locate lm_head; tried {_LM_HEAD_PATHS}.")
    return ModelInternals(wrapper=wrapper, family=fam, decoder=decoder, lm_head=lm_head)
