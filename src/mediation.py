"""
FCCT-style causal mediation helpers for VLM wrappers.

Provides model-family dispatch, forward-with-logits, activation capture/patch
hooks, and a generalized ScoringTarget for token-probability mediation.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch


# ── Family dispatch ─────────────────────────────────────────────────────────

@dataclass
class FamilyDispatch:
    family: str
    n_layers: int
    hidden_dim: int

    def get_lm_head(self, wrapper):
        raise NotImplementedError

    def get_layer(self, wrapper, idx: int):
        raise NotImplementedError

    def get_attn(self, wrapper, idx: int):
        raise NotImplementedError

    def get_mlp(self, wrapper, idx: int):
        raise NotImplementedError


class _LLaVADispatch(FamilyDispatch):
    def get_lm_head(self, wrapper):
        return wrapper.model.language_model.lm_head

    def _layers(self, wrapper):
        return wrapper.model.language_model.model.layers

    def get_layer(self, wrapper, idx):
        return self._layers(wrapper)[idx]

    def get_attn(self, wrapper, idx):
        return self._layers(wrapper)[idx].self_attn

    def get_mlp(self, wrapper, idx):
        return self._layers(wrapper)[idx].mlp


class _LlamaDispatch(FamilyDispatch):
    def get_lm_head(self, wrapper):
        return wrapper.model.lm_head

    def _layers(self, wrapper):
        return wrapper.model.model.layers

    def get_layer(self, wrapper, idx):
        return self._layers(wrapper)[idx]

    def get_attn(self, wrapper, idx):
        return self._layers(wrapper)[idx].self_attn

    def get_mlp(self, wrapper, idx):
        return self._layers(wrapper)[idx].mlp


class _QwenVLChatDispatch(FamilyDispatch):
    def get_lm_head(self, wrapper):
        return wrapper.model.lm_head

    def _layers(self, wrapper):
        return wrapper.model.transformer.h

    def get_layer(self, wrapper, idx):
        return self._layers(wrapper)[idx]

    def get_attn(self, wrapper, idx):
        return self._layers(wrapper)[idx].attn

    def get_mlp(self, wrapper, idx):
        return self._layers(wrapper)[idx].mlp


def _family_for(wrapper) -> str:
    cls_name = type(wrapper).__name__
    if cls_name == "LLaVAWrapper":
        return "llava"
    if cls_name == "ShareGPT4VWrapper":
        return "llama_raw"
    if cls_name == "QwenVLWrapper":
        return "qwen_vl_chat"
    raise NotImplementedError(
        f"Causal mediation dispatch not implemented for wrapper {cls_name}. "
        "Supported families: LLaVAWrapper, ShareGPT4VWrapper, QwenVLWrapper."
    )


def get_dispatch(wrapper) -> FamilyDispatch:
    fam = _family_for(wrapper)
    n_layers = wrapper.num_layers
    hidden_dim = wrapper.hidden_dim
    if fam == "llava":
        return _LLaVADispatch(family=fam, n_layers=n_layers, hidden_dim=hidden_dim)
    if fam == "llama_raw":
        return _LlamaDispatch(family=fam, n_layers=n_layers, hidden_dim=hidden_dim)
    if fam == "qwen_vl_chat":
        return _QwenVLChatDispatch(family=fam, n_layers=n_layers, hidden_dim=hidden_dim)
    raise NotImplementedError(fam)


def verify_layout(wrapper, dispatch: FamilyDispatch):
    head = dispatch.get_lm_head(wrapper)
    if not isinstance(head, torch.nn.Module):
        raise RuntimeError(f"lm_head is not nn.Module: {type(head)}")
    dispatch.get_layer(wrapper, 0)
    dispatch.get_attn(wrapper, 0)
    dispatch.get_mlp(wrapper, 0)


# ── Scoring targets ─────────────────────────────────────────────────────────

def _get_tokenizer(wrapper):
    tok = getattr(wrapper, "tokenizer", None)
    if tok is None:
        proc = getattr(wrapper, "processor", None)
        tok = getattr(proc, "tokenizer", None) if proc is not None else None
    if tok is None:
        raise AttributeError(
            f"Wrapper {type(wrapper).__name__} has no tokenizer.")
    return tok


def resolve_token_ids(wrapper, variants: List[str]) -> List[int]:
    """Map text variants to deduped first-subtoken ids."""
    tok = _get_tokenizer(wrapper)
    ids = set()
    for v in variants:
        pieces = tok(v, add_special_tokens=False).get("input_ids", [])
        if not pieces:
            continue
        first = pieces[0]
        if isinstance(first, list):
            first = first[0]
        ids.add(int(first))
    return sorted(ids)


@dataclass
class ScoringTarget:
    """Token-set probability target for mediation scoring."""
    token_variants: List[str]
    token_ids: List[int] = field(default_factory=list)

    @classmethod
    def yes_no(cls, wrapper, yes_variants=None, no_variants=None) -> Tuple["ScoringTarget", "ScoringTarget"]:
        yes_variants = yes_variants or ["Yes", "yes", " YES", " yes"]
        no_variants = no_variants or ["No", "no", " NO", " no"]
        yes = cls(token_variants=yes_variants)
        no = cls(token_variants=no_variants)
        yes.token_ids = resolve_token_ids(wrapper, yes.token_variants)
        no.token_ids = resolve_token_ids(wrapper, no.token_variants)
        return yes, no

    @classmethod
    def from_variants(cls, wrapper, variants: List[str]) -> "ScoringTarget":
        t = cls(token_variants=variants)
        t.token_ids = resolve_token_ids(wrapper, variants)
        return t


def yes_token_ids(wrapper) -> List[int]:
    """Backward-compatible yes-token resolver."""
    return ScoringTarget.from_variants(wrapper, ["Yes", "yes", " YES", " yes"]).token_ids


# ── Forward-with-logits ─────────────────────────────────────────────────────

def _prepare_inputs_for_logits(wrapper, image, text):
    fam = _family_for(wrapper)
    if fam == "llava":
        return wrapper._prepare_vl(text, image), None
    if fam == "llama_raw":
        inputs_embeds, _ = wrapper._prepare_vl_embeds(image, text)
        return {"inputs_embeds": inputs_embeds}, None
    if fam == "qwen_vl_chat":
        input_ids, img_path = wrapper._prepare_vl(image, text)
        import os
        def _cleanup():
            try:
                os.unlink(img_path)
            except OSError:
                pass
        return {"input_ids": input_ids}, _cleanup
    raise NotImplementedError(fam)


@torch.no_grad()
def forward_with_logits(wrapper, image, text) -> Tuple[torch.Tensor, int]:
    forward_kwargs, cleanup = _prepare_inputs_for_logits(wrapper, image, text)
    try:
        outputs = wrapper.model(
            **forward_kwargs,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
    finally:
        if cleanup is not None:
            cleanup()
    if not hasattr(outputs, "logits"):
        raise RuntimeError(
            f"Forward pass for {type(wrapper).__name__} did not return logits.")
    logits = outputs.logits
    return logits[0, -1, :].detach(), int(logits.shape[1])


@torch.no_grad()
def target_token_probs(
    wrapper,
    image,
    text,
    target: Union[ScoringTarget, List[int]],
) -> Tuple[float, int]:
    """Return (sum P(token in target), prefill_seq_len)."""
    token_ids = target.token_ids if isinstance(target, ScoringTarget) else target
    last_logits, seq_len = forward_with_logits(wrapper, image, text)
    probs = torch.softmax(last_logits.float(), dim=-1)
    return float(probs[token_ids].sum().item()), seq_len


@torch.no_grad()
def compute_yes_prob(wrapper, image, text, yes_ids: List[int]) -> Tuple[float, int]:
    return target_token_probs(wrapper, image, text, yes_ids)


# ── Capture / patch hooks ───────────────────────────────────────────────────

COMPONENTS = ("hidden_state", "mlp", "attn")


def _output_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_tensor(output, new_tensor):
    if isinstance(output, tuple):
        return (new_tensor,) + output[1:]
    return new_tensor


def _make_capture_hook(store: Dict, key: Tuple[int, str]):
    def hook(_module, _inputs, output):
        t = _output_tensor(output)
        if t.dim() < 3 or t.shape[1] <= 1:
            return output
        store[key] = t[..., -1, :].detach().to(dtype=torch.float32, device="cpu").clone()
        return output
    return hook


def _make_patch_hook(cached: torch.Tensor):
    def hook(_module, _inputs, output):
        t = _output_tensor(output)
        if t.dim() < 3 or t.shape[1] <= 1:
            return output
        new = t.clone()
        new[..., -1, :] = cached.to(device=new.device, dtype=new.dtype)
        return _replace_tensor(output, new)
    return hook


@torch.no_grad()
def capture_clean_activations(
    wrapper,
    dispatch: FamilyDispatch,
    image,
    text,
    layers: Optional[List[int]] = None,
) -> Tuple[Dict[Tuple[int, str], torch.Tensor], int]:
    if layers is None:
        layers = list(range(dispatch.n_layers))
    store: Dict[Tuple[int, str], torch.Tensor] = {}
    handles = []
    try:
        for L in layers:
            for comp in COMPONENTS:
                if comp == "hidden_state":
                    mod = dispatch.get_layer(wrapper, L)
                elif comp == "mlp":
                    mod = dispatch.get_mlp(wrapper, L)
                else:
                    mod = dispatch.get_attn(wrapper, L)
                handles.append(mod.register_forward_hook(
                    _make_capture_hook(store, (L, comp))))
        forward_kwargs, cleanup = _prepare_inputs_for_logits(wrapper, image, text)
        try:
            wrapper.model(
                **forward_kwargs,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
            )
        finally:
            if cleanup is not None:
                cleanup()
    finally:
        for h in handles:
            h.remove()
    if not store:
        raise RuntimeError("Capture pass collected no activations.")
    return store, -1


@contextmanager
def patch_hook_ctx(wrapper, dispatch: FamilyDispatch, layer_idx: int,
                   component: str, cached_act: torch.Tensor):
    if component == "hidden_state":
        mod = dispatch.get_layer(wrapper, layer_idx)
    elif component == "mlp":
        mod = dispatch.get_mlp(wrapper, layer_idx)
    elif component == "attn":
        mod = dispatch.get_attn(wrapper, layer_idx)
    else:
        raise ValueError(f"Unknown component: {component}")
    handle = mod.register_forward_hook(_make_patch_hook(cached_act))
    try:
        yield
    finally:
        handle.remove()
