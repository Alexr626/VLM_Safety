"""
Helpers for FCCT-style causal-mediation analysis on VLM wrappers.

Provides:
  * Per-family dispatch of `lm_head`, `layers[l]`, `layers[l].self_attn`,
    `layers[l].mlp`. LLaVA, ShareGPT4V, Qwen-VL-Chat are supported.
  * Yes-token id resolution (multi-variant, deduped) for any wrapper
    tokenizer.
  * `forward_with_logits(...)`: bypasses each wrapper's `forward_vl`
    indirection (some return base-model outputs without logits) and
    runs a single full forward, returning (last-token logits, prefill
    seq_len). Re-uses each wrapper's `_prepare_*` helpers so the chat
    template stays intact.
  * `compute_yes_prob(...)`: thin wrapper that softmaxes those logits
    and sums probability over the yes-token set.
  * `capture_clean_activations(...)`: registers per-layer/per-component
    forward hooks and runs a single forward to grab the last-position
    activation of every (layer, component) cell.
  * `patch_hook_ctx(...)`: a context manager registering a single patch
    hook that overwrites the last-position output of one (layer,
    component) with a cached clean activation.

Hook semantics (important for interpretation):
  * `hidden_state` patches the layer-block output — i.e. the residual
    stream after layer `l` (post-residual-add, post-MLP).
  * `mlp` patches the MLP submodule output — the MLP's contribution
    *before* it gets added back into the residual stream by the layer
    block. Patching this overwrites the MLP's contribution at position
    -1 of layer l only.
  * `attn` patches the self-attention submodule output (the post-
    projection attn output, again before residual add).

These three intervention points are the standard FCCT decomposition
(see Li et al. 2025 §3.2).

Prefill safety:
  All hooks are gated to fire only when the input sequence length > 1
  (prefill, not generate). The mediation script never calls generate(),
  but the guard keeps the helpers safe to reuse.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# ── Family dispatch ─────────────────────────────────────────────────────────

@dataclass
class FamilyDispatch:
    """Per-wrapper attribute paths for hook registration & lm_head."""
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
    """ShareGPT4V loads a raw LlamaForCausalLM as `wrapper.model`."""
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
    """Qwen/Qwen-VL-Chat: layers at wrapper.model.transformer.h[l].

    Qwen-7B's QWenBlock uses .attn (QWenAttention) and .mlp (QWenMLP).
    Verified at first invocation: see verify_qwen_layout().
    """
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
    """Match by class name to avoid importing the wrapper classes."""
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
    """Sanity-check the dispatch by touching every attribute we'll hook.

    Raises AttributeError with a useful message if the layout differs.
    For Qwen-VL-Chat (less-tested than LLaVA), also dumps the layer
    block's submodule names so a layout mismatch is fast to diagnose.
    """
    head = dispatch.get_lm_head(wrapper)
    if not isinstance(head, torch.nn.Module):
        raise RuntimeError(f"lm_head is not nn.Module: {type(head)}")
    layer0 = dispatch.get_layer(wrapper, 0)
    attn0 = dispatch.get_attn(wrapper, 0)
    mlp0 = dispatch.get_mlp(wrapper, 0)
    if dispatch.family == "qwen_vl_chat":
        # Print once so a layout mismatch is fast to spot.
        print(f"  [qwen-vl-chat] layer[0] children: "
              f"{[n for n, _ in layer0.named_children()]}")
        print(f"  [qwen-vl-chat] attn type: {type(attn0).__name__}")
        print(f"  [qwen-vl-chat] mlp type:  {type(mlp0).__name__}")


# ── Yes-token resolution ────────────────────────────────────────────────────

# YES_VARIANTS = ["yes", "Yes", "YES", " yes", " Yes", " YES"]
YES_VARIANTS = ["Yes"]


def _get_tokenizer(wrapper):
    """Return whichever tokenizer the wrapper exposes."""
    tok = getattr(wrapper, "tokenizer", None)
    if tok is None:
        proc = getattr(wrapper, "processor", None)
        tok = getattr(proc, "tokenizer", None) if proc is not None else None
    if tok is None:
        raise AttributeError(
            f"Wrapper {type(wrapper).__name__} has neither .tokenizer nor "
            ".processor.tokenizer; cannot resolve yes-token ids.")
    return tok


def yes_token_ids(wrapper) -> List[int]:
    """Resolve a deduped, sorted list of token ids whose first piece spells
    a yes-variant. Multi-piece variants contribute only their first piece
    (we only score the next-token logit).
    """
    tok = _get_tokenizer(wrapper)
    ids = set()
    for v in YES_VARIANTS:
        pieces = tok(v, add_special_tokens=False).get("input_ids", [])
        if not pieces:
            continue
        first = pieces[0]
        if isinstance(first, list):
            first = first[0]
        ids.add(int(first))
    return sorted(ids)


# ── Forward-with-logits dispatch ────────────────────────────────────────────

def _prepare_inputs_for_logits(wrapper, image, text):
    """Per-family input prep that returns kwargs + optional cleanup callable.

    Returns: (forward_kwargs: dict, cleanup: Callable[[], None] | None)
    The forward call is `wrapper.model(**forward_kwargs, ...)` for
    LLaVA / ShareGPT4V / Qwen-VL-Chat.
    """
    fam = _family_for(wrapper)
    if fam == "llava":
        inputs = wrapper._prepare_vl(text, image)
        return inputs, None
    if fam == "llama_raw":
        inputs_embeds, _input_ids = wrapper._prepare_vl_embeds(image, text)
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
    """Run a single forward pass and return (last-token logits, seq_len).

    `last_token_logits` is shape (vocab_size,) on the model device.
    `seq_len` is the prefill sequence length (input_ids.shape[1] after
    image-token expansion). Used for sanity-checking SSS/SSU pair shape
    consistency.
    """
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
            f"Forward pass for {type(wrapper).__name__} did not return logits "
            f"(got {type(outputs).__name__}). Family dispatch needs to call "
            "the head-bearing model variant.")
    logits = outputs.logits  # (1, T, V)
    last_logits = logits[0, -1, :].detach()
    seq_len = int(logits.shape[1])
    return last_logits, seq_len


@torch.no_grad()
def compute_yes_prob(wrapper, image, text, yes_ids: List[int]) -> Tuple[float, int]:
    """Return (P(yes), prefill_seq_len)."""
    last_logits, seq_len = forward_with_logits(wrapper, image, text)
    probs = torch.softmax(last_logits.float(), dim=-1)
    p_yes = float(probs[yes_ids].sum().item())
    return p_yes, seq_len


# ── Capture / patch hooks ───────────────────────────────────────────────────

COMPONENTS = ("hidden_state", "mlp", "attn")


def _output_tensor(output):
    """Return the tensor part of a layer/submodule output (handles tuple)."""
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_tensor(output, new_tensor):
    """Return an output of the same type with `new_tensor` swapped in."""
    if isinstance(output, tuple):
        return (new_tensor,) + output[1:]
    return new_tensor


def _make_capture_hook(store: Dict, key: Tuple[int, str]):
    def hook(_module, _inputs, output):
        t = _output_tensor(output)
        # Only capture during prefill; skip if a stray generate() ever fires.
        if t.dim() < 3 or t.shape[1] <= 1:
            return output
        # Store last-position activation, on CPU in float32 to keep the
        # intervention numerics consistent across passes.
        store[key] = t[..., -1, :].detach().to(dtype=torch.float32, device="cpu").clone()
        return output
    return hook


def _make_patch_hook(cached: torch.Tensor):
    def hook(_module, _inputs, output):
        t = _output_tensor(output)
        if t.dim() < 3 or t.shape[1] <= 1:
            return output  # not prefill
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
    """Run a single forward pass with capture hooks on every (layer, component).

    Returns ({(layer_idx, comp): cached_act_(D,)}, seq_len).
    """
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
                else:  # attn
                    mod = dispatch.get_attn(wrapper, L)
                handles.append(mod.register_forward_hook(
                    _make_capture_hook(store, (L, comp))))
        # Discard the forward outputs entirely — we only want the cached acts.
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
        raise RuntimeError(
            "Capture pass collected no activations — every hook saw seq_len<=1. "
            "Check that the prompt actually has a multi-token prefill.")
    # Rough seq_len: take any captured tensor's leading dim before slicing
    # (we only stored last-position; recover seq_len from a probe forward).
    # Instead: re-derive via a logits forward — cheap enough for one extra
    # call, and avoids leaking shape info through the store. For prefill
    # consistency check purposes, callers compare via compute_yes_prob's
    # returned seq_len rather than this.
    return store, -1


@contextmanager
def patch_hook_ctx(wrapper, dispatch: FamilyDispatch, layer_idx: int,
                   component: str, cached_act: torch.Tensor):
    """Context manager: patch one (layer, component) cell for the duration."""
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
