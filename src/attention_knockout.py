"""Attention knockout (Geva et al. 2023 / Neo et al. 2024) for VLM decoders.

Blocks selected (query, key) edges by adding −inf to the pre-softmax attention
mask at chosen decoder layers. Remaining attention renormalizes via softmax /
SDPA. Flash-attn is not installed in this environment; SDPA accepts 4D masks.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from src.mediation import (
    _family_for,
    _prepare_inputs_for_logits,
    get_dispatch,
)


def build_knockout_additive_mask(
    *,
    batch_size: int,
    seq_len: int,
    query_positions: Sequence[int],
    key_positions: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """4D additive mask ``(B, 1, Q, K)``: 0 allow, dtype.min block."""
    mask = torch.zeros(
        batch_size, 1, seq_len, seq_len, device=device, dtype=dtype
    )
    if not query_positions or not key_positions:
        return mask
    q_idx = [q for q in query_positions if 0 <= int(q) < seq_len]
    k_idx = [k for k in key_positions if 0 <= int(k) < seq_len]
    if not q_idx or not k_idx:
        return mask
    block_val = torch.finfo(dtype).min
    # Vectorized: mask[:, :, q, k] = min for all pairs
    q_t = torch.tensor(q_idx, device=device, dtype=torch.long)
    k_t = torch.tensor(k_idx, device=device, dtype=torch.long)
    mask[:, :, q_t[:, None], k_t[None, :]] = block_val
    return mask


def query_positions_for_scope(
    scope: str,
    *,
    clause_end: int,
    seq_len: int,
) -> List[int]:
    """``block_last_token_reading`` or ``block_all_downstream_reading``."""
    if scope in ("block_last_token_reading", "last_token", "last"):
        return [seq_len - 1]
    if scope in ("block_all_downstream_reading", "all_downstream", "downstream"):
        return list(range(clause_end, seq_len))
    raise ValueError(f"Unknown query scope: {scope!r}")


def _merge_masks(
    existing: Optional[torch.Tensor],
    knockout: torch.Tensor,
) -> torch.Tensor:
    if existing is None:
        return knockout
    if existing.dtype != knockout.dtype:
        existing = existing.to(dtype=knockout.dtype)
    if existing.dim() == 2:
        # (B, K) padding mask → broadcast additive (0 / min)
        bsz, kv = existing.shape
        q = knockout.shape[2]
        # Treat 0 as blocked in HF 2D masks; nonzero as keep — convert to additive
        # Common HF: 1 = keep, 0 = mask. Convert keep→0, mask→min.
        keep = existing.to(dtype=knockout.dtype)
        additive = torch.zeros(
            bsz, 1, q, kv, device=knockout.device, dtype=knockout.dtype
        )
        additive = additive.masked_fill(keep[:, None, None, :] <= 0, torch.finfo(knockout.dtype).min)
        # If shapes mismatch on kv, fall back to knockout-only on top of broadcast
        if additive.shape[-1] != knockout.shape[-1]:
            return knockout
        return additive + knockout
    if existing.dim() == 4:
        if existing.shape[-2:] != knockout.shape[-2:]:
            # Causal mask sometimes shaped (B, 1, Q, K) with Q==K; require match
            if (
                existing.shape[0] == knockout.shape[0]
                and existing.shape[-1] == knockout.shape[-1]
                and existing.shape[-2] == 1
            ):
                existing = existing.expand(-1, -1, knockout.shape[-2], -1)
            else:
                raise RuntimeError(
                    f"Cannot merge attention masks of shapes {tuple(existing.shape)} "
                    f"and {tuple(knockout.shape)}"
                )
        return existing + knockout
    raise RuntimeError(f"Unsupported attention_mask dim={existing.dim()}")


@contextmanager
def attention_knockout_ctx(
    wrapper,
    *,
    layer_indices: Iterable[int],
    query_positions: Sequence[int],
    key_positions: Sequence[int],
):
    """Inject 4D knockout masks into decoder self-attn at selected layers."""
    dispatch = get_dispatch(wrapper)
    layers = sorted({int(i) for i in layer_indices})
    handles = []
    # Cache one knockout tensor per seq_len/dtype/device
    cache: dict = {}

    def _make_hook(_layer_idx: int):
        def hook(module, args, kwargs):  # noqa: ARG001
            args = list(args) if args is not None else []
            hidden = kwargs.get("hidden_states", None)
            if hidden is None and args:
                hidden = args[0]
            if hidden is None:
                return tuple(args), kwargs
            bsz, seq_len, _ = hidden.shape
            device = hidden.device
            # Match attn compute dtype (often float16/bfloat16)
            dtype = hidden.dtype
            key = (bsz, seq_len, device, dtype)
            if key not in cache:
                cache[key] = build_knockout_additive_mask(
                    batch_size=bsz,
                    seq_len=seq_len,
                    query_positions=query_positions,
                    key_positions=key_positions,
                    device=device,
                    dtype=dtype,
                )
            knockout = cache[key]
            existing = kwargs.get("attention_mask", None)
            if existing is None and len(args) >= 2:
                existing = args[1]
                args[1] = _merge_masks(existing, knockout)
            else:
                kwargs["attention_mask"] = _merge_masks(existing, knockout)
            return tuple(args), kwargs

        return hook

    for idx in layers:
        attn = dispatch.get_attn(wrapper, idx)
        handles.append(
            attn.register_forward_pre_hook(_make_hook(idx), with_kwargs=True)
        )
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def forward_with_logits_and_ids(wrapper, image, text):
    """Like ``forward_with_logits`` but also returns prepared ``input_ids``."""
    forward_kwargs, cleanup = _prepare_inputs_for_logits(wrapper, image, text)
    try:
        outputs = wrapper.model(
            **forward_kwargs,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
        input_ids = forward_kwargs.get("input_ids")
        if not hasattr(outputs, "logits"):
            raise RuntimeError(
                f"Forward pass for {type(wrapper).__name__} did not return logits."
            )
        logits = outputs.logits
        return logits[0, -1, :].detach(), int(logits.shape[1]), input_ids
    finally:
        if cleanup is not None:
            cleanup()


@torch.no_grad()
def forward_with_knockout(
    wrapper,
    image,
    text,
    *,
    layer_indices: Sequence[int],
    query_positions: Sequence[int],
    key_positions: Sequence[int],
) -> Tuple[torch.Tensor, int]:
    """Prefill forward with attention knockout; returns (last_logits, seq_len)."""
    with attention_knockout_ctx(
        wrapper,
        layer_indices=layer_indices,
        query_positions=query_positions,
        key_positions=key_positions,
    ):
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
            f"Forward pass for {type(wrapper).__name__} did not return logits."
        )
    logits = outputs.logits
    return logits[0, -1, :].detach(), int(logits.shape[1])


@torch.no_grad()
def verify_knockout_zero_attention(
    wrapper,
    image,
    text,
    *,
    layer_indices: Sequence[int],
    query_positions: Sequence[int],
    key_positions: Sequence[int],
    atol: float = 1e-6,
) -> dict:
    """Eager-mode check: blocked edges have ~0 post-softmax weight.

    Temporarily forces ``attn_implementation='eager'`` when supported, runs one
    forward with ``output_attentions=True``, asserts blocked (q,k) mass is ~0.
    """
    model = wrapper.model
    fam = _family_for(wrapper)
    # Resolve decoder config
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
    targets = [c for c in (cfg, text_cfg) if c is not None]

    prev = []
    for c in targets:
        prev.append((c, getattr(c, "_attn_implementation", None)))
        try:
            c._attn_implementation = "eager"
        except Exception:
            pass

    # Also try model-level attribute used by some HF versions
    prev_model_impl = getattr(model, "_attn_implementation", None)
    if hasattr(model, "_attn_implementation"):
        try:
            model._attn_implementation = "eager"
        except Exception:
            pass

    try:
        with attention_knockout_ctx(
            wrapper,
            layer_indices=layer_indices,
            query_positions=query_positions,
            key_positions=key_positions,
        ):
            forward_kwargs, cleanup = _prepare_inputs_for_logits(
                wrapper, image, text
            )
            try:
                outputs = wrapper.model(
                    **forward_kwargs,
                    output_hidden_states=False,
                    output_attentions=True,
                    use_cache=False,
                )
            finally:
                if cleanup is not None:
                    cleanup()

        attns = outputs.attentions
        if attns is None:
            raise RuntimeError(
                "output_attentions=True returned None; eager mode may be unavailable"
            )

        max_blocked = 0.0
        checked_layers = []
        for layer_idx in layer_indices:
            if layer_idx >= len(attns) or attns[layer_idx] is None:
                continue
            # (B, H, Q, K)
            a = attns[layer_idx][0].float()
            for q in query_positions:
                if q < 0 or q >= a.shape[-2]:
                    continue
                for k in key_positions:
                    if k < 0 or k >= a.shape[-1]:
                        continue
                    val = float(a[:, q, k].max().item())
                    max_blocked = max(max_blocked, val)
            checked_layers.append(int(layer_idx))

        ok = max_blocked <= atol
        if not ok:
            raise AssertionError(
                f"Knockout verification failed: max blocked attention={max_blocked} "
                f"> atol={atol} (family={fam}, layers={checked_layers})"
            )
        return {
            "ok": True,
            "max_blocked_attention": max_blocked,
            "checked_layers": checked_layers,
            "family": fam,
        }
    finally:
        for c, impl in prev:
            if impl is None:
                continue
            try:
                c._attn_implementation = impl
            except Exception:
                pass
        if prev_model_impl is not None and hasattr(model, "_attn_implementation"):
            try:
                model._attn_implementation = prev_model_impl
            except Exception:
                pass
