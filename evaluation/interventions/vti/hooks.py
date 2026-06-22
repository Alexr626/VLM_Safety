"""Forward-hook context manager for VTI textual steering."""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional

import torch

from src.mediation import FamilyDispatch, get_dispatch, verify_layout

from .steer import HOOK_SITES, steer


def _output_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_tensor(output, new_tensor):
    if isinstance(output, tuple):
        return (new_tensor,) + output[1:]
    return new_tensor


def _resolve_module(dispatch: FamilyDispatch, wrapper, layer_idx: int, hook_site: str):
    if hook_site == "mlp":
        return dispatch.get_mlp(wrapper, layer_idx)
    if hook_site == "layer":
        return dispatch.get_layer(wrapper, layer_idx)
    raise ValueError(f"Unknown hook_site '{hook_site}'. Choose from {HOOK_SITES}.")


def _decode_tokens(wrapper, input_ids: Optional[torch.Tensor]) -> Optional[List[str]]:
    if input_ids is None:
        return None
    tok = getattr(wrapper, "tokenizer", None)
    if tok is None:
        proc = getattr(wrapper, "processor", None)
        tok = getattr(proc, "tokenizer", None) if proc is not None else None
    if tok is None:
        return None
    row = input_ids[0] if input_ids.dim() == 2 else input_ids
    return [tok.decode([int(t)]) for t in row.tolist()]


@contextmanager
def vti_hook_ctx(
    wrapper,
    directions,  # np.ndarray (num_layers, hidden_dim) or list of tensors
    *,
    variant: str,
    alpha: float,
    hook_site: str,
    eps_coeff: float = 0.1,
    log_lambda_sim: bool = False,
    log_records: Optional[List[dict]] = None,
    decode_input_ids: Optional[torch.Tensor] = None,
    steer_prefill: bool = True,
    skip_first_token: bool = False,
):
    """Register per-layer steer hooks; always removed on exit.

    Debug knobs (default-off; defaults reproduce production behavior exactly):
        steer_prefill: if False, skip steering on multi-token (prefill) calls,
            steering only single-token decode steps.
        skip_first_token: if True, leave sequence position 0 unsteered during
            prefill (the BOS / attention-sink position).
    """
    dispatch = get_dispatch(wrapper)
    verify_layout(wrapper, dispatch)

    if hook_site not in HOOK_SITES:
        raise ValueError(f"Unknown hook_site '{hook_site}'")

    n_layers = wrapper.num_layers
    if len(directions) != n_layers:
        raise ValueError(
            f"Expected {n_layers} direction vectors, got {len(directions)}"
        )

    token_strings = _decode_tokens(wrapper, decode_input_ids)
    handles = []

    def _make_hook(layer_idx: int):
        direction = directions[layer_idx]
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction, device=wrapper.device)

        def hook(_module, _inputs, output):
            t = _output_tensor(output)
            if t.dim() != 3:
                return output
            is_prefill = t.size(1) > 1
            if is_prefill and not steer_prefill:
                return output
            steered = steer(
                t,
                direction.to(device=t.device),
                alpha,
                variant,
                eps_coeff,
                log_lambda_sim=log_lambda_sim and variant == "gated_rotation",
                log_records=log_records,
                layer_idx=layer_idx,
                token_strings=token_strings,
            )
            if skip_first_token and is_prefill:
                steered = steered.clone()
                steered[:, 0, :] = t[:, 0, :]
            return _replace_tensor(output, steered)

        return hook

    try:
        for layer_idx in range(n_layers):
            mod = _resolve_module(dispatch, wrapper, layer_idx, hook_site)
            handles.append(mod.register_forward_hook(_make_hook(layer_idx)))
        yield
    finally:
        for h in handles:
            h.remove()
