"""
Forward-hook context managers for reading and editing the residual stream.

Both work during `model.generate()`: hooks fire once per forward call, i.e.
once for the prefill and once per decoded token (seq_len == 1 with KV cache).
`call_idx` counts those calls from 0 (0 = prefill).

Sites (per 0-based decoder layer i):
    "input"   residual stream entering layer i   (= hidden_states[i])
    "normed"  output of layer i's input RMSNorm   (g * x / RMS(x)); capture only
    "output"  residual stream leaving layer i     (= hidden_states[i + 1])

Usage
-----
    mi = get_internals(wrapper)
    with capture_residual(mi, layers=[15], site="output") as store:
        wrapper.generate_vl(image, prompt)
    store[15]        # list of (batch, d) CPU float32 tensors, one per call

    def edit(h, call_idx):          # h: (batch, seq, d), model dtype
        h[:, -1, :] += delta
        return h
    with edit_residual(mi, layer=15, edit_fn=edit):
        wrapper.generate_vl(image, prompt)
"""

from contextlib import contextmanager
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch

from src.model_internals import ModelInternals

SITES = ("input", "normed", "output")
Positions = Union[str, Sequence[int]]
EditFn = Callable[[torch.Tensor, int], Optional[torch.Tensor]]


def _output_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_output(output, new_tensor):
    if isinstance(output, tuple):
        return (new_tensor,) + tuple(output[1:])
    return new_tensor


def _input_tensor(args, kwargs):
    if args:
        return args[0]
    return kwargs["hidden_states"]


def _replace_input(args, kwargs, new_tensor):
    if args:
        return (new_tensor,) + tuple(args[1:]), kwargs
    kwargs = dict(kwargs)
    kwargs["hidden_states"] = new_tensor
    return args, kwargs


def _select(t: torch.Tensor, positions: Positions) -> torch.Tensor:
    if positions == "last":
        t = t[:, -1, :]
    elif positions != "all":
        t = t[:, list(positions), :]
    return t.detach().to(device="cpu", dtype=torch.float32).clone()


@contextmanager
def capture_residual(internals: ModelInternals, layers: Iterable[int],
                     site: str = "output", positions: Positions = "last"):
    """Record states at `site` for each layer on every forward call.

    Yields {layer: [tensor per forward call]}. positions: "last" -> (batch, d),
    "all" -> (batch, seq, d), or a list of sequence indices -> (batch, k, d).
    """
    if site not in SITES:
        raise ValueError(f"site must be one of {SITES}, got {site!r}")
    layers = list(layers)
    store: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    handles = []

    def _out_hook(l):
        def hook(_module, _inputs, output):
            store[l].append(_select(_output_tensor(output), positions))
        return hook

    def _pre_hook(l):
        def hook(_module, args, kwargs):
            store[l].append(_select(_input_tensor(args, kwargs), positions))
        return hook

    try:
        for l in layers:
            if site == "output":
                handles.append(internals.layer(l).register_forward_hook(_out_hook(l)))
            elif site == "normed":
                handles.append(internals.input_layernorm(l).register_forward_hook(_out_hook(l)))
            else:
                handles.append(internals.layer(l).register_forward_pre_hook(
                    _pre_hook(l), with_kwargs=True))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def edit_residual(internals: ModelInternals, layer: int, edit_fn: EditFn,
                  site: str = "output"):
    """Apply `edit_fn(hidden, call_idx)` to the residual stream at one layer.

    `hidden` is the full (batch, seq, d) tensor in model dtype; return the
    edited tensor, or None to leave this call unchanged. site is "input" or
    "output" (an edit at "normed" would not reach the residual stream).
    """
    if site not in ("input", "output"):
        raise ValueError(f"edit site must be 'input' or 'output', got {site!r}")
    calls = [0]

    def _out_hook(_module, _inputs, output):
        new = edit_fn(_output_tensor(output), calls[0])
        calls[0] += 1
        return None if new is None else _replace_output(output, new)

    def _pre_hook(_module, args, kwargs):
        new = edit_fn(_input_tensor(args, kwargs), calls[0])
        calls[0] += 1
        return None if new is None else _replace_input(args, kwargs, new)

    mod = internals.layer(layer)
    if site == "output":
        handle = mod.register_forward_hook(_out_hook)
    else:
        handle = mod.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
