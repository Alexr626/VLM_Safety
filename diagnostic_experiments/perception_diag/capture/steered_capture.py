"""Steered-state capture helpers for perception diagnostics (N1–N3)."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from evaluation.classifiers.metrics import _normalize_yes_no
from evaluation.interventions.vti.hooks import vti_hook_ctx
from evaluation.interventions.vti.steer import steer
from src.mediation import ScoringTarget, forward_with_logits, get_dispatch


def parse_outcome(response: str) -> str:
    p = _normalize_yes_no(response)
    return "unparseable" if p is None else p


def degeneracy_flag(response: str) -> bool:
    """Heuristic for collapsed / highly repetitive generations (S1-style)."""
    from collections import Counter

    t = (response or "").strip()
    if not t:
        return False
    words = t.lower().split()
    if len(words) < 12:
        return False
    grams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
    if grams:
        top = Counter(grams).most_common(1)[0][1]
        if top >= 4:
            return True
    uniq = len(set(words)) / max(len(words), 1)
    return uniq < 0.35 and len(words) >= 24


def response_truncated(wrapper, response: str, max_new_tokens: int) -> bool:
    """True if the decoded response is near the generation length cap."""
    tok = getattr(wrapper, "tokenizer", None)
    if tok is None:
        proc = getattr(wrapper, "processor", None)
        tok = getattr(proc, "tokenizer", None) if proc is not None else None
    if tok is None:
        # Fallback: rough whitespace bound
        return len((response or "").split()) >= max(8, int(0.9 * max_new_tokens))
    try:
        ids = tok.encode(response or "", add_special_tokens=False)
    except Exception:
        return False
    return len(ids) >= max(1, max_new_tokens - 1)


def score_yes_no_logits(wrapper, image, text: str) -> Dict[str, Any]:
    yes_t, no_t = ScoringTarget.yes_no(wrapper)
    logits, seq_len = forward_with_logits(wrapper, image, text)
    probs = torch.softmax(logits.float(), dim=-1)
    p_yes_raw = float(probs[yes_t.token_ids].sum().item())
    p_no_raw = float(probs[no_t.token_ids].sum().item())
    answer_mass = p_yes_raw + p_no_raw
    p_yes_norm = (p_yes_raw / answer_mass) if answer_mass > 0 else float("nan")
    yes_logit = float(logits[yes_t.token_ids].float().sum().item())
    no_logit = float(logits[no_t.token_ids].float().sum().item())
    first_pred = "yes" if p_yes_raw >= p_no_raw else "no"
    return {
        "p_yes_raw": p_yes_raw,
        "p_no_raw": p_no_raw,
        "answer_mass": answer_mass,
        "p_yes_norm": p_yes_norm,
        "yes_logit_sum": yes_logit,
        "no_logit_sum": no_logit,
        "logit_margin_yes_minus_no": yes_logit - no_logit,
        "first_token_pred": first_pred,
        "prefill_seq_len": int(seq_len),
    }


def _stack_hidden(hidden_states) -> Tuple[np.ndarray, np.ndarray]:
    stacks, norms = [], []
    for hs in hidden_states:
        t = hs[0].detach().float()
        stacks.append(t[-1].half().cpu().numpy())
        norms.append(t.norm(dim=-1).half().cpu().numpy())
    max_t = max(n.shape[0] for n in norms)
    norm_arr = np.full((len(norms), max_t), np.nan, dtype=np.float16)
    for i, n in enumerate(norms):
        norm_arr[i, : n.shape[0]] = n
    return np.stack(stacks, axis=0), norm_arr


@contextmanager
def optional_steer_ctx(wrapper, directions, cell: Optional[dict]):
    if cell is None or cell.get("method") in (None, "none"):
        yield
        return
    layer_indices = cell.get("layer_indices")
    with vti_hook_ctx(
        wrapper,
        directions,
        variant=cell["method"],
        alpha=float(cell["strength"]),
        hook_site=cell["site"],
        eps_coeff=float(cell.get("eps_coeff", 0.1)),
        layer_indices=layer_indices,
    ):
        yield


@contextmanager
def layer_last_token_capture_ctx(wrapper, store: Dict[int, torch.Tensor]):
    """Capture last-prefill token at each decoder layer (hidden_state / get_layer).

    Register *after* entering ``vti_hook_ctx`` so these hooks see steered outputs.
    """
    dispatch = get_dispatch(wrapper)
    handles = []

    def _make(idx: int):
        def hook(_m, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            if t.dim() == 3 and t.size(1) > 1:
                store[idx] = t[0, -1, :].detach().to(dtype=torch.float16, device="cpu").clone()
            return output
        return hook

    try:
        for i in range(wrapper.num_layers):
            handles.append(dispatch.get_layer(wrapper, i).register_forward_hook(_make(i)))
        yield
    finally:
        for h in handles:
            h.remove()


def run_capture_generation(
    wrapper,
    image,
    prompt: str,
    *,
    directions: Optional[np.ndarray],
    cell: Optional[dict],
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Steered last-prefill acts + first-token scores + generated response."""
    layer_store: Dict[int, torch.Tensor] = {}
    with optional_steer_ctx(wrapper, directions, cell):
        with layer_last_token_capture_ctx(wrapper, layer_store):
            with torch.no_grad():
                hidden, _, _ = wrapper.forward_vl(image, prompt)
            last_from_hs, pos_norms = _stack_hidden(hidden)
            scores = score_yes_no_logits(wrapper, image, prompt)
        # generate without re-capturing layer store
        response = wrapper.generate_vl(image, prompt, max_new_tokens=max_new_tokens)

    # Prefer hook-captured decoder rows; prepend embedding row from hidden_states[0]
    embed = last_from_hs[0]
    dec = []
    for i in range(wrapper.num_layers):
        if i in layer_store:
            dec.append(layer_store[i].numpy())
        else:
            dec.append(last_from_hs[i + 1])
    last_token = np.stack([embed] + dec, axis=0)

    outcome = parse_outcome(response)
    agree = outcome != "unparseable" and outcome == scores["first_token_pred"]
    truncated = response_truncated(wrapper, response, max_new_tokens)
    return {
        "response": response,
        "parsed_outcome": outcome,
        "first_vs_parsed_agree": bool(agree),
        "degeneracy_flag": bool(degeneracy_flag(response)),
        "truncated": bool(truncated),
        "status": "ok",
        "scores": scores,
        "last_token_acts_fp16": last_token,
        "prefill_pos_norms_fp16": pos_norms,
        "n_layers_plus": int(last_token.shape[0]),
        "hidden_dim": int(last_token.shape[1]),
    }


def verify_steered_capture_equality(
    wrapper,
    image,
    prompt: str,
    directions: np.ndarray,
    cell: dict,
    *,
    tol: float = 1e-3,
) -> Dict[str, Any]:
    """G1: at first steered module, steered out == offline steer(clean out).

    Captures the *same* hook site as steering (mlp or layer). Only layer 0 is
    valid (later layers compound). Capture hook is registered *after* steer
    hooks so it observes the post-steer tensor.
    """
    dispatch = get_dispatch(wrapper)
    site = cell["site"]
    # G1 probes the first steered layer (layer 0 when full-depth; first index
    # inside a window when layer_indices is set — layer 0 may be unsteered).
    layer_indices = cell.get("layer_indices")
    if layer_indices:
        probe_layer = int(min(layer_indices))
    else:
        probe_layer = 0
    mod = (
        dispatch.get_mlp(wrapper, probe_layer)
        if site == "mlp"
        else dispatch.get_layer(wrapper, probe_layer)
    )

    def _cap(store):
        def hook(_m, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            if t.dim() == 3 and t.size(1) > 1:
                store["t"] = t[0, -1, :].detach().float().cpu().clone()
            return output
        return hook

    clean_out: Dict[str, torch.Tensor] = {}
    h = mod.register_forward_hook(_cap(clean_out))
    try:
        with torch.no_grad():
            wrapper.forward_vl(image, prompt)
    finally:
        h.remove()

    steer_out: Dict[str, torch.Tensor] = {}
    with optional_steer_ctx(wrapper, directions, cell):
        h2 = mod.register_forward_hook(_cap(steer_out))
        try:
            with torch.no_grad():
                wrapper.forward_vl(image, prompt)
        finally:
            h2.remove()

    if "t" not in clean_out or "t" not in steer_out:
        return {
            "pass": False,
            "error": "capture_miss",
            "layer": probe_layer,
            "site": site,
            "method": cell.get("method"),
            "strength": cell.get("strength"),
        }

    d0 = torch.tensor(directions[probe_layer], dtype=torch.float32)
    x = clean_out["t"].view(1, 1, -1)
    offline = steer(
        x, d0, float(cell["strength"]), cell["method"], float(cell.get("eps_coeff", 0.1)),
    )[0, 0].float()
    max_abs = float((steer_out["t"] - offline).abs().max().item())
    return {
        "layer": probe_layer,
        "site": site,
        "method": cell.get("method"),
        "strength": cell.get("strength"),
        "max_abs_diff": max_abs,
        "pass": max_abs <= tol,
        "tol": tol,
    }
