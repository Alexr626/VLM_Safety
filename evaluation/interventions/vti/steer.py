"""VTI steering geometries (additive, uniform_rotation, gated_rotation)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

STEER_VARIANTS = ("additive", "uniform_rotation", "gated_rotation")
HOOK_SITES = ("mlp", "layer")


def steer(
    x: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
    variant: str,
    eps_coeff: float = 0.1,
    *,
    log_lambda_sim: bool = False,
    log_records: Optional[List[dict]] = None,
    layer_idx: Optional[int] = None,
    token_strings: Optional[List[str]] = None,
) -> torch.Tensor:
    """Apply a per-layer steering direction to activations ``x``.

    Args:
        x: ``(batch, seq, hidden)`` decoder activation tensor.
        direction: ``(hidden,)`` unit direction for this layer (unnormalized ok).
        variant: ``additive`` | ``uniform_rotation`` | ``gated_rotation``.
        eps_coeff: rotation blend coefficient (reference default 0.1).
        log_lambda_sim: when True and variant is ``gated_rotation``, append
            per-token diagnostics to ``log_records``.
        log_records: mutable list receiving dicts with keys
            ``layer``, ``token_idx``, ``token``, ``lambda_sim``, ``cos_sim_pre_clamp``.
        layer_idx: layer index for logging.
        token_strings: decoded token strings for the current sequence positions.
    """
    if variant not in STEER_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {STEER_VARIANTS}.")

    d = F.normalize(direction.float(), dim=-1)
    if variant == "additive":
        return (x.float() + alpha * d).to(x.dtype)

    norm = x.float().norm(dim=-1, keepdim=True)
    if variant == "gated_rotation" and x.size(1) < 2:
        cos_sim = F.cosine_similarity(
            x.float(), -d[None, None, :], dim=-1,
        )
        lam = 1.0 + torch.clamp(cos_sim, min=0.0).unsqueeze(-1)
        if log_lambda_sim and log_records is not None:
            cos_cpu = cos_sim.detach().cpu()
            lam_cpu = lam.squeeze(-1).detach().cpu()
            for tok_i in range(x.size(1)):
                log_records.append({
                    "layer": layer_idx,
                    "token_idx": tok_i,
                    "token": (
                        token_strings[tok_i]
                        if token_strings and tok_i < len(token_strings)
                        else None
                    ),
                    "lambda_sim": float(lam_cpu[tok_i].item()),
                    "cos_sim_pre_clamp": float(cos_cpu[tok_i].item()),
                })
    else:
        lam = 1.0

    y = alpha * lam * d
    x_out = F.normalize(F.normalize(x.float(), dim=-1) + eps_coeff * y, dim=-1) * norm
    return x_out.to(x.dtype)
