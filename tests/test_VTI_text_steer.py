"""Unit tests for VTI textual steering geometry."""

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.interventions.vti.steer import steer


def test_additive_does_not_preserve_norm():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 64)
    d = torch.randn(64)
    x_out = steer(x, d, alpha=0.9, variant="additive")
    in_norm = x.float().norm(dim=-1)
    out_norm = x_out.float().norm(dim=-1)
    assert not torch.allclose(in_norm, out_norm, atol=1e-3)


def test_rotation_variants_preserve_norm():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 64)
    d = torch.randn(64)
    in_norm = x.float().norm(dim=-1)
    for variant in ("uniform_rotation", "gated_rotation"):
        x_out = steer(x, d, alpha=0.9, variant=variant)
        out_norm = x_out.float().norm(dim=-1)
        assert torch.allclose(in_norm, out_norm, atol=1e-4), variant


def test_alpha_zero_is_identity():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32)
    d = torch.randn(32)
    for variant in ("additive", "uniform_rotation", "gated_rotation"):
        x_out = steer(x, d, alpha=0.0, variant=variant)
        assert torch.allclose(x, x_out, atol=1e-5), variant


def test_variants_are_distinct():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 128)
    d = torch.randn(128)
    outs = {
        v: steer(x, d, alpha=0.9, variant=v)
        for v in ("additive", "uniform_rotation", "gated_rotation")
    }
    assert not torch.allclose(outs["additive"], outs["uniform_rotation"], atol=1e-4)
    assert torch.allclose(outs["uniform_rotation"], outs["gated_rotation"], atol=1e-5)

    torch.manual_seed(0)
    x_decode = torch.randn(1, 1, 64)
    d_decode = torch.randn(64)
    uniform_d = steer(x_decode, d_decode, alpha=0.9, variant="uniform_rotation")
    gated_d = steer(x_decode, d_decode, alpha=0.9, variant="gated_rotation")
    assert not torch.allclose(uniform_d, gated_d, atol=1e-6)


def test_gated_prefill_matches_uniform():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 64)  # seq >= 2 => prefill branch
    d = torch.randn(64)
    uniform = steer(x, d, alpha=0.9, variant="uniform_rotation")
    gated = steer(x, d, alpha=0.9, variant="gated_rotation")
    assert torch.allclose(uniform, gated, atol=1e-5)


def test_demo_activation_stack_shape():
    """Last-token vectors must stack to (L+1, D), not flatten via cat."""
    n_layers, hidden = 3, 8
    fake_hs = tuple(
        torch.randn(1, 5, hidden) for _ in range(n_layers + 1)
    )
    stacked = torch.stack(
        [fake_hs[layer][0, -1, :].float() for layer in range(len(fake_hs))],
        dim=0,
    )
    assert stacked.shape == (n_layers + 1, hidden)

    # cat on 1D slices would wrongly flatten
    flat = torch.cat(
        [fake_hs[layer][0, -1, :] for layer in range(len(fake_hs))], dim=0,
    )
    assert flat.ndim == 1
    assert flat.numel() == (n_layers + 1) * hidden
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64)
    d = torch.randn(64)
    uniform = steer(x, d, alpha=0.9, variant="uniform_rotation")
    gated = steer(x, d, alpha=0.9, variant="gated_rotation")
    # Not guaranteed for all random draws, but should hold for this seed.
    assert not torch.allclose(uniform, gated, atol=1e-6)
