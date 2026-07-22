"""Zero-GPU unit tests for vti_hook_ctx layer_indices registration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.interventions.vti.hooks import vti_hook_ctx  # noqa: E402


class _FakeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hook_count = 0

    def register_forward_hook(self, hook):  # noqa: ARG002
        self.hook_count += 1
        handle = MagicMock()
        handle.remove = MagicMock()
        return handle


def _make_fake_wrapper(n_layers: int = 8, hidden: int = 16):
    wrapper = MagicMock()
    wrapper.num_layers = n_layers
    wrapper.hidden_dim = hidden
    wrapper.device = torch.device("cpu")
    wrapper.tokenizer = None
    return wrapper


def _make_fake_dispatch(n_layers: int):
    mlps = [_FakeModule() for _ in range(n_layers)]
    layers = [_FakeModule() for _ in range(n_layers)]
    dispatch = MagicMock()
    dispatch.get_mlp = MagicMock(side_effect=lambda _w, i: mlps[i])
    dispatch.get_layer = MagicMock(side_effect=lambda _w, i: layers[i])
    dispatch.get_lm_head = MagicMock(return_value=torch.nn.Linear(4, 4))
    dispatch.get_attn = MagicMock(return_value=_FakeModule())
    return dispatch, mlps, layers


def test_layer_indices_registers_only_requested_mlp_hooks():
    n_layers, hidden = 8, 16
    wrapper = _make_fake_wrapper(n_layers, hidden)
    dispatch, mlps, layers = _make_fake_dispatch(n_layers)
    directions = np.random.randn(n_layers, hidden).astype(np.float32)
    wanted = [2, 3, 5]

    with patch("evaluation.interventions.vti.hooks.get_dispatch", return_value=dispatch), \
         patch("evaluation.interventions.vti.hooks.verify_layout"):
        with vti_hook_ctx(
            wrapper,
            directions,
            variant="additive",
            alpha=0.5,
            hook_site="mlp",
            layer_indices=wanted,
        ):
            for i, m in enumerate(mlps):
                if i in wanted:
                    assert m.hook_count == 1, f"mlp layer {i} should have a hook"
                else:
                    assert m.hook_count == 0, f"mlp layer {i} should have no hook"
            for m in layers:
                assert m.hook_count == 0


def test_layer_indices_registers_only_requested_layer_hooks():
    n_layers, hidden = 8, 16
    wrapper = _make_fake_wrapper(n_layers, hidden)
    dispatch, mlps, layers = _make_fake_dispatch(n_layers)
    directions = np.random.randn(n_layers, hidden).astype(np.float32)
    wanted = [0, 7]

    with patch("evaluation.interventions.vti.hooks.get_dispatch", return_value=dispatch), \
         patch("evaluation.interventions.vti.hooks.verify_layout"):
        with vti_hook_ctx(
            wrapper,
            directions,
            variant="uniform_rotation",
            alpha=0.9,
            hook_site="layer",
            layer_indices=wanted,
        ):
            for i, m in enumerate(layers):
                if i in wanted:
                    assert m.hook_count == 1, f"layer {i} should have a hook"
                else:
                    assert m.hook_count == 0, f"layer {i} should have no hook"
            for m in mlps:
                assert m.hook_count == 0


def test_layer_indices_none_registers_all_layers():
    n_layers, hidden = 6, 8
    wrapper = _make_fake_wrapper(n_layers, hidden)
    dispatch, mlps, _layers = _make_fake_dispatch(n_layers)
    directions = np.random.randn(n_layers, hidden).astype(np.float32)

    with patch("evaluation.interventions.vti.hooks.get_dispatch", return_value=dispatch), \
         patch("evaluation.interventions.vti.hooks.verify_layout"):
        with vti_hook_ctx(
            wrapper,
            directions,
            variant="additive",
            alpha=0.2,
            hook_site="mlp",
            layer_indices=None,
        ):
            assert all(m.hook_count == 1 for m in mlps)
