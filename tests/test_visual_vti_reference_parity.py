"""Unit tests for visual VTI direction math (no GPU)."""

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.interventions.vti.pca import PCA
from evaluation.interventions.vti.visual_directions import (
    reconstruct_direction_from_diffs,
)


def _reference_legacy_direction(demo_diffs):
    """Vendored ``obtain_visual_vti`` math on pre-captured diffs."""
    n_layers, n_tokens, feat_dim = demo_diffs[0].shape
    hidden_states_all = []
    for diff in demo_diffs:
        hidden_states_all.append(diff.reshape(n_tokens, -1))
    fit_data = torch.stack(hidden_states_all, dim=1)
    pca = PCA(n_components=1).to(fit_data.device).fit(fit_data.float())
    direction = (
        pca.components_.sum(dim=1, keepdim=True) + pca.mean_
    ).mean(1).view(n_layers, n_tokens, feat_dim)
    return direction[1:]


def test_legacy_math_parity_on_synthetic_diffs():
    torch.manual_seed(0)
    n_demos = 8
    demo_diffs = [
        torch.randn(25, 577, 1024) for _ in range(n_demos)
    ]
    ours = reconstruct_direction_from_diffs(
        demo_diffs, "legacy_pc_plus_mean", include_cls=True,
    )
    ref = _reference_legacy_direction(demo_diffs)
    assert ours.shape == ref.shape == (24, 577, 1024)
    cos = torch.nn.functional.cosine_similarity(
        ours.reshape(-1, 1024).float(),
        ref.reshape(-1, 1024).float(),
        dim=-1,
    )
    assert (cos > 0.9999).all()


def test_top_pc_shape():
    torch.manual_seed(1)
    demo_diffs = [torch.randn(25, 577, 1024) for _ in range(5)]
    out = reconstruct_direction_from_diffs(demo_diffs, "top_pc", include_cls=True)
    assert out.shape == (24, 577, 1024)
