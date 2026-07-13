"""VTI visual (ViT encoder) inference-time intervention."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..base import InterventionBase
from .steer import HOOK_SITES, STEER_VARIANTS
from .visual_directions import (
    DirectionRecon,
    compute_or_load_visual_directions,
)
from .visual_hooks import vti_visual_hook_ctx
from .visual_perturb import MaskFill, PerturbType


class VTIVisualIntervention(InterventionBase):
    """Steer ViT encoder activations with PCA-derived visual VTI directions."""

    def __init__(
        self,
        model_id: str,
        *,
        variant: str = "uniform_rotation",
        hook_site: str = "mlp",
        alpha: float = 0.9,
        num_demos: int = 70,
        seed: int = 42,
        eps_coeff: float = 0.1,
        perturb_type: PerturbType = "patch_mask",
        mask_ratio: float = 0.99,
        mask_fill: MaskFill = "zero",
        noise_sigma: float = 0.1,
        num_trials: int = 50,
        direction_recon: DirectionRecon = "top_pc",
        include_cls: bool = True,
        patch_size: int = 14,
        demos_path: Optional[Path] = None,
        direction_cache: Optional[Path] = None,
        _directions: Optional[np.ndarray] = None,
    ):
        if variant not in STEER_VARIANTS:
            raise ValueError(f"variant must be one of {STEER_VARIANTS}")
        if hook_site not in HOOK_SITES:
            raise ValueError(f"hook_site must be one of {HOOK_SITES}")

        self._model_id = model_id
        self._variant = variant
        self._hook_site = hook_site
        self._alpha = alpha
        self._num_demos = num_demos
        self._seed = seed
        self._eps_coeff = eps_coeff
        self._perturb_type = perturb_type
        self._mask_ratio = mask_ratio
        self._mask_fill = mask_fill
        self._noise_sigma = noise_sigma
        self._num_trials = num_trials
        self._direction_recon = direction_recon
        self._include_cls = include_cls
        self._patch_size = patch_size
        self._demos_path = demos_path
        self._direction_cache = direction_cache
        self._directions: Optional[np.ndarray] = _directions

    @property
    def name(self) -> str:
        return f"vti_visual_{self._variant}_{self._hook_site}"

    @property
    def config(self) -> dict:
        return {
            "variant": self._variant,
            "hook_site": self._hook_site,
            "alpha": self._alpha,
            "num_demos": self._num_demos,
            "seed": self._seed,
            "eps_coeff": self._eps_coeff,
            "perturb_type": self._perturb_type,
            "mask_ratio": self._mask_ratio,
            "mask_fill": self._mask_fill,
            "noise_sigma": self._noise_sigma,
            "num_trials": self._num_trials,
            "direction_recon": self._direction_recon,
            "include_cls": self._include_cls,
            "patch_size": self._patch_size,
        }

    def ensure_directions(self, wrapper) -> np.ndarray:
        if self._directions is None:
            self._directions = compute_or_load_visual_directions(
                wrapper,
                wrapper.model_name,
                num_demos=self._num_demos,
                seed=self._seed,
                demos_path=self._demos_path,
                perturb_type=self._perturb_type,
                mask_ratio=self._mask_ratio,
                mask_fill=self._mask_fill,
                noise_sigma=self._noise_sigma,
                num_trials=self._num_trials,
                direction_recon=self._direction_recon,
                include_cls=self._include_cls,
                patch_size=self._patch_size,
            )
        return self._directions

    def generate(
        self,
        wrapper,
        image: Optional[Image.Image],
        question: str,
        max_new_tokens: int = 256,
        caption: Optional[str] = None,
    ) -> str:
        del caption
        if image is None:
            return wrapper.generate_text(question, max_new_tokens=max_new_tokens)
        directions = self.ensure_directions(wrapper)
        with vti_visual_hook_ctx(
            wrapper,
            directions,
            variant=self._variant,
            alpha=self._alpha,
            hook_site=self._hook_site,
            eps_coeff=self._eps_coeff,
            include_cls=self._include_cls,
        ):
            return wrapper.generate_vl(
                image, question, max_new_tokens=max_new_tokens,
            )
