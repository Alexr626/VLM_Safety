"""VTI textual (decoder-only) inference-time intervention."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..base import InterventionBase
from .directions import compute_or_load_textual_directions
from .hooks import vti_hook_ctx
from .steer import HOOK_SITES, STEER_VARIANTS


class VTITextualIntervention(InterventionBase):
    """Steer decoder activations with PCA-derived textual VTI directions."""

    def __init__(
        self,
        model_id: str,
        *,
        variant: str = "uniform_rotation",
        hook_site: str = "mlp",
        alpha_text: float = 0.9,
        num_demos: int = 70,
        rank: int = 1,
        seed: int = 42,
        eps_coeff: float = 0.1,
        demos_path: Optional[Path] = None,
        direction_cache: Optional[Path] = None,
        log_lambda_sim: bool = False,
        _directions: Optional[np.ndarray] = None,
    ):
        if variant not in STEER_VARIANTS:
            raise ValueError(f"variant must be one of {STEER_VARIANTS}")
        if hook_site not in HOOK_SITES:
            raise ValueError(f"hook_site must be one of {HOOK_SITES}")

        self._model_id = model_id
        self._variant = variant
        self._hook_site = hook_site
        self._alpha_text = alpha_text
        self._num_demos = num_demos
        self._rank = rank
        self._seed = seed
        self._eps_coeff = eps_coeff
        self._demos_path = demos_path
        self._direction_cache = direction_cache
        self._log_lambda_sim = log_lambda_sim

        self._directions: Optional[np.ndarray] = _directions
        self._lambda_log: list[dict] = []

    @property
    def name(self) -> str:
        return f"vti_textual_{self._variant}_{self._hook_site}"

    @property
    def config(self) -> dict:
        return {
            "variant": self._variant,
            "hook_site": self._hook_site,
            "alpha_text": self._alpha_text,
            "num_demos": self._num_demos,
            "rank": self._rank,
            "seed": self._seed,
            "eps_coeff": self._eps_coeff,
            "log_lambda_sim": self._log_lambda_sim,
        }

    @property
    def lambda_log(self) -> list[dict]:
        return self._lambda_log

    def ensure_directions(self, wrapper) -> np.ndarray:
        if self._directions is None:
            self._directions = compute_or_load_textual_directions(
                wrapper,
                wrapper.model_name,
                num_demos=self._num_demos,
                rank=self._rank,
                seed=self._seed,
                demos_path=self._demos_path,
                cache_dir=self._direction_cache,
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
        del caption  # VTI textual arm does not use TT captions at inference.
        directions = self.ensure_directions(wrapper)
        self._lambda_log = []

        with vti_hook_ctx(
            wrapper,
            directions,
            variant=self._variant,
            alpha=self._alpha_text,
            hook_site=self._hook_site,
            eps_coeff=self._eps_coeff,
            log_lambda_sim=self._log_lambda_sim,
            log_records=self._lambda_log,
        ):
            if image is not None:
                return wrapper.generate_vl(
                    image, question, max_new_tokens=max_new_tokens,
                )
            return wrapper.generate_text(question, max_new_tokens=max_new_tokens)
