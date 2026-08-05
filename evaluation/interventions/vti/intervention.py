"""VTI textual (decoder-only) inference-time intervention."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from ..base import InterventionBase
from .directions import compute_or_load_textual_directions
from .directions_v2 import (
    DIFF_POLARITY,
    SELECTION_POLICY,
    STEER_COMPONENT,
    TOKEN_POLICY,
    compute_or_load_textual_directions_v2,
    demos_content_hash,
    load_textual_v2_directions,
)
from .hooks import vti_hook_ctx
from .steer import HOOK_SITES, STEER_VARIANTS


def _is_demos_v2_path(path: Optional[Path]) -> bool:
    if path is None:
        return False
    return Path(path).name == "demos_v2.jsonl"


class VTITextualIntervention(InterventionBase):
    """Steer decoder activations with textual VTI directions (PCA or precomputed)."""

    def __init__(
        self,
        model_id: str,
        *,
        variant: str = "uniform_rotation",
        hook_site: str = "mlp",
        beta: float = 0.9,
        num_demos: int = 70,
        rank: int = 1,
        seed: int = 42,
        eps_coeff: float = 0.1,
        demos_path: Optional[Path] = None,
        direction_cache: Optional[Path] = None,
        log_lambda_sim: bool = False,
        vector_dimension: Optional[str] = None,
        directions_dir: Optional[Path] = None,
        layer_indices: Optional[Sequence[int]] = None,
        layer_set_label: Optional[str] = None,
        _directions: Optional[np.ndarray] = None,
    ):
        if variant not in STEER_VARIANTS:
            raise ValueError(f"variant must be one of {STEER_VARIANTS}")
        if hook_site not in HOOK_SITES:
            raise ValueError(f"hook_site must be one of {HOOK_SITES}")

        self._model_id = model_id
        self._variant = variant
        self._hook_site = hook_site
        self._beta = beta  # textual steering coefficient (paper notation)
        self._num_demos = num_demos
        self._rank = rank
        self._seed = seed
        self._eps_coeff = eps_coeff
        self._demos_path = Path(demos_path) if demos_path is not None else None
        self._direction_cache = direction_cache
        self._log_lambda_sim = log_lambda_sim
        self._vector_dimension = vector_dimension
        self._directions_dir = (
            Path(directions_dir) if directions_dir is not None else None
        )
        self._layer_indices = (
            list(layer_indices) if layer_indices is not None else None
        )
        self._layer_set_label = layer_set_label
        self._directions_meta: Optional[dict] = None
        if self._directions_dir is not None:
            # Load metadata eagerly so `config` (used for result-dir naming)
            # is complete before the first generate call.
            _, self._directions_meta = load_textual_v2_directions(self._directions_dir)

        self._directions: Optional[np.ndarray] = _directions
        self._lambda_log: list[dict] = []
        self._demos_hash: Optional[str] = None
        if self._demos_path is not None and self._demos_path.exists():
            self._demos_hash = demos_content_hash(self._demos_path)

    @property
    def name(self) -> str:
        return f"vti_textual_{self._variant}_{self._hook_site}"

    @property
    def uses_demos_v2(self) -> bool:
        return self._vector_dimension is not None or _is_demos_v2_path(self._demos_path)

    @property
    def uses_directions_dir(self) -> bool:
        return self._directions_dir is not None

    @property
    def config(self) -> dict:
        cfg = {
            "variant": self._variant,
            "hook_site": self._hook_site,
            "beta": self._beta,
            "num_demos": self._num_demos,
            "rank": self._rank,
            "seed": self._seed,
            "eps_coeff": self._eps_coeff,
            "log_lambda_sim": self._log_lambda_sim,
        }
        if self._demos_path is not None:
            cfg["demos_path"] = str(self._demos_path)
        if self._demos_hash is not None:
            cfg["demos_content_hash_sha256_16"] = self._demos_hash
        if self.uses_directions_dir:
            meta = self._directions_meta or {}
            cfg["directions_dir"] = str(self._directions_dir)
            cfg["direction_slug"] = meta.get("slug")
            cfg["steer_reconstruction"] = meta.get("steer_reconstruction")
            cfg["demos_content_hash_sha256_16"] = meta.get(
                "content_hash_sha256_16", self._demos_hash,
            )
            cfg["dimension"] = meta.get("dimension", self._vector_dimension or "all")
            cfg["num_demos"] = meta.get("n_pairs", self._num_demos)
            cfg["selection_policy"] = meta.get("selection_policy")
            cfg["diff_polarity"] = meta.get("diff_polarity")
            cfg["token_policy"] = meta.get("token_policy")
        elif self.uses_demos_v2:
            cfg["dimension"] = self._vector_dimension or "all"
            cfg["selection_policy"] = SELECTION_POLICY
            cfg["steer_component"] = STEER_COMPONENT
            cfg["diff_polarity"] = DIFF_POLARITY
            cfg["token_policy"] = TOKEN_POLICY
            cfg["steer_reconstruction"] = "live_pc1_plus_mean"
        cfg["layer_indices"] = (
            sorted(self._layer_indices) if self._layer_indices is not None else None
        )
        cfg["layer_set_label"] = self._layer_set_label
        return cfg

    @property
    def lambda_log(self) -> list[dict]:
        return self._lambda_log

    def ensure_directions(self, wrapper) -> np.ndarray:
        if self._directions is None:
            if self.uses_directions_dir:
                directions, meta = load_textual_v2_directions(self._directions_dir)
                expected = (wrapper.num_layers, wrapper.hidden_dim)
                if directions.shape != expected:
                    raise RuntimeError(
                        f"Directions in {self._directions_dir} have shape "
                        f"{directions.shape}, expected {expected}"
                    )
                self._directions = directions
                self._directions_meta = meta
            elif self.uses_demos_v2:
                dim = self._vector_dimension or "all"
                # demos_v2 always caches rank>=2 components; steer with PC1+mean.
                rank = max(self._rank, 2)
                self._directions = compute_or_load_textual_directions_v2(
                    wrapper,
                    wrapper.model_name,
                    dimension=dim,
                    num_demos=self._num_demos,
                    rank=rank,
                    seed=self._seed,
                    demos_path=self._demos_path,
                )
            else:
                self._directions = compute_or_load_textual_directions(
                    wrapper,
                    wrapper.model_name,
                    num_demos=self._num_demos,
                    rank=self._rank,
                    seed=self._seed,
                    demos_path=self._demos_path,
                    cache_dir=self._direction_cache,
                )
        elif self.uses_directions_dir and self._directions_meta is None:
            _, self._directions_meta = load_textual_v2_directions(self._directions_dir)
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
            alpha=self._beta,  # textual beta -> geometry-agnostic steer() coefficient
            hook_site=self._hook_site,
            eps_coeff=self._eps_coeff,
            log_lambda_sim=self._log_lambda_sim,
            log_records=self._lambda_log,
            layer_indices=self._layer_indices,
        ):
            if image is not None:
                return wrapper.generate_vl(
                    image, question, max_new_tokens=max_new_tokens,
                )
            return wrapper.generate_text(question, max_new_tokens=max_new_tokens)
