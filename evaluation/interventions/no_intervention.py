"""No-intervention baseline (direct model generation)."""

from typing import Optional
from PIL import Image

from .base import InterventionBase


class NoIntervention(InterventionBase):
    """No intervention; direct model generation."""

    @property
    def name(self) -> str:
        return "no_intervention"

    def generate(self, wrapper, image: Optional[Image.Image], question: str,
                 max_new_tokens: int = 256, caption: Optional[str] = None) -> str:
        if image is not None:
            return wrapper.generate_vl(image, question,
                                       max_new_tokens=max_new_tokens)
        return wrapper.generate_text(question, max_new_tokens=max_new_tokens)
