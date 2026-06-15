"""Abstract base class for inference-time hallucination mitigation interventions."""

from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image


class InterventionBase(ABC):
    """Abstract base for VLM interventions used at inference time."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in result filenames (e.g. 'no_intervention')."""

    @property
    def config(self) -> dict:
        """Hyperparameters to log alongside results. Override in subclasses."""
        return {}

    @abstractmethod
    def generate(
        self,
        wrapper,
        image: Optional[Image.Image],
        question: str,
        max_new_tokens: int = 256,
        caption: Optional[str] = None,
    ) -> str:
        """Generate a response under this intervention. Return the raw string.

        Args:
            caption: image caption for the TT counterpart. Required by
                CompSafetyShift to compute the modality-induced shift;
                ignored by other interventions.
        """
