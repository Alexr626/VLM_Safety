"""Inference-time hallucination mitigation interventions."""

from .base import InterventionBase
from .no_intervention import NoIntervention


def get_intervention(name: str, model_id: str, **kwargs) -> InterventionBase:
    registry = {
        "no_intervention": lambda: NoIntervention(),
    }
    if name not in registry:
        raise ValueError(
            f"Unknown intervention '{name}'. Available: {sorted(registry)}"
        )
    return registry[name]()


ALL_INTERVENTIONS = ["no_intervention"]

__all__ = [
    "InterventionBase",
    "NoIntervention",
    "get_intervention",
    "ALL_INTERVENTIONS",
]
