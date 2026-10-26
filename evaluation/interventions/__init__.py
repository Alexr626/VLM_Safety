"""Inference-time interventions evaluated by `evaluation/run_eval.py`.

To add one: subclass `InterventionBase` in a new module and register a
factory in `_REGISTRY` below.
"""

from .base import InterventionBase
from .no_intervention import NoIntervention


_REGISTRY = {
    "no_intervention": lambda model_id, **kw: NoIntervention(),
}


def get_intervention(name: str, model_id: str, **kwargs) -> InterventionBase:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown intervention '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](model_id=model_id, **kwargs)


ALL_INTERVENTIONS = sorted(_REGISTRY)

__all__ = [
    "InterventionBase",
    "NoIntervention",
    "get_intervention",
    "ALL_INTERVENTIONS",
]
