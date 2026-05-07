"""Inference-time defense interventions and registry."""

from .base import InterventionBase
from .vanilla import VanillaIntervention
from .comp_safety_shift import CompSafetyShiftIntervention
from .adashield_s import AdaShieldSIntervention


def get_intervention(name: str, model_id: str, **kwargs) -> InterventionBase:
    """Factory: return an intervention instance by short name."""
    registry = {
        "vanilla":           lambda: VanillaIntervention(),
        "comp_safety_shift": lambda: CompSafetyShiftIntervention(model_id=model_id, **kwargs),
        "adashield_s":       lambda: AdaShieldSIntervention(),
    }
    if name not in registry:
        raise ValueError(
            f"Unknown intervention '{name}'. Available: {sorted(registry)}"
        )
    return registry[name]()


ALL_INTERVENTIONS = ["vanilla", "comp_safety_shift", "adashield_s"]

__all__ = [
    "InterventionBase",
    "VanillaIntervention",
    "CompSafetyShiftIntervention",
    "AdaShieldSIntervention",
    "get_intervention",
    "ALL_INTERVENTIONS",
]
