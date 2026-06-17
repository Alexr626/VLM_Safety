"""Inference-time hallucination mitigation interventions."""

from .base import InterventionBase
from .no_intervention import NoIntervention
from .vti import VTITextualIntervention
from .vti.steer import HOOK_SITES, STEER_VARIANTS


def _make_vti_factory(variant: str, hook_site: str):
    def factory(model_id: str, **kwargs):
        return VTITextualIntervention(
            model_id, variant=variant, hook_site=hook_site, **kwargs,
        )
    return factory


_VTI_REGISTRY = {
    f"vti_textual_{variant}_{hook_site}": _make_vti_factory(variant, hook_site)
    for variant in STEER_VARIANTS
    for hook_site in HOOK_SITES
}


def get_intervention(name: str, model_id: str, **kwargs) -> InterventionBase:
    registry = {
        "no_intervention": lambda model_id, **kw: NoIntervention(),
        **_VTI_REGISTRY,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown intervention '{name}'. Available: {sorted(registry)}"
        )
    return registry[name](model_id=model_id, **kwargs)


ALL_INTERVENTIONS = ["no_intervention"] + sorted(_VTI_REGISTRY.keys())

__all__ = [
    "InterventionBase",
    "NoIntervention",
    "VTITextualIntervention",
    "get_intervention",
    "ALL_INTERVENTIONS",
]
