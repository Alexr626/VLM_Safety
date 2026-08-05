"""Inference-time hallucination mitigation interventions."""

from .base import InterventionBase
from .no_intervention import NoIntervention
from .vti import VTITextualIntervention, VTIVisualIntervention
from .vti.steer import HOOK_SITES, STEER_VARIANTS

VISUAL_STEER_VARIANTS = ("additive", "uniform_rotation")


def _make_textual_factory(variant: str, hook_site: str):
    def factory(model_id: str, **kwargs):
        kw = {k: v for k, v in kwargs.items() if k != "alpha"}
        return VTITextualIntervention(
            model_id, variant=variant, hook_site=hook_site, **kw,
        )
    return factory


def _make_visual_factory(variant: str, hook_site: str):
    def factory(model_id: str, **kwargs):
        drop = {"beta", "directions_dir", "layer_indices", "layer_set_label"}
        kw = {k: v for k, v in kwargs.items() if k not in drop}
        return VTIVisualIntervention(
            model_id, variant=variant, hook_site=hook_site, **kw,
        )
    return factory


_VTI_TEXTUAL_REGISTRY = {
    f"vti_textual_{variant}_{hook_site}": _make_textual_factory(variant, hook_site)
    for variant in STEER_VARIANTS
    for hook_site in HOOK_SITES
}

_VTI_VISUAL_REGISTRY = {
    f"vti_visual_{variant}_{hook_site}": _make_visual_factory(variant, hook_site)
    for variant in VISUAL_STEER_VARIANTS
    for hook_site in HOOK_SITES
}


def get_intervention(name: str, model_id: str, **kwargs) -> InterventionBase:
    registry = {
        "no_intervention": lambda model_id, **kw: NoIntervention(),
        **_VTI_TEXTUAL_REGISTRY,
        **_VTI_VISUAL_REGISTRY,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown intervention '{name}'. Available: {sorted(registry)}"
        )
    return registry[name](model_id=model_id, **kwargs)


ALL_INTERVENTIONS = (
    ["no_intervention"]
    + sorted(_VTI_TEXTUAL_REGISTRY.keys())
    + sorted(_VTI_VISUAL_REGISTRY.keys())
)

__all__ = [
    "InterventionBase",
    "NoIntervention",
    "VTITextualIntervention",
    "VTIVisualIntervention",
    "get_intervention",
    "ALL_INTERVENTIONS",
    "VISUAL_STEER_VARIANTS",
    "HOOK_SITES",
    "STEER_VARIANTS",
]
