"""Benchmark loaders. Each loader returns a list of EvalSample instances."""

from dataclasses import dataclass
from typing import Optional
from PIL import Image


@dataclass
class EvalSample:
    id: str
    question: str
    image: Optional[Image.Image]
    benchmark: str
    scenario_id: Optional[int] = None
    scenario_name: Optional[str] = None
    image_type: Optional[str] = None
    safety_label: Optional[str] = None


from .mm_safetybench import load_mm_safetybench  # noqa: E402
from .figstep import load_figstep  # noqa: E402
from .mssbench import load_mssbench  # noqa: E402

__all__ = ["EvalSample", "load_mm_safetybench", "load_figstep", "load_mssbench"]
