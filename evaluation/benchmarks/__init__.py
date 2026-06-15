"""Benchmark loaders returning EvalSample instances."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from PIL import Image


@dataclass
class EvalSample:
    id: str
    question: str
    image: Optional[Image.Image]
    benchmark: str
    task: Optional[str] = None
    ground_truth: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


def _sample_from_dict(s: dict) -> EvalSample:
    return EvalSample(
        id=str(s["id"]),
        question=s["text"],
        image=s.get("image_pil"),
        benchmark=s["benchmark"],
        task=s.get("task"),
        ground_truth=s.get("label"),
        metadata={
            "category": s.get("category"),
            "label_idx": s.get("label_idx"),
            "image_path": s.get("image_path"),
            "raw": s.get("raw", {}),
        },
    )


def load_pope_eval(limit: Optional[int] = None, **kwargs) -> list:
    from src.dataset import load_pope
    return [_sample_from_dict(s) for s in load_pope(limit=limit, **kwargs)]


def load_amber_eval(limit: Optional[int] = None, **kwargs) -> list:
    from src.dataset import load_amber
    return [_sample_from_dict(s) for s in load_amber(limit=limit, **kwargs)]


def load_chair_eval(limit: Optional[int] = None, **kwargs) -> list:
    from src.dataset import load_chair
    return [_sample_from_dict(s) for s in load_chair(limit=limit, **kwargs)]


def load_hallusionbench_eval(limit: Optional[int] = None, **kwargs) -> list:
    from src.dataset import load_hallusionbench
    return [_sample_from_dict(s) for s in load_hallusionbench(limit=limit, **kwargs)]


def load_mmhal_bench_eval(limit: Optional[int] = None, **kwargs) -> list:
    from src.dataset import load_mmhal_bench
    return [_sample_from_dict(s) for s in load_mmhal_bench(limit=limit, **kwargs)]


__all__ = [
    "EvalSample",
    "load_pope_eval",
    "load_amber_eval",
    "load_chair_eval",
    "load_hallusionbench_eval",
    "load_mmhal_bench_eval",
]
