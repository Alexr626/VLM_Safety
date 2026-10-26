"""Every module imports without a GPU, model weights, or downloaded data."""

import importlib

import pytest

MODULES = [
    "src.model",
    "src.model_internals",
    "src.hooks",
    "src.dataset",
    "src.extraction",
    "src.paths",
    "src.liseco",
    "evaluation.benchmarks",
    "evaluation.classifiers.metrics",
    "evaluation.classifiers.chair_objects",
    "evaluation.interventions",
    "evaluation.runners.eval_runner",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_create_wrapper_dispatch():
    from src.model import LLaVAWrapper, Qwen2VLWrapper, create_wrapper
    assert isinstance(create_wrapper("llava-hf/llava-1.5-7b-hf", max_pixels=1),
                      LLaVAWrapper)
    assert isinstance(create_wrapper("Qwen/Qwen2.5-VL-7B-Instruct", pad_to_square=True),
                      Qwen2VLWrapper)
    with pytest.raises(ValueError):
        create_wrapper("some/other-model")


def test_expand2square():
    from PIL import Image
    from src.model import expand2square
    out = expand2square(Image.new("RGB", (40, 20), (255, 0, 0)), (0, 0, 0))
    assert out.size == (40, 40)
    assert out.getpixel((0, 0)) == (0, 0, 0)
    assert out.getpixel((20, 20)) == (255, 0, 0)


def test_registered_benchmarks():
    from src.dataset import ALL_BENCHMARKS
    assert set(ALL_BENCHMARKS) == {"pope", "amber", "chair"}
