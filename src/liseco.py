"""
Import shim for the LiSeCo reference code (Cheng & Amo Alonso).

The authors' repo is cloned (not vendored) into `llm-control/` at the project
root:

    git clone https://github.com/chengemily1/llm-control.git

Its modules use bare sibling imports (`from data_util import encode_data`),
so `llm-control/src` has to be on sys.path before importing them.

Usage
-----
    from src.liseco import load_liseco_module
    control_wrapper = load_liseco_module("control_wrapper")
    LiSeCoWrapper = control_wrapper.LiSeCoWrapper
"""

import importlib
import sys
from types import ModuleType

from src.paths import llm_control_root

_CLONE_URL = "https://github.com/chengemily1/llm-control.git"


def ensure_llm_control_on_path() -> str:
    src_dir = llm_control_root() / "src"
    if not (src_dir / "control_wrapper.py").exists():
        raise FileNotFoundError(
            f"LiSeCo code not found at {src_dir}. Clone it into the project root:\n"
            f"  git clone {_CLONE_URL} {llm_control_root()}"
        )
    path = str(src_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def load_liseco_module(name: str = "control_wrapper") -> ModuleType:
    """Import a module from `llm-control/src` (e.g. control_wrapper, data_util).

    Scripts such as train_probe.py and main.py parse argv at import time; run
    those as scripts instead of importing them.
    """
    ensure_llm_control_on_path()
    return importlib.import_module(name)
