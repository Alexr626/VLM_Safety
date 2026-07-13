"""Defaults for the demos_v2 generation pipeline (CLI overrides allowed)."""

from __future__ import annotations

import math
from typing import Callable

N_CANDIDATES = 300
N_FINAL = 150
COUNT_MIN, COUNT_MAX = 2, 9
REL_GAP_FRAC = 0.05
REL_MIN_AREA_FRAC = 0.015
COUNT_MIN_AREA_FRAC = 0.005  # median bbox area gate for counting anchor
DISTRACTOR_TOP_K = 5
SEED = 42
ALLOW_IMAGE_DOWNLOAD = True
TOPUP_BATCH = 100
PIPELINE_VERSION = "demos_v2_2026-07-12"
QUESTION = "Describe this image in detail."

STAGE1_PROVIDER = "anthropic:claude-sonnet-4-6"
STAGE2_PROVIDER = "anthropic:claude-opus-4-8"
STAGE3_PROVIDER = "anthropic:claude-haiku-4-5"
STAGE4_PROVIDER = "anthropic:claude-sonnet-4-6"

BANNED_HEDGES = [
    "possibly", "likely", "appears", "appear", "seems", "seem", "perhaps",
    "might", "may", "probably", "some kind of", "what looks like",
]

FALSE_COUNT: Callable[[int], int] = lambda n: min(max(n + 2, math.ceil(1.5 * n)), 20)

# Score weights for stage-0 ranking
W_REL_GAP = 0.45
W_COUNT_AREA = 0.35
W_N_CATS = 0.20
