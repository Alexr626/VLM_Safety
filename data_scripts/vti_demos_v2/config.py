"""Defaults for the demos_v2.1 generation pipeline (CLI overrides allowed)."""

from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

N_CANDIDATES = 300
N_FINAL = 150
COUNT_MIN, COUNT_MAX = 2, 9
REL_MIN_AREA_FRAC = 0.015
COUNT_MIN_AREA_FRAC = 0.005  # median bbox area gate for counting options
DISTRACTOR_TOP_K = 5
SEED = 42
ALLOW_IMAGE_DOWNLOAD = True
TOPUP_BATCH = 100
PIPELINE_VERSION = "demos_v2.1_2026-07-13"
QUESTION = "Describe this image in detail."

STAGE1_PROVIDER = "anthropic:claude-sonnet-5"
STAGE2_PROVIDER = "anthropic:claude-opus-4-8"
STAGE3_PROVIDER = "anthropic:claude-haiku-4-5"
STAGE4_PROVIDER = "anthropic:claude-sonnet-5"

BANNED_HEDGES = [
    "possibly", "likely", "appears", "appear", "seems", "seem", "perhaps",
    "might", "may", "probably", "some kind of", "what looks like",
]

FALSE_COUNT: Callable[[int], int] = lambda n: min(max(n + 2, math.ceil(1.5 * n)), 20)

# Score weights for stage-0 ranking
W_REL_GAP = 0.30
W_COUNT_AREA = 0.25
W_N_CATS = 0.15
W_N_GE3 = 0.15          # prefer images with a counting option N >= 3
W_REL_TYPE_DIV = 0.15   # prefer images with more distinct relation types

# Relation types and geometry
REL_TYPES: Tuple[str, ...] = ("horizontal", "vertical", "support", "proximity")
REL_GAP_FRAC = 0.05            # edge-to-edge, per chosen instance pair
V_GAP_FRAC = 0.05
X_OVERLAP_MIN = 0.30
SUPPORT_SURFACES = ("dining table", "bed", "couch", "bench", "chair")
SUPPORT_MAX_AREA_RATIO = 0.25
NEAR_MAX_FRAC = 0.02
FAR_MIN_FRAC = 0.40
REL_TYPE_TARGET: Dict[str, float] = {t: 0.25 for t in REL_TYPES}

# Attribute types
ATTR_TYPES: Tuple[str, ...] = ("color", "material", "state", "action", "texture")
ATTR_TYPE_TARGET: Dict[str, float] = {
    "color": 0.35, "material": 0.20, "state": 0.20, "action": 0.15, "texture": 0.10,
}
ATTR_VALUE_CAP_FRAC = 0.08
K_ATTR = 3

# Existence distractors
DISTRACTOR_TAU = 0.5
DISTRACTOR_CAP_FRAC = 0.04

# Counting
COUNT_MODE_AT_MOST_FRAC = 0.5
AT_MOST_MIN_N = 3
N_GE3_TARGET_FRAC = 0.40
PERSON_COUNT_CAP_FRAC = 0.15

# Scene stratification (stage 0)
SUPERCAT_CAP_FRAC = 0.20

ALLOC_SEED = 42
