"""CPU-only regression tests for the v2.1 seeded allocator."""
from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "data_scripts")]
from vti_demos_v2.stage1b_allocate import allocate

def _row(i: int) -> dict:
    return {"id": str(i), "verified_counting_options": [{"category": "cup", "count": 3, "count_complete": True}],
            "verified_relation_options": [{"a": "chair", "b": "table", "type": "horizontal", "a_side": "left"}],
            "verified_distractor_options": [{"category": "fork", "score": .5}],
            "verified_attribute_options": [{"type": "color", "object": "chair", "true_value": "red", "false_value": "blue"}]}

def test_allocator_is_seeded_and_topup_immutable():
    rows = [_row(i) for i in range(4)]
    first = allocate(rows)
    assert allocate(rows) == first
    topup = allocate(rows + [_row(9)], existing=first)
    assert topup[:len(first)] == first
    assert topup[-1]["id"] == "9"
