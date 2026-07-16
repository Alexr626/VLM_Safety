"""Geometry predicates for demos_v2.1 relation options (COCO boxes).

Image coords: x increases to viewer-right, y increases downward.
Boxes are ``(x1, y1, x2, y2)`` in absolute pixels.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


def box_from_coco_bbox(bbox: Sequence[float]) -> Box:
    x, y, w, h = bbox
    return (float(x), float(y), float(x + w), float(y + h))


def box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_contains(outer: Box, inner: Box, tol: float = 1.0) -> bool:
    """True if inner is contained in outer (small pixel tolerance)."""
    return (
        inner[0] >= outer[0] - tol
        and inner[1] >= outer[1] - tol
        and inner[2] <= outer[2] + tol
        and inner[3] <= outer[3] + tol
    )


def boxes_disjoint(a: Box, b: Box) -> bool:
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def x_overlap_frac(a: Box, b: Box) -> float:
    """Overlap length / min(widths). 0 if no overlap."""
    wa = max(0.0, a[2] - a[0])
    wb = max(0.0, b[2] - b[0])
    if wa <= 0 or wb <= 0:
        return 0.0
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return overlap / min(wa, wb)


def edge_gap_x(a: Box, b: Box) -> float:
    """Positive if a is fully left of b (edge-to-edge)."""
    return b[0] - a[2]


def edge_gap_y(a: Box, b: Box) -> float:
    """Positive if a is fully above b (smaller y = higher in image)."""
    return b[1] - a[3]


def min_box_distance(a: Box, b: Box) -> float:
    """Euclidean distance between closest points of two axis-aligned boxes."""
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return math.hypot(dx, dy)


def evaluate_horizontal(
    a: Box,
    b: Box,
    *,
    img_w: float,
    img_area: float,
    gap_frac: float,
    min_area_frac: float,
) -> Optional[dict]:
    if box_area(a) / img_area < min_area_frac or box_area(b) / img_area < min_area_frac:
        return None
    if box_contains(a, b) or box_contains(b, a):
        return None
    gap_min = gap_frac * img_w
    # A left of B
    gap = edge_gap_x(a, b)
    if gap >= gap_min:
        return {
            "type": "horizontal",
            "a_side": "left",
            "gap_frac": round(gap / img_w, 6),
            "geometry": {"gap_px": round(gap, 3)},
        }
    # A right of B ⇔ B left of A
    gap = edge_gap_x(b, a)
    if gap >= gap_min:
        return {
            "type": "horizontal",
            "a_side": "right",
            "gap_frac": round(gap / img_w, 6),
            "geometry": {"gap_px": round(gap, 3)},
        }
    return None


def evaluate_vertical(
    a: Box,
    b: Box,
    *,
    img_h: float,
    img_area: float,
    v_gap_frac: float,
    x_overlap_min: float,
    min_area_frac: float,
) -> Optional[dict]:
    if box_area(a) / img_area < min_area_frac or box_area(b) / img_area < min_area_frac:
        return None
    if box_contains(a, b) or box_contains(b, a):
        return None
    if x_overlap_frac(a, b) < x_overlap_min:
        return None
    gap_min = v_gap_frac * img_h
    # A above B (A has smaller y2 than B's y1)
    gap = edge_gap_y(a, b)
    if gap >= gap_min:
        return {
            "type": "vertical",
            "a_side": "above",
            "gap_frac": round(gap / img_h, 6),
            "geometry": {"gap_px": round(gap, 3), "x_overlap": round(x_overlap_frac(a, b), 4)},
        }
    gap = edge_gap_y(b, a)
    if gap >= gap_min:
        return {
            "type": "vertical",
            "a_side": "below",
            "gap_frac": round(gap / img_h, 6),
            "geometry": {"gap_px": round(gap, 3), "x_overlap": round(x_overlap_frac(a, b), 4)},
        }
    return None


def evaluate_support(
    a: Box,
    b: Box,
    *,
    b_category: str,
    support_surfaces: Sequence[str],
    max_area_ratio: float,
    img_area: float,
    min_area_frac: float,
) -> Optional[dict]:
    if b_category not in support_surfaces:
        return None
    if box_area(a) / img_area < min_area_frac or box_area(b) / img_area < min_area_frac:
        return None
    if not box_contains(b, a):
        return None
    aa, ab = box_area(a), box_area(b)
    if ab <= 0 or aa / ab > max_area_ratio:
        return None
    # A in upper half of B
    mid_y = 0.5 * (b[1] + b[3])
    a_cy = 0.5 * (a[1] + a[3])
    if a_cy > mid_y:
        return None
    return {
        "type": "support",
        "a_side": "on_top_of",
        "gap_frac": 0.0,
        "geometry": {
            "area_ratio": round(aa / ab, 6),
            "a_center_y_frac_in_b": round((a_cy - b[1]) / max(b[3] - b[1], 1e-6), 4),
        },
    }


def evaluate_proximity(
    a: Box,
    b: Box,
    *,
    img_w: float,
    img_h: float,
    near_max_frac: float,
    far_min_frac: float,
    img_area: float,
    min_area_frac: float,
) -> Optional[dict]:
    if box_area(a) / img_area < min_area_frac or box_area(b) / img_area < min_area_frac:
        return None
    if box_contains(a, b) or box_contains(b, a):
        return None
    diag = math.hypot(img_w, img_h)
    if diag <= 0:
        return None
    d = min_box_distance(a, b) / diag
    if d <= near_max_frac:
        return {
            "type": "proximity",
            "a_side": "near",
            "gap_frac": round(d, 6),
            "geometry": {"dist_frac": round(d, 6)},
        }
    if d >= far_min_frac:
        return {
            "type": "proximity",
            "a_side": "far",
            "gap_frac": round(d, 6),
            "geometry": {"dist_frac": round(d, 6)},
        }
    return None


def evaluate_all_relation_types(
    a: Box,
    b: Box,
    *,
    b_category: str,
    img_w: float,
    img_h: float,
    img_area: float,
    cfg,
) -> List[dict]:
    """Return zero or more relation option geometry dicts for ordered pair (A,B)."""
    out: List[dict] = []
    h = evaluate_horizontal(
        a, b,
        img_w=img_w, img_area=img_area,
        gap_frac=cfg.REL_GAP_FRAC, min_area_frac=cfg.REL_MIN_AREA_FRAC,
    )
    if h:
        out.append(h)
    v = evaluate_vertical(
        a, b,
        img_h=img_h, img_area=img_area,
        v_gap_frac=cfg.V_GAP_FRAC, x_overlap_min=cfg.X_OVERLAP_MIN,
        min_area_frac=cfg.REL_MIN_AREA_FRAC,
    )
    if v:
        out.append(v)
    s = evaluate_support(
        a, b,
        b_category=b_category,
        support_surfaces=cfg.SUPPORT_SURFACES,
        max_area_ratio=cfg.SUPPORT_MAX_AREA_RATIO,
        img_area=img_area, min_area_frac=cfg.REL_MIN_AREA_FRAC,
    )
    if s:
        out.append(s)
    p = evaluate_proximity(
        a, b,
        img_w=img_w, img_h=img_h,
        near_max_frac=cfg.NEAR_MAX_FRAC, far_min_frac=cfg.FAR_MIN_FRAC,
        img_area=img_area, min_area_frac=cfg.REL_MIN_AREA_FRAC,
    )
    if p:
        out.append(p)
    return out
