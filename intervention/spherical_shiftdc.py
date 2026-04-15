"""
Spherical ShiftDC — Core Math
==============================
Implements two activation calibration strategies:

  1. **Original ShiftDC** (Zou et al. 2025)
       x_hat = x_vl - proj_{s^l}(m^l)
     Subtracts the safety-relevant component of the modality shift.
     Changes the activation norm (||x_hat|| ≠ ||x_vl||), which can
     interfere with RMSNorm layers downstream.

  2. **Spherical ShiftDC** (proposed)
       Same target direction as original ShiftDC, but uses Slerp to
       *rotate* x_vl toward that target rather than subtracting from it.
       Strictly preserves the activation norm: ||x_hat|| = ||x_vl||.
       Motivation: LLaVA's LLaMA backbone uses RMSNorm, which normalises
       each layer's output; norm-preserving interventions are less likely
       to degrade generation quality.

Both functions operate on float64 numpy arrays (single sample, single layer).

Usage
-----
    from intervention.spherical_shiftdc import (
        compute_shiftdc_calibration,
        compute_spherical_shiftdc,
        slerp,
    )

    s_l   = safety_vecs["layer_31"]              # (4096,) float32
    x_vl  = cache.load(sample_id, "vl")[31]      # (4096,) float32
    x_tt  = cache.load(sample_id, "tt")[31]      # (4096,) float32

    x_hat_orig = compute_shiftdc_calibration(x_vl, x_tt, s_l)
    x_hat_sph  = compute_spherical_shiftdc(x_vl, x_tt, s_l, t=1.0)

    # Norm check
    assert abs(np.linalg.norm(x_hat_sph) / np.linalg.norm(x_vl) - 1.0) < 1e-5
"""

import numpy as np


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _project(v: np.ndarray, onto: np.ndarray) -> np.ndarray:
    """Scalar projection of v onto `onto`, returned as a vector.

    proj_{onto}(v) = (v · onto / ||onto||^2) * onto
    """
    s2 = float(np.dot(onto, onto))
    if s2 < 1e-12:
        return np.zeros_like(v)
    return (np.dot(v, onto) / s2) * onto


def slerp(v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit vectors.

    Interpolates along the great circle from v1 (t=0) to v2 (t=1).
    Inputs are normalised internally; output is a unit vector.

    Falls back to linear interpolation (+ renormalisation) when the
    angle between v1 and v2 is very small (< 1e-6 rad).

    Args:
        v1: Start unit vector, shape (d,).
        v2: End unit vector, shape (d,).
        t:  Interpolation parameter in [0, 1].

    Returns:
        Unit vector of shape (d,).
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    theta = np.arccos(dot)
    if np.abs(theta) < 1e-6:
        # Vectors nearly parallel — linear interpolation is numerically stable
        out = (1.0 - t) * v1 + t * v2
        norm = np.linalg.norm(out)
        return out / norm if norm > 1e-12 else v1
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - t) * theta) / sin_theta) * v1 \
         + (np.sin(t * theta)         / sin_theta) * v2


# ── Public calibration functions ──────────────────────────────────────────────

def compute_shiftdc_calibration(
    x_vl: np.ndarray,
    x_tt: np.ndarray,
    s_l: np.ndarray,
) -> np.ndarray:
    """Original ShiftDC calibration (Zou et al. 2025).

    Subtracts the safety-relevant component of the modality shift:
        m^l     = x_vl - x_tt
        x_hat   = x_vl - proj_{s^l}(m^l)

    NOTE: The output norm ||x_hat|| is generally ≠ ||x_vl||.

    Args:
        x_vl: VL activation at a given layer, shape (d,).
        x_tt: TT (text-only) activation at a given layer, shape (d,).
        s_l:  Safety direction at that layer, shape (d,).

    Returns:
        Calibrated activation, shape (d,), float64.
    """
    x_vl = x_vl.astype(np.float64)
    x_tt = x_tt.astype(np.float64)
    s_l  = s_l.astype(np.float64)
    m = x_vl - x_tt
    return x_vl - _project(m, s_l)


def compute_spherical_shiftdc(
    x_vl: np.ndarray,
    x_tt: np.ndarray,
    s_l: np.ndarray,
    t: float = 1.0,
    gate_by_alignment: bool = False,
    gate_threshold: float = 0.0,
) -> np.ndarray:
    """Spherical ShiftDC calibration (proposed).

    Rotates x_vl toward the ShiftDC target direction, strictly preserving
    the activation norm:
        target   = x_vl - proj_{s^l}(m^l)           # ShiftDC direction
        d_target = target / ||target||               # normalise
        x_hat    = ||x_vl|| * slerp(x_vl/||x_vl||, d_target, t)

    The interpolation parameter t in [0, 1] controls the rotation strength:
      t = 0.0  → no change (x_hat = x_vl)
      t = 1.0  → full rotation to ShiftDC target direction

    Optional gating: if `gate_by_alignment=True`, t is multiplied by
    max(0, cosine(m^l, s^l) - gate_threshold), so the intervention is
    suppressed when the modality shift is not well aligned with the safety
    direction (i.e., when the intervention would have little effect anyway).

    Args:
        x_vl:              VL activation, shape (d,).
        x_tt:              TT activation, shape (d,).
        s_l:               Safety direction, shape (d,).
        t:                 Rotation strength in [0, 1] (default 1.0 = full).
        gate_by_alignment: If True, scale t by cos(m^l, s^l).
        gate_threshold:    Minimum cosine alignment to activate gating.

    Returns:
        Calibrated activation, shape (d,), float64.
        Norm is preserved: ||x_hat|| == ||x_vl|| (up to float64 precision).
    """
    x_vl = x_vl.astype(np.float64)
    x_tt = x_tt.astype(np.float64)
    s_l  = s_l.astype(np.float64)

    norm_vl = np.linalg.norm(x_vl)
    if norm_vl < 1e-12:
        return x_vl.copy()

    # Compute ShiftDC target
    target = compute_shiftdc_calibration(x_vl, x_tt, s_l)
    norm_target = np.linalg.norm(target)
    if norm_target < 1e-12:
        # Target is zero — nothing to rotate toward
        return x_vl.copy()

    # Optional: gate t by alignment between m^l and s^l
    effective_t = t
    if gate_by_alignment:
        m = x_vl - x_tt
        m_norm = np.linalg.norm(m)
        s_norm = np.linalg.norm(s_l)
        if m_norm > 1e-12 and s_norm > 1e-12:
            cos_align = float(np.dot(m, s_l) / (m_norm * s_norm))
            effective_t = t * max(0.0, cos_align - gate_threshold)
        else:
            effective_t = 0.0

    if effective_t < 1e-9:
        return x_vl.copy()

    # Slerp in direction space, restore original norm
    x_unit      = x_vl     / norm_vl
    d_target    = target    / norm_target
    x_hat_unit  = slerp(x_unit, d_target, effective_t)
    return norm_vl * x_hat_unit


# ── Diagnostics ───────────────────────────────────────────────────────────────

def calibration_stats(
    x_vl: np.ndarray,
    x_hat: np.ndarray,
    s_l: np.ndarray,
) -> dict:
    """Compute diagnostic statistics comparing x_vl and x_hat.

    Returns:
        dict with keys:
          norm_ratio     : ||x_hat|| / ||x_vl||  (1.0 = perfect norm preservation)
          cosine_change  : cos(x_vl, x_hat)      (1.0 = no rotation)
          proj_vl        : dot(x_vl, s^l) / ||s^l||^2  (original projection)
          proj_hat       : dot(x_hat, s^l) / ||s^l||^2 (calibrated projection)
          proj_reduction : (proj_vl - proj_hat) / |proj_vl|  (fractional reduction)
    """
    x_vl  = x_vl.astype(np.float64)
    x_hat = x_hat.astype(np.float64)
    s_l   = s_l.astype(np.float64)

    norm_vl  = float(np.linalg.norm(x_vl))
    norm_hat = float(np.linalg.norm(x_hat))
    norm_s   = float(np.dot(s_l, s_l))

    cos_change = float(np.dot(x_vl, x_hat) / (norm_vl * norm_hat + 1e-12))
    proj_vl    = float(np.dot(x_vl,  s_l) / (norm_s + 1e-12))
    proj_hat   = float(np.dot(x_hat, s_l) / (norm_s + 1e-12))
    proj_red   = (proj_vl - proj_hat) / (abs(proj_vl) + 1e-12)

    return {
        "norm_ratio":     norm_hat / (norm_vl + 1e-12),
        "cosine_change":  cos_change,
        "proj_vl":        proj_vl,
        "proj_hat":       proj_hat,
        "proj_reduction": proj_red,
    }
