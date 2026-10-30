"""Surveyed cable tangents and geometry-only adaptive tensor widths.

Copyright (c) 2026 Aleix Seguí Ugalde.
Affiliation: University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter1d

from ._validation import positive as _positive


@dataclass(frozen=True)
class GeometryConfig:
    smoothing_fwhm_m: float = 20.0
    max_sigma_m: float = 100.0
    max_turn_degrees: float = 20.0
    footprint_sigmas: float = 3.0
    boundary_sigmas: float = 4.0

    def __post_init__(self) -> None:
        _positive(self.smoothing_fwhm_m, "smoothing_fwhm_m", allow_zero=True)
        for name in ("max_sigma_m", "max_turn_degrees", "footprint_sigmas", "boundary_sigmas"):
            _positive(getattr(self, name), name)
        if self.boundary_sigmas < self.footprint_sigmas:
            raise ValueError("boundary_sigmas must be >= footprint_sigmas")


@dataclass
class CableGeometry:
    distance_m: np.ndarray
    position_enu_m: np.ndarray
    tangent_enu: np.ndarray
    smoothing_fwhm_m: float

    @property
    def azimuth_degrees(self) -> np.ndarray:
        """Clockwise from north, looking towards increasing channel distance."""
        e, n = self.tangent_enu[:, 0], self.tangent_enu[:, 1]
        return np.where(np.hypot(e, n) > 0, np.degrees(np.arctan2(e, n)) % 360, np.nan)


@dataclass
class AdaptiveWidths:
    sigma_m: np.ndarray
    max_turn_degrees: np.ndarray
    cumulative_turn_rad: np.ndarray
    boundary_cap_m: np.ndarray
    subchannel: np.ndarray
    config: GeometryConfig


def _distance(distance_m: ArrayLike) -> np.ndarray:
    s = np.asarray(distance_m, float)
    if s.ndim != 1 or len(s) < 3 or not np.isfinite(s).all() or np.any(np.diff(s) <= 0):
        raise ValueError("distance_m must be finite, increasing, and contain >=3 points")
    return s


def cable_geometry(
    distance_m: ArrayLike, position_enu_m: ArrayLike, *, smoothing_fwhm_m: float = 20.0
) -> CableGeometry:
    """Construct geometry on a uniform, correctly registered channel grid.

    Positions are local east/north/up in metres. Two columns imply up=0.
    Geographic projection and channel-to-survey registration must be done
    explicitly upstream. No offset is guessed and no route is extrapolated.
    Smoothing is applied to positions, then derivatives are normalised to
    unit 3-D tangents. It is distinct from tensor averaging.
    """
    s = _distance(distance_m)
    if not np.allclose(np.diff(s), np.diff(s)[0], rtol=1e-6, atol=1e-9):
        raise ValueError("Resample surveyed positions to a uniform channel-distance grid first")
    pos = np.array(position_enu_m, dtype=float, copy=True)
    if pos.shape not in ((len(s), 2), (len(s), 3)) or not np.isfinite(pos).all():
        raise ValueError("Expected one finite east/north[/up] position per channel")
    if pos.shape[1] == 2:
        pos = np.column_stack((pos, np.zeros(len(s))))
    _positive(smoothing_fwhm_m, "smoothing_fwhm_m", allow_zero=True)
    if smoothing_fwhm_m:
        sigma = smoothing_fwhm_m / np.sqrt(8 * np.log(2)) / (s[1] - s[0])
        pos = gaussian_filter1d(pos, sigma, axis=0, mode="nearest")
    tangent = np.gradient(pos, s, axis=0, edge_order=2)
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm == 0):
        raise ValueError("Repeated or unresolved positions give zero-length tangents")
    return CableGeometry(s.copy(), pos, tangent / norm[:, None], smoothing_fwhm_m)


def adaptive_tensor_widths(
    cable: CableGeometry, *, config: GeometryConfig | None = None
) -> AdaptiveWidths:
    """Largest sigma with cumulative 3-D turning <= tolerance within ±h*sigma.

    Sum unsigned angles between adjacent unit tangents: successive opposite
    bends cannot cancel. Restrict the full truncated tensor footprint to known
    geometry with boundary_sigmas (set to the tensor's truncate). There is no
    forced lower width: sharp bends and route edges may require zero/subchannel
    smoothing. This constrains tensor mixing, not Riesz nonlocality or p bias.
    """
    cfg = config or GeometryConfig(smoothing_fwhm_m=cable.smoothing_fwhm_m)
    if not np.isclose(cfg.smoothing_fwhm_m, cable.smoothing_fwhm_m):
        raise ValueError("GeometryConfig smoothing must match the supplied CableGeometry")
    s = _distance(cable.distance_m)
    v = np.asarray(cable.tangent_enu, float)
    if v.shape != (len(s), 3) or not np.isfinite(v).all():
        raise ValueError("Expected finite nonzero 3-D tangents")
    norms = np.linalg.norm(v, axis=1)
    if np.any(norms == 0):
        raise ValueError("Expected finite nonzero 3-D tangents")
    v = v / norms[:, None]
    angles = 2 * np.arctan2(
        np.linalg.norm(np.diff(v, axis=0), axis=1), np.linalg.norm(v[1:] + v[:-1], axis=1)
    )
    cumulative = np.r_[0.0, np.cumsum(angles)]
    cap = np.minimum(cfg.max_sigma_m, np.minimum(s - s[0], s[-1] - s) / cfg.boundary_sigmas)

    def turn(sigma: np.ndarray) -> np.ndarray:
        radius = cfg.footprint_sigmas * sigma
        return np.maximum(
            np.interp(s + radius, s, cumulative) - cumulative,
            cumulative - np.interp(s - radius, s, cumulative),
        )

    lower, upper = np.zeros_like(s), cap.copy()
    tolerance = np.deg2rad(cfg.max_turn_degrees)
    for _ in range(48):
        mid = (lower + upper) / 2
        admissible = turn(mid) <= tolerance
        lower, upper = np.where(admissible, mid, lower), np.where(admissible, upper, mid)
    sigma = np.where(turn(cap) <= tolerance, cap, lower)
    return AdaptiveWidths(
        sigma, np.degrees(turn(sigma)), cumulative, cap, sigma < np.min(np.diff(s)), cfg
    )
