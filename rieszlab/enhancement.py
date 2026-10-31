"""Continuous coherency and direction-selective enhancement.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
Enhanced strain is for display or downstream use, not independent Riesz validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import fft

from ._validation import positive
from .riesz import MonogenicResult, RieszResult, _symbols


@dataclass(frozen=True)
class DirectionSelectionConfig:
    """Soft axial-angle selection in the SAME metric as the Riesz transform."""

    slowness_s_per_m: float
    angular_sigma_degrees: float = 10.0
    coherence_power: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.slowness_s_per_m):
            raise ValueError("slowness_s_per_m must be finite")
        positive(self.angular_sigma_degrees, "angular_sigma_degrees")
        positive(self.coherence_power, "coherence_power")


@dataclass
class EnhancementResult:
    data: np.ndarray
    weights: np.ndarray
    direction_weight: np.ndarray | None = None
    selection: DirectionSelectionConfig | None = None


def _synthesise(mono: MonogenicResult, weight: np.ndarray) -> EnhancementResult:
    domain = mono._domain
    crop = domain.crop
    ms, mt = _symbols(
        domain.even.shape,
        mono.dx_m,
        mono.dt_s,
        mono.config.reference_velocity_mps,
        domain.even.dtype,
    )
    total = fft.rfft2(weight * domain.even, workers=1)
    for component, symbol in ((domain.riesz_space, ms), (domain.riesz_time, mt)):
        total += 1j * symbol * fft.rfft2(weight * component, workers=1)
    total /= 1 + ms**2 + mt**2
    data = fft.irfft2(total, s=domain.even.shape, workers=1)[crop].copy()
    return EnhancementResult(data, weight[crop])


def reconstruct(mono: MonogenicResult, weights: ArrayLike) -> EnhancementResult:
    """Weight all three components and reconstruct their least-squares scalar field.

    Uses the cached padded coefficients and canonical inverse, including DC
    and Nyquist bins. Weights must be finite, in [0,1], and broadcastable to
    the input grid; they are extended at the boundary using the padding mode
    (edge extension for zero-padded data). Constant weights scale the input
    exactly. Returns the enhanced field in .data and applied weights in .weights.
    """
    g = np.asarray(weights, dtype=mono.even.dtype)
    g = np.broadcast_to(g, mono.even.shape)
    if not np.isfinite(g).all() or np.any((g < 0) | (g > 1)):
        raise ValueError("weights must be finite, in [0,1], and broadcast to the data")
    ps, pt = mono.config.pad_samples
    mode = "edge" if mono.config.pad_mode == "constant" else mono.config.pad_mode
    return _synthesise(mono, np.pad(g, ((ps, ps), (pt, pt)), mode=mode))


def enhance_coherent(result: RieszResult, *, power: float = 1.0) -> EnhancementResult:
    """Continuous g=X**power enhancement, without a gate or gain normalisation.

    Coherency is evaluated on the padded analysis domain before reconstruction.
    Re-estimating Riesz attributes afterwards is circular, not an independent
    check of slowness or physical mode separation.
    """
    positive(power, "power")
    if result._tensor_domain is None:
        raise ValueError("Coherent enhancement requires a tensor; supply tensor=...")
    weight = result._tensor_domain.coherence_regularized**power
    return _synthesise(result.monogenic, weight)


def direction_weight(
    orientation: ArrayLike, reference_velocity_mps: float, *, selection: DirectionSelectionConfig
) -> np.ndarray:
    """Gaussian agreement with a prescribed normal; undefined directions get zero."""
    positive(reference_velocity_mps, "reference_velocity_mps")
    target = np.arctan2(1.0, -reference_velocity_mps * selection.slowness_s_per_m)
    delta = np.asarray(orientation) - target
    delta = 0.5 * np.arctan2(np.sin(2 * delta), np.cos(2 * delta))
    weight = np.exp(-0.5 * (delta / np.deg2rad(selection.angular_sigma_degrees)) ** 2)
    return np.where(np.isfinite(weight), weight, 0)


def enhance_directional(
    result: RieszResult,
    *,
    selection: DirectionSelectionConfig,
) -> EnhancementResult:
    """Reconstruct with g=X**power times soft agreement with a selected direction.

    Works with either an ordinary or steered tensor. Opposite-direction
    outputs are conditional selections, not complementary physical modes.
    """
    if result._tensor_domain is None:
        raise ValueError("Direction-selective enhancement requires a tensor; supply tensor=...")
    mono, tensor = result.monogenic, result._tensor_domain
    angular = direction_weight(
        tensor.orientation, mono.config.reference_velocity_mps, selection=selection
    )
    weight = tensor.coherence_regularized**selection.coherence_power * angular
    enhanced = _synthesise(mono, weight)
    enhanced.direction_weight = angular[mono._domain.crop]
    enhanced.selection = selection
    return enhanced
