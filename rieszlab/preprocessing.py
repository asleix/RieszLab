"""Explicit, amplitude-preserving conditioning before monogenic analysis.

Copyright (c) 2026 Aleix Seguí Ugalde.
Affiliation: University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
Common-mode selection/subtraction follows Sergi Ventosa's
rm_common_mode_noise2.c (2024); no channel normalisation is performed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import fft, signal

from ._validation import matrix as _matrix, positive as _positive


@dataclass
class CommonModeResult:
    data: np.ndarray
    template: np.ndarray
    gains: np.ndarray
    selected_channels: np.ndarray
    channel_energy: np.ndarray
    rejection_threshold: float


def detrend(data: ArrayLike, *, kind: str = "linear") -> np.ndarray:
    """Remove a per-channel linear trend or constant; never divide amplitudes."""
    if kind not in ("linear", "constant"):
        raise ValueError("kind must be linear or constant")
    return signal.detrend(_matrix(data), axis=1, type=kind)


def butterworth(
    data: ArrayLike,
    dt_s: float,
    *,
    band_hz: tuple[float, float] = (3.0, 30.0),
    order: int = 4,
    zero_phase: bool = True,
    demean: bool = True,
) -> np.ndarray:
    """Temporal SOS bandpass in input units.

    zero_phase=True uses forward–backward filtering (two-sided footprint,
    squared magnitude response). False uses a causal single pass with zero
    initial state and a startup transient. Neither makes the Riesz transform
    causal. Filtering and tensor averaging have separate footprints.
    """
    x = _matrix(data)
    _positive(dt_s, "dt_s")
    if len(band_hz) != 2 or not 0 < band_hz[0] < band_hz[1] < 0.5 / dt_s:
        raise ValueError("Require 0 < low < high < Nyquist")
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    sos = signal.butter(order, band_hz, fs=1 / dt_s, btype="bandpass", output="sos")
    # Double-precision filtering; return the input floating-point precision.
    working = x.astype(np.float64)
    working = working - working.mean(axis=1, keepdims=True) if demean else working
    filtered = (signal.sosfiltfilt if zero_phase else signal.sosfilt)(sos, working, axis=1)
    return filtered.astype(x.dtype)


def remove_common_mode(
    data: ArrayLike,
    *,
    channel_mask: ArrayLike | None = None,
    estimator: str = "mean",
    adaptive: bool = False,
    energy_mad_multiplier: float | None = 3.0,
) -> CommonModeResult:
    """Subtract a mean/median template estimated from selected channels.

    By default, exclude channels with energy > median(E)+3*1.4826*MAD(E)
    from TEMPLATE estimation only. This does not repair/remove those traces.
    Set energy_mad_multiplier=None to use only the explicit channel_mask.
    MAD is never used to divide any trace. adaptive=True fits a subtraction
    gain <x,n>/<n,n>; the default gain is one. Inspect selected_channels.
    """
    x = _matrix(data)
    if estimator not in ("mean", "median"):
        raise ValueError("estimator must be mean or median")
    selected = np.ones(len(x), bool) if channel_mask is None else np.asarray(channel_mask).copy()
    if selected.dtype != bool or selected.shape != (len(x),) or not selected.any():
        raise ValueError("channel_mask must be Boolean and select at least one channel")
    energy = np.einsum("ij,ij->i", x, x, dtype=np.float64) / x.shape[1]
    threshold = np.inf
    if energy_mad_multiplier is not None:
        _positive(energy_mad_multiplier, "energy_mad_multiplier", allow_zero=True)
        center = np.median(energy[selected])
        threshold = center + energy_mad_multiplier * 1.4826 * np.median(
            np.abs(energy[selected] - center)
        )
        selected &= energy <= threshold
    if not selected.any():
        raise ValueError("No channels survived common-mode selection")
    template = (
        np.mean(x[selected], axis=0, dtype=np.float64)
        if estimator == "mean"
        else np.median(x[selected], axis=0)
    )
    power = template @ template
    gains = np.ones(len(x))
    if adaptive:
        gains = (x @ template) / power if power > 0 else np.zeros(len(x))
    cleaned = (x - gains[:, None] * template).astype(x.dtype)
    return CommonModeResult(cleaned, template, gains, selected, energy, float(threshold))


def interpolate_bad_channels(data: ArrayLike, bad_channels: ArrayLike) -> np.ndarray:
    """Linear interpolation along channels; require a good neighbour on both sides.

    Use before FFT analysis to limit contamination from known bad traces.
    Do not interpret or report the repaired channels as measured observations.
    The caller keeps bad_channels and excludes them from subsequent summaries.
    """
    x = np.asarray(data)
    bad = np.asarray(bad_channels)
    if x.ndim != 2 or np.iscomplexobj(x) or min(x.shape) < 2:
        raise ValueError("Expected a real (channel, time) matrix")
    if bad.dtype != bool or bad.shape != (len(x),):
        raise ValueError("bad_channels must be a Boolean channel mask")
    if bad[0] or bad[-1]:
        raise ValueError("Cannot interpolate edge channels; crop them explicitly")
    out = np.array(x, dtype=np.result_type(x.dtype, np.float32), copy=True)
    good = np.flatnonzero(~bad)
    if not np.isfinite(out[good]).all():
        raise ValueError("Unflagged channels contain NaN/Inf")
    for i in np.flatnonzero(bad):
        upper = good[np.searchsorted(good, i)]
        lower = good[np.searchsorted(good, i) - 1]
        fraction = (i - lower) / (upper - lower)
        out[i] = (1 - fraction) * out[lower] + fraction * out[upper]
    return out


def _spectral_axis(
    data: ArrayLike,
    spacing: float,
    axis: int,
    response: Callable[[np.ndarray], np.ndarray],
    pad_samples: int,
    pad_mode: str,
) -> np.ndarray:
    x = _matrix(data)
    _positive(spacing, "spacing")
    if not isinstance(pad_samples, (int, np.integer)) or pad_samples < 0:
        raise ValueError("pad_samples must be a nonnegative integer")
    if pad_mode not in ("reflect", "constant", "edge"):
        raise ValueError("pad_mode must be reflect, constant, or edge")
    padding = [(0, 0), (0, 0)]
    padding[axis] = (pad_samples, pad_samples)
    padded = np.pad(x, padding, mode=pad_mode)
    freq = fft.rfftfreq(padded.shape[axis], spacing)
    transfer = response(freq).astype(x.dtype)
    shape = [1, 1]
    shape[axis] = len(transfer)
    spectrum = fft.rfft(padded, axis=axis, workers=1) * transfer.reshape(shape)
    filtered = fft.irfft(spectrum, n=padded.shape[axis], axis=axis, workers=1)
    crop = [slice(None), slice(None)]
    crop[axis] = slice(pad_samples, pad_samples + x.shape[axis])
    return filtered[tuple(crop)].copy()


def _ricker_response(frequency: np.ndarray, peak: float) -> np.ndarray:
    ratio = frequency / peak
    return ratio**2 * np.exp(1 - ratio**2)


def ricker_time(
    data: ArrayLike,
    dt_s: float,
    *,
    peak_frequency_hz: float,
    pad_samples: int = 256,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Zero-phase temporal Ricker band; peak gain one at peak_frequency_hz.

    W(f)=(f/f0)^2 exp(1-(f/f0)^2). This is an analysis band, not a
    reconstruction bank. Bandwidth changes temporal localisation, not units.
    """
    _positive(dt_s, "dt_s")
    _positive(peak_frequency_hz, "peak_frequency_hz")
    if peak_frequency_hz >= 0.5 / dt_s:
        raise ValueError("Ricker peak must lie below temporal Nyquist")
    return _spectral_axis(
        data, dt_s, 1, lambda f: _ricker_response(f, peak_frequency_hz), pad_samples, pad_mode
    )


def ricker_space(
    data: ArrayLike,
    dx_m: float,
    *,
    peak_wavelength_m: float,
    pad_samples: int = 128,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Zero-phase spatial Ricker band; peak gain one at k=1/wavelength.

    This selects spatial wavelengths, not temporal frequencies or velocities.
    For a travelling mode, f=v/lambda couples the two. No post-filter velocity
    rescaling is appropriate. Filtering can change the mixture being measured.
    """
    _positive(dx_m, "dx_m")
    _positive(peak_wavelength_m, "peak_wavelength_m")
    if peak_wavelength_m <= 2 * dx_m:
        raise ValueError("Peak wavelength must exceed two channel spacings")
    return _spectral_axis(
        data, dx_m, 0, lambda k: _ricker_response(k, 1 / peak_wavelength_m), pad_samples, pad_mode
    )


def log_gabor_space(
    data: ArrayLike,
    dx_m: float,
    *,
    peak_wavelength_m: float,
    log_sigma: float = 0.6,
    pad_samples: int = 128,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Optional spatial log-Gabor band, peak gain one; log_sigma uses natural log."""
    _positive(dx_m, "dx_m")
    _positive(peak_wavelength_m, "peak_wavelength_m")
    _positive(log_sigma, "log_sigma")
    if peak_wavelength_m <= 2 * dx_m:
        raise ValueError("Peak wavelength must exceed two channel spacings")

    def response(k: np.ndarray) -> np.ndarray:
        out = np.zeros_like(k)
        use = k > 0
        out[use] = np.exp(-0.5 * (np.log(k[use] * peak_wavelength_m) / log_sigma) ** 2)
        return out

    return _spectral_axis(data, dx_m, 0, response, pad_samples, pad_mode)
