"""Amplitude-weighted Gaussian tensors: fixed, adaptive or slowness-steered.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
Tensor formulation: Unser, Sage & Van De Ville (2009), IEEE TIP 18, 2402–2418.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal, sparse
from scipy.ndimage import gaussian_filter1d

from ._validation import divide, matrix, positive


@dataclass(frozen=True, kw_only=True)
class TensorConfig:
    """Common tensor settings; tau has squared input-amplitude units."""

    tau: float = 0.0
    truncate: float = 4.0

    def __post_init__(self) -> None:
        positive(self.tau, "tau", allow_zero=True)
        positive(self.truncate, "truncate")

    def average(
        self, data: np.ndarray, dx_m: float, dt_s: float, reference_velocity_mps: float | None
    ) -> np.ndarray:
        raise NotImplementedError("Choose GaussianTensorConfig or SteeredTensorConfig")


@dataclass(frozen=True)
class GaussianTensorConfig(TensorConfig):
    """Axis-aligned SDs in metres/seconds; optionally one width per axis sample."""

    sigma_space_m: float | np.ndarray = 90.0
    sigma_time_s: float | np.ndarray = 0.09

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("sigma_space_m", "sigma_time_s"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim > 1 or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError(f"{name} must be a nonnegative scalar or finite 1D profile")
            if value.ndim:
                value.setflags(write=False)
                object.__setattr__(self, name, value)

    def average(
        self,
        data: np.ndarray,
        dx_m: float,
        dt_s: float,
        reference_velocity_mps: float | None = None,
    ) -> np.ndarray:
        out = _smooth_axis(data, np.asarray(self.sigma_time_s) / dt_s, 1, self.truncate)
        return _smooth_axis(out, np.asarray(self.sigma_space_m) / dx_m, 0, self.truncate)


@dataclass(frozen=True)
class SteeredTensorConfig(TensorConfig):
    """Centred Gaussian aligned with t=p*s; principal SDs are metres in (s,c*t).

    Steering changes the averaging neighbourhood, not the Riesz metric.
    It is separate from the subsequent direction-selection weight.
    """

    slowness_s_per_m: float
    sigma_parallel_m: float = 300.0
    sigma_normal_m: float = 50.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(self.slowness_s_per_m):
            raise ValueError("slowness_s_per_m must be finite")
        positive(self.sigma_parallel_m, "sigma_parallel_m")
        positive(self.sigma_normal_m, "sigma_normal_m")

    def kernel(self, dx_m: float, dt_s: float, reference_velocity_mps: float) -> np.ndarray:
        """Positive, unit-sum sampled kernel, truncated at marginal SDs."""
        positive(dx_m, "dx_m")
        positive(dt_s, "dt_s")
        positive(reference_velocity_mps, "reference_velocity_mps")
        c = reference_velocity_mps
        along = np.array([1.0, c * self.slowness_s_per_m])
        along /= np.linalg.norm(along)
        normal = np.array([-along[1], along[0]])
        cov = self.sigma_parallel_m**2 * np.outer(along, along)
        cov += self.sigma_normal_m**2 * np.outer(normal, normal)
        sampling = np.array([dx_m, c * dt_s])
        cov /= np.outer(sampling, sampling)
        radius = np.ceil(self.truncate * np.sqrt(np.diag(cov))).astype(int)
        s = np.arange(-radius[0], radius[0] + 1)[:, None]
        t = np.arange(-radius[1], radius[1] + 1)[None, :]
        inv = np.linalg.inv(cov)
        weights = np.exp(-0.5 * (inv[0, 0] * s * s + 2 * inv[0, 1] * s * t + inv[1, 1] * t * t))
        return weights / weights.sum()

    def average(
        self,
        data: np.ndarray,
        dx_m: float,
        dt_s: float,
        reference_velocity_mps: float | None = None,
    ) -> np.ndarray:
        if reference_velocity_mps is None:
            raise ValueError("A steered tensor needs the Riesz reference velocity")
        kernel = self.kernel(dx_m, dt_s, reference_velocity_mps)
        radius = [(n // 2, n // 2) for n in kernel.shape]
        padded = np.pad(data.astype(np.float64), radius, mode="symmetric")
        # Linear overlap-add convolution limits memory and avoids circular wraparound.
        return signal.oaconvolve(padded, kernel, mode="valid").astype(data.dtype)


@dataclass
class TensorResult:
    """Energies in input units squared; axial orientation in radians."""

    j_ss: np.ndarray
    j_st: np.ndarray
    j_tt: np.ndarray
    lambda_max: np.ndarray
    lambda_min: np.ndarray
    orientation: np.ndarray
    coherence: np.ndarray
    coherence_regularized: np.ndarray
    config: TensorConfig


def _smooth_axis(data: np.ndarray, sigma: ArrayLike, axis: int, truncate: float) -> np.ndarray:
    width = np.asarray(sigma, dtype=float)
    if width.ndim == 0:
        if width == 0:
            return data.copy()
        return gaussian_filter1d(data, float(width), axis=axis, mode="reflect", truncate=truncate)
    if width.shape != (data.shape[axis],):
        raise ValueError("Tensor width profile must match its corresponding matrix axis")
    if np.all(width == width[0]):
        return _smooth_axis(data, width[0], axis, truncate)
    n = len(width)
    rows, cols, values = [], [], []
    for i, w in enumerate(width):
        radius = int(truncate * w + 0.5)
        lag = np.arange(-radius, radius + 1)
        weights = np.exp(-0.5 * (lag / w) ** 2) if w > 0 else np.ones(1)
        weights /= weights.sum()
        reflected = (i + lag) % (2 * n)
        reflected = np.where(reflected < n, reflected, 2 * n - 1 - reflected)
        rows.extend([i] * len(lag))
        cols.extend(reflected)
        values.extend(weights)
    operator = sparse.csr_matrix((values, (rows, cols)), shape=(n, n), dtype=data.dtype)
    return operator @ data if axis == 0 else (operator @ data.T).T


def structure_tensor(
    riesz_space: ArrayLike,
    riesz_time: ArrayLike,
    dx_m: float,
    dt_s: float,
    *,
    config: TensorConfig,
    reference_velocity_mps: float | None = None,
) -> TensorResult:
    """J=G*(r rᵀ): squared Riesz amplitude already supplies the weighting.

    X=(lambda_max-lambda_min)/(lambda_max+lambda_min+tau). Adding tau is
    equivalent to loading (tau/2)I; it is not a signal-detection probability.
    """
    positive(dx_m, "dx_m")
    positive(dt_s, "dt_s")
    a, b = matrix(riesz_space), matrix(riesz_time)
    if a.shape != b.shape:
        raise ValueError("Riesz components must have identical shapes")

    def smooth(x: np.ndarray) -> np.ndarray:
        return config.average(x, dx_m, dt_s, reference_velocity_mps)

    jss, jst, jtt = smooth(a * a), smooth(a * b), smooth(b * b)
    # Remove tiny negative diagonal roundoff from FFT-based averaging.
    np.maximum(jss, 0, out=jss)
    np.maximum(jtt, 0, out=jtt)
    trace = jss + jtt
    contrast = np.minimum(np.hypot(jss - jtt, 2 * jst), trace)
    major, minor = (trace + contrast) / 2, (trace - contrast) / 2
    theta = np.mod(0.5 * np.arctan2(2 * jst, jss - jtt), np.pi)
    theta = np.where(contrast > 8 * np.finfo(theta.dtype).eps * trace, theta, np.nan)
    return TensorResult(
        jss,
        jst,
        jtt,
        major,
        minor,
        theta,
        divide(contrast, trace, fill=0),
        divide(contrast, trace + config.tau, fill=0),
        config,
    )
