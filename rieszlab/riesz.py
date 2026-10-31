"""Full-matrix monogenic analysis in distance and time.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

import numpy as np
from numpy.typing import ArrayLike
from scipy import fft

from ._validation import divide as _divide, matrix as _matrix, positive as _positive
from .tensors import GaussianTensorConfig, TensorConfig, TensorResult, structure_tensor

METRES_PER_KILOMETRE = 1000.0


@dataclass(frozen=True)
class RieszConfig:
    """FFT choices. Omitted c is resolved from explicit dx/dt, never guessed."""

    reference_velocity_mps: float | None = None
    pad_samples: tuple[int, int] = (128, 256)
    pad_mode: str = "reflect"
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.reference_velocity_mps is not None:
            _positive(self.reference_velocity_mps, "reference_velocity_mps")
        if len(self.pad_samples) != 2 or any(
            not isinstance(p, (int, np.integer)) or p < 0 for p in self.pad_samples
        ):
            raise ValueError("pad_samples must contain two nonnegative integers")
        if self.pad_mode not in ("reflect", "constant", "edge"):
            raise ValueError("pad_mode must be reflect, constant (zero), or edge")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")


@dataclass
class _AnalysisDomain:
    even: np.ndarray
    riesz_space: np.ndarray
    riesz_time: np.ndarray
    crop: tuple[slice, slice]


@dataclass
class MonogenicResult:
    """Primary attributes on the input grid; padded coefficients are reused internally."""

    even: np.ndarray
    riesz_space: np.ndarray
    riesz_time: np.ndarray
    amplitude: np.ndarray
    phase_unsigned: np.ndarray
    orientation_instantaneous: np.ndarray
    dx_m: float
    dt_s: float
    config: RieszConfig
    _domain: _AnalysisDomain = field(repr=False)


@dataclass
class RieszResult:
    """Unmasked attributes; without a tensor, orientation and slowness are instantaneous."""

    monogenic: MonogenicResult
    tensor: TensorResult | None
    slowness_s_per_m: np.ndarray
    phase_directional: np.ndarray
    valid: np.ndarray
    _tensor_domain: TensorResult | None = field(default=None, repr=False)

    def with_tensor(self, tensor: TensorConfig | None) -> RieszResult:
        """Return new estimates using the cached monogenic components.

        Does not repeat the Riesz transform or modify this result. Primary
        arrays are shared, not copied. None restores instantaneous estimates
        with coherency=None; sampling, metric and preprocessing stay fixed.
        """
        return _analyse_monogenic(self.monogenic, tensor=tensor)

    @property
    def amplitude(self) -> np.ndarray:
        return self.monogenic.amplitude

    @property
    def orientation(self) -> np.ndarray:
        return (
            self.monogenic.orientation_instantaneous
            if self.tensor is None
            else self.tensor.orientation
        )

    @property
    def coherency(self) -> np.ndarray | None:
        """Regularised directional coherency, or None when tensor=None."""
        return None if self.tensor is None else self.tensor.coherency_regularized

    @property
    def slowness_s_per_km(self) -> np.ndarray:
        return METRES_PER_KILOMETRE * self.slowness_s_per_m

    def velocity_mps(self, *, min_abs_slowness_s_per_m: float = 0.0) -> np.ndarray:
        _positive(min_abs_slowness_s_per_m, "min_abs_slowness_s_per_m", allow_zero=True)
        p = self.slowness_s_per_m
        return np.where(np.abs(p) > min_abs_slowness_s_per_m, _divide(1.0, p), np.nan)


def _resolve_config(
    dx_m: float | None,
    dt_s: float | None,
    config: RieszConfig | None,
    reference_velocity_mps: float | None,
) -> RieszConfig:
    if dx_m is None or dt_s is None:
        raise ValueError(
            "Supply dx_m and dt_s. Physical widths require sampling; c alone is not a sampling grid."
        )
    _positive(dx_m, "dx_m")
    _positive(dt_s, "dt_s")
    cfg = config or RieszConfig()
    c = cfg.reference_velocity_mps
    if reference_velocity_mps is not None:
        _positive(reference_velocity_mps, "reference_velocity_mps")
        if c is not None and not np.isclose(c, reference_velocity_mps, rtol=1e-12, atol=0):
            raise ValueError("Conflicting reference velocities in the arguments and RieszConfig")
        c = reference_velocity_mps
    return replace(cfg, reference_velocity_mps=dx_m / dt_s if c is None else c)


def _symbols(
    shape: tuple[int, int], dx_m: float, dt_s: float, reference_velocity_mps: float, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    k = fft.fftfreq(shape[0], dx_m).astype(dtype)[:, None]
    y = fft.rfftfreq(shape[1], dt_s).astype(dtype)[None, :] / reference_velocity_mps
    radius = np.hypot(k, y)
    ms, mt = _divide(k, radius, fill=0), _divide(y, radius, fill=0)
    if shape[0] % 2 == 0:
        ms[shape[0] // 2, :] = 0
    if shape[1] % 2 == 0:
        mt[:, -1] = 0
    return ms, mt


def _monogenic_signal(
    data: ArrayLike,
    dx_m: float | None = None,
    dt_s: float | None = None,
    *,
    config: RieszConfig | None = None,
    reference_velocity_mps: float | None = None,
) -> MonogenicResult:
    """Analyse a finite (channel,time) matrix with one full padded FFT.

    c defaults to the supplied dx/dt; an explicit reference_velocity_mps
    overrides this metric, not the sampling. Symbols are -i*(k,f/c)/hypot(k,f/c).
    Amplitude is sqrt(u²+Rs²+Rt²); unsigned phase is atan2(hypot(Rs,Rt),u).
    No preprocessing, tensor averaging, normalisation or gating is performed.
    """
    cfg = _resolve_config(dx_m, dt_s, config, reference_velocity_mps)
    x = _matrix(data, cfg.dtype)
    ps, pt = cfg.pad_samples
    # Analyse the full record; retain padding for consistent later reconstruction.
    u = np.pad(x, ((ps, ps), (pt, pt)), mode=cfg.pad_mode)
    ms, mt = _symbols(u.shape, dx_m, dt_s, cfg.reference_velocity_mps, u.dtype)
    spectrum = fft.rfft2(u, workers=1)
    rs = fft.irfft2(-1j * ms * spectrum, s=u.shape, workers=1)
    rt = fft.irfft2(-1j * mt * spectrum, s=u.shape, workers=1)
    crop = (slice(ps, ps + x.shape[0]), slice(pt, pt + x.shape[1]))
    domain = _AnalysisDomain(u, rs, rt, crop)
    u, rs, rt = (a[crop] for a in (u, rs, rt))
    q = np.hypot(rs, rt)
    amplitude = np.hypot(u, q)
    phase = np.where(amplitude > 0, np.arctan2(q, u), np.nan)
    orientation = np.where(q > 0, np.mod(np.arctan2(rt, rs), np.pi), np.nan)
    return MonogenicResult(u, rs, rt, amplitude, phase, orientation, dx_m, dt_s, cfg, domain)


def _domain_config(mono: MonogenicResult, config: TensorConfig) -> TensorConfig:
    if not isinstance(config, GaussianTensorConfig):
        return config
    kwargs = {}
    for axis, name in enumerate(("sigma_space_m", "sigma_time_s")):
        width = np.asarray(getattr(config, name))
        if width.ndim:
            if width.shape != (mono.even.shape[axis],):
                raise ValueError("Tensor width profile must match its corresponding matrix axis")
            pad = mono.config.pad_samples[axis]
            kwargs[name] = np.pad(width, (pad, pad), mode="edge")
    return replace(config, **kwargs)


def orientation_to_slowness(orientation: ArrayLike, reference_velocity_mps: float) -> np.ndarray:
    """Signed p=-cos(theta)/(c*sin(theta)), in s/m; no wavelet correction applies."""
    _positive(reference_velocity_mps, "reference_velocity_mps")
    theta = np.asarray(orientation)
    ns, nt = np.cos(theta), np.sin(theta)
    eps = np.finfo(np.result_type(theta, np.float32)).eps
    return np.where(np.abs(nt) > 8 * eps, _divide(-ns, reference_velocity_mps * nt), np.nan)


def _analyse_monogenic(mono: MonogenicResult, *, tensor: TensorConfig | None = None) -> RieszResult:
    """Derive direction, phase and slowness without repeating the Riesz FFT.

    tensor=None uses instantaneous orientation and returns coherency=None.
    Otherwise averaging is performed before cropping the padded coefficients.
    Directional phase uses the normal with nonnegative temporal component.
    """
    full, ten = None, None
    theta = mono.orientation_instantaneous
    if tensor is not None:
        full = structure_tensor(
            mono._domain.riesz_space,
            mono._domain.riesz_time,
            mono.dx_m,
            mono.dt_s,
            config=_domain_config(mono, tensor),
            reference_velocity_mps=mono.config.reference_velocity_mps,
        )
        values = {
            f.name: getattr(full, f.name)[mono._domain.crop]
            for f in fields(TensorResult)
            if f.name != "config"
        }
        ten = TensorResult(**values, config=tensor)
        theta = ten.orientation
    p = orientation_to_slowness(theta, mono.config.reference_velocity_mps)
    quadrature = np.cos(theta) * mono.riesz_space + np.sin(theta) * mono.riesz_time
    phase = np.arctan2(quadrature, mono.even)
    valid = np.isfinite(p) & (mono.amplitude > 0)
    return RieszResult(mono, ten, p, phase, valid, full)


def monogenic(
    data: ArrayLike,
    dx_m: float | None = None,
    dt_s: float | None = None,
    *,
    reference_velocity_mps: float | None = None,
    config: RieszConfig | None = None,
    tensor: TensorConfig | None = None,
) -> RieszResult:
    """Monogenic attributes and optional tensor estimates on a (channel,time) field.

    dx_m/dt_s supplies the metric unless c is explicitly given. Both sample
    spacings are required. tensor=None performs no averaging and has no coherency.
    Positive slowness means later arrival at increasing channel distance.
    Validity is numerical only: nonlocal Riesz tails can also be coherent.
    Use result.with_tensor(config) to change averaging without recomputing
    the Riesz components, or result.with_tensor(None) to remove averaging.
    """
    # 1. Decompose the input into the even field and two Riesz components.
    mono = _monogenic_signal(
        data, dx_m, dt_s, config=config, reference_velocity_mps=reference_velocity_mps
    )
    # 2. Optionally average directions; 3. convert orientation to phase and slowness.
    return _analyse_monogenic(mono, tensor=tensor)
