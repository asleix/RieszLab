"""Optional horizontal propagation fitting from projected cable slowness.

Copyright (c) 2026 Aleix Seguí Ugalde.
Affiliation: University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from ._validation import positive as _positive


@dataclass(frozen=True)
class FitConfig:
    """Huber by default (transition scale 0.03 s/km).

    MSE remains an explicit alternative for comparison. Fit residuals and
    huber_scale_s_per_m use SI slowness units; bin weights default to equal.
    """

    loss: str = "huber"
    huber_scale_s_per_m: float = 3e-5
    max_condition_number: float = 30.0

    def __post_init__(self) -> None:
        if self.loss not in ("mse", "huber"):
            raise ValueError("loss must be mse or huber")
        _positive(self.huber_scale_s_per_m, "huber_scale_s_per_m")
        _positive(self.max_condition_number, "max_condition_number")


@dataclass
class PropagationFit:
    slowness_en_s_per_m: np.ndarray
    propagation_azimuth_degrees: float
    backazimuth_degrees: float
    predicted_s_per_m: np.ndarray
    residual_s_per_m: np.ndarray
    used: np.ndarray
    condition_number: float
    config: FitConfig


@dataclass
class BinnedSlowness:
    distance_m: np.ndarray
    slowness_s_per_m: np.ndarray
    tangent_en: np.ndarray
    count: np.ndarray
    bin_width_m: float
    bin_ids: np.ndarray
    bin_origin_m: float


@dataclass
class DirectionInterval:
    fit: PropagationFit
    backazimuth_samples_degrees: np.ndarray
    interval_degrees: np.ndarray
    requested_draws: int
    accepted_draws: int
    confidence: float


def fit_propagation(
    tangent_en: ArrayLike,
    slowness_s_per_m: ArrayLike,
    *,
    config: FitConfig | None = None,
    weights: ArrayLike | None = None,
) -> PropagationFit:
    """Fit p_app = t_E*p_E + t_N*p_N, with p_U=0 explicitly assumed.

    Pass horizontal components of UNIT 3-D tangents without renormalising
    their horizontal projection. Two-column averaged tangents are also
    accepted. Missing observations are omitted and reported in used.
    weights, if given, multiply squared residuals for MSE; for Huber the
    loss acts on sqrt(weight)*residual. No weights are inferred from energy.
    Nearly straight/ill-conditioned apertures raise rather than invent cones.
    Backazimuth is the local reverse propagation bearing, not a located source.
    """
    cfg = config or FitConfig()
    tangent = np.asarray(tangent_en, float)
    p = np.asarray(slowness_s_per_m, float)
    if tangent.ndim != 2 or tangent.shape[1] not in (2, 3) or p.shape != (len(tangent),):
        raise ValueError("Expected (channels,2/3) tangents and one slowness per channel")

    # 1. Form horizontal cable projections without renormalising the 3-D unit tangents.
    design = tangent[:, :2]
    w = np.ones_like(p) if weights is None else np.asarray(weights, float)
    if w.shape != p.shape or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("weights must be finite, nonnegative and match slowness")
    used = np.isfinite(design).all(axis=1) & np.isfinite(p) & (w > 0)
    if used.sum() < 3:
        raise ValueError("At least three finite observations with positive weights are required")

    # 2. Weight residuals and check that the aperture resolves both horizontal components.
    a = design[used] * np.sqrt(w[used, None])
    b = p[used] * np.sqrt(w[used])
    condition = np.linalg.cond(a)
    if np.linalg.matrix_rank(a) < 2 or condition > cfg.max_condition_number:
        raise ValueError(f"Direction is not resolved by this aperture (condition={condition:.3g})")

    # 3. Solve least squares; optionally use that solution to initialise the Huber fit.
    vector = np.linalg.lstsq(a, b, rcond=None)[0]
    if cfg.loss == "huber":
        # The Huber transition is at |sqrt(weight)*residual| = scale, in s/m.
        # Scaling residuals also avoids unit-dependent solver stopping.
        scale = cfg.huber_scale_s_per_m
        solved = least_squares(
            lambda z: (a @ z - b) / scale,
            vector,
            jac=lambda z: a / scale,
            loss="huber",
            f_scale=1.0,
            x_scale="jac",
        )
        if not solved.success:
            raise RuntimeError(f"Propagation fit did not converge: {solved.message}")
        vector = solved.x

    # 4. Convert to clockwise-from-north bearings; incoming backazimuth reverses the vector.
    norm = np.linalg.norm(vector)
    azimuth = float(np.degrees(np.arctan2(vector[0], vector[1])) % 360) if norm > 0 else np.nan
    prediction = design @ vector
    return PropagationFit(
        vector,
        azimuth,
        (azimuth + 180) % 360,
        prediction,
        p - prediction,
        used,
        float(condition),
        cfg,
    )


def bin_slowness(
    distance_m: ArrayLike,
    slowness_s_per_m: ArrayLike,
    tangent_en: ArrayLike,
    *,
    bin_width_m: float = 250.0,
    min_points: int = 5,
    bin_origin_m: float = 0.0,
) -> BinnedSlowness:
    """Per-distance-bin median p and mean tangent; each retained bin is one vote."""
    s, p, tangent = (
        np.asarray(distance_m, float),
        np.asarray(slowness_s_per_m, float),
        np.asarray(tangent_en, float),
    )
    _positive(bin_width_m, "bin_width_m")
    if not np.isfinite(bin_origin_m):
        raise ValueError("bin_origin_m must be finite")
    if s.ndim != 1 or len(s) < 1 or not np.isfinite(s).all() or np.any(np.diff(s) <= 0):
        raise ValueError("distance_m must be finite and strictly increasing")
    if p.shape != s.shape or tangent.shape not in ((len(s), 2), (len(s), 3)):
        raise ValueError("Slowness/tangents must match distances")
    if not isinstance(min_points, (int, np.integer)) or min_points < 1:
        raise ValueError("min_points must be a positive integer")

    # 1. Assign physical-distance bin IDs, preserving gaps for the later block bootstrap.
    ids = np.floor((s - bin_origin_m) / bin_width_m).astype(int)
    valid = np.isfinite(p) & np.isfinite(tangent).all(axis=1)

    # 2. Pair median slowness with the mean tangent of the same accepted channels.
    rows = []
    for index in np.unique(ids):
        use = (ids == index) & valid
        if use.sum() >= min_points:
            rows.append(
                (
                    np.mean(s[use]),
                    np.median(p[use]),
                    *np.mean(tangent[use, :2], axis=0),
                    use.sum(),
                    index,
                )
            )
    table = np.asarray(rows, float).reshape(-1, 6)
    return BinnedSlowness(
        table[:, 0],
        table[:, 1],
        table[:, 2:4],
        table[:, 4].astype(int),
        bin_width_m,
        table[:, 5].astype(int),
        bin_origin_m,
    )


def bootstrap_direction(
    bins: BinnedSlowness,
    *,
    config: FitConfig | None = None,
    block_length_m: float = 500.0,
    draws: int = 400,
    confidence: float = 0.95,
    seed: int = 0,
    min_accepted_fraction: float = 0.5,
    min_bins: int = 10,
) -> DirectionInterval:
    """Spatial block pairs bootstrap, conditional on selected packets and geometry.

    Resample whole contiguous groups of bins, never bridge missing-bin gaps.
    Intervals omit source/model/geometry/selection uncertainty and must not be
    called a location cone. Bounds are unwrapped around the fitted bearing;
    they may extend below 0 or above 360 degrees. Report accepted_draws.
    """
    _positive(block_length_m, "block_length_m")
    if (
        not isinstance(draws, (int, np.integer))
        or draws < 2
        or not 0 < confidence < 1
        or not 0 < min_accepted_fraction <= 1
    ):
        raise ValueError("Invalid draws, confidence, or min_accepted_fraction")
    s = bins.distance_m
    if not isinstance(min_bins, (int, np.integer)) or min_bins < 3:
        raise ValueError("min_bins must be an integer >=3")
    if len(s) < min_bins or np.any(np.diff(s) <= 0):
        raise ValueError(f"At least {min_bins} ordered bins required")

    # 1. Fit the original bins once; each resample uses the same loss and settings.
    fit = fit_propagation(bins.tangent_en, bins.slowness_s_per_m, config=config)

    # 2. Build overlapping blocks of contiguous bin IDs, never spanning missing-bin gaps.
    spacing = bins.bin_width_m
    count = max(1, int(round(block_length_m / spacing)))
    blocks = [
        np.arange(i, i + count)
        for i in range(len(s) - count + 1)
        if np.all(np.diff(bins.bin_ids[i : i + count]) == 1)
    ]
    if len(blocks) < 2:
        raise ValueError(
            "Too few contiguous blocks; shorten block_length_m or enlarge the aperture"
        )

    # 3. Resample blocks with replacement, keeping each slowness/tangent pair together.
    rng = np.random.default_rng(seed)
    angles = []
    for _ in range(draws):
        sample = np.concatenate(
            [blocks[j] for j in rng.integers(len(blocks), size=int(np.ceil(len(s) / count)))]
        )[: len(s)]
        try:
            trial = fit_propagation(
                bins.tangent_en[sample], bins.slowness_s_per_m[sample], config=fit.config
            )
        except ValueError:
            continue
        if np.isfinite(trial.backazimuth_degrees):
            angles.append(trial.backazimuth_degrees)
    if len(angles) < max(2, int(np.ceil(draws * min_accepted_fraction))):
        raise ValueError(f"Only {len(angles)}/{draws} bootstrap fits were identifiable")

    # 4. Centre angular differences on the fitted bearing before taking interval quantiles.
    angles = np.asarray(angles)
    delta = (angles - fit.backazimuth_degrees + 180) % 360 - 180
    tail = (1 - confidence) / 2
    interval = fit.backazimuth_degrees + np.quantile(delta, [tail, 1 - tail])
    return DirectionInterval(fit, angles, interval, draws, len(angles), confidence)
