"""Associate selected oscillations using directional phase and tensor orientation.

Copyright (c) 2026 Aleix Seguí Ugalde.
Affiliation: University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
from numpy.typing import ArrayLike

from ._validation import positive as _positive
from .riesz import RieszResult


@dataclass(frozen=True)
class AssociationConfig:
    target_phase_rad: float = 0.0
    coherency_min: float = 0.4
    amplitude_min: float = 0.0
    max_period_fraction: float = 0.35
    max_orientation_change_degrees: float = 40.0
    max_gap_m: float = 30.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.target_phase_rad) or not 0 <= self.coherency_min <= 1:
            raise ValueError("Require finite target phase and coherency_min in [0,1]")
        _positive(self.amplitude_min, "amplitude_min", allow_zero=True)
        _positive(self.max_gap_m, "max_gap_m", allow_zero=True)
        if not 0 < self.max_period_fraction < 0.5:
            raise ValueError(
                "max_period_fraction must be in (0,0.5) to limit cycle jumps"
            )
        if not 0 <= self.max_orientation_change_degrees <= 90:
            raise ValueError("max_orientation_change_degrees must lie in [0,90]")


@dataclass
class Crossings:
    time_s: np.ndarray
    sample_index: np.ndarray
    period_s: np.ndarray
    slowness_s_per_m: np.ndarray
    orientation_rad: np.ndarray
    coherency: np.ndarray
    amplitude: np.ndarray


@dataclass
class PhaseCandidates:
    channels: list[Crossings]
    distance_m: np.ndarray
    dt_s: float
    target_phase_rad: float
    time_start_s: float = 0.0


@dataclass
class PhaseTracks:
    """Arrays are (selected cycles, channels); gaps are NaN, indices are -1."""

    time_s: np.ndarray
    period_s: np.ndarray
    slowness_s_per_m: np.ndarray
    orientation_rad: np.ndarray
    coherency: np.ndarray
    amplitude: np.ndarray
    correction_periods: np.ndarray
    candidate_index: np.ndarray
    distance_m: np.ndarray
    config: AssociationConfig
    seed_channel: int
    seed_indices: np.ndarray
    dt_s: float
    time_start_s: float
    cycle_scores: np.ndarray | None = None


@dataclass
class SlownessSummary:
    per_cycle_s_per_m: np.ndarray
    median_s_per_m: np.ndarray
    lower_s_per_m: np.ndarray
    upper_s_per_m: np.ndarray
    count: np.ndarray
    quantiles: tuple[float, float]
    per_cycle_coherency: np.ndarray
    per_cycle_sample_count: np.ndarray
    min_cycles: int


def phase_candidates(
    result: RieszResult,
    *,
    distance_start_m: float = 0.0,
    time_start_s: float = 0.0,
    target_phase_rad: float = 0.0,
    channel_mask: ArrayLike | None = None,
) -> PhaseCandidates:
    """Positive-going crossings of arg(u+i*n.R), interpolated within a sample.

    The tensor normal has nonnegative temporal component. Rotation by the
    target phase distinguishes opposite half-cycles. Periods are estimated
    from adjacent crossings; no expected speed/period is imposed. Phase can
    be unreliable near mixtures or tensor-direction switches: association
    must still test continuity, energy and coherency.
    """
    if not np.isfinite([distance_start_m, time_start_s, target_phase_rad]).all():
        raise ValueError("Coordinates and target phase must be finite")
    if result.coherency is None:
        raise ValueError(
            "Cycle association requires tensor coherency; supply tensor=..."
        )
    mono = result.monogenic
    time = time_start_s + np.arange(mono.even.shape[1]) * mono.dt_s
    distance = distance_start_m + np.arange(mono.even.shape[0]) * mono.dx_m
    measured = (
        np.ones(len(distance), bool)
        if channel_mask is None
        else np.asarray(channel_mask)
    )
    if measured.shape != (len(distance),) or measured.dtype != bool:
        raise ValueError("channel_mask must be Boolean with one entry per channel")
    channels = []
    for i in range(len(distance)):
        # 1. Project the Riesz vector onto the tensor normal and rotate by the target phase.
        u, theta = mono.even[i], result.orientation[i]
        q = np.cos(theta) * mono.riesz_space[i] + np.sin(theta) * mono.riesz_time[i]
        a = u * np.cos(target_phase_rad) + q * np.sin(target_phase_rad)
        b = q * np.cos(target_phase_rad) - u * np.sin(target_phase_rad)

        # 2. Interpolate positive-going crossings; a > 0 excludes the opposite half-cycle.
        ids = np.flatnonzero((b[:-1] <= 0) & (b[1:] > 0))
        if not measured[i]:
            ids = ids[:0]
        alpha = -b[ids] / (b[ids + 1] - b[ids])
        keep = (a[ids] * (1 - alpha) + a[ids + 1] * alpha) > 0
        ids, alpha = ids[keep], alpha[keep]

        def interp(values: np.ndarray) -> np.ndarray:
            return values[ids] * (1 - alpha) + values[ids + 1] * alpha

        # 3. Estimate periods from neighbouring crossings, before any coherency gate.
        times = interp(time)
        delta = np.diff(times)
        period = (
            0.5 * (np.r_[delta[0], delta] + np.r_[delta, delta[-1]])
            if len(delta)
            else np.full(len(times), np.nan)
        )

        # 4. Interpolate attributes at each root; doubled angles respect axial periodicity.
        angle = np.mod(
            0.5 * np.arctan2(interp(np.sin(2 * theta)), interp(np.cos(2 * theta))),
            np.pi,
        )
        channels.append(
            Crossings(
                times,
                ids + alpha,
                period,
                interp(result.slowness_s_per_m[i]),
                angle,
                interp(result.coherency[i]),
                interp(result.amplitude[i]),
            )
        )
    return PhaseCandidates(
        channels, distance, mono.dt_s, target_phase_rad, time_start_s
    )


def associate_cycles(
    candidates: PhaseCandidates,
    seed_channel: int,
    seed_indices: ArrayLike | None = None,
    *,
    time_window_s: tuple[float, float] | None = None,
    config: AssociationConfig | None = None,
) -> PhaseTracks:
    """Trace both ways using dt = 0.5*(p_previous+p_candidate)*ds.

    Choose the closest phase crossing within the period-fraction tolerance,
    with coherency, amplitude and axial-turn gates. There are no jumps to
    remote energy maxima or automatic reseeds. Missing runs beyond max_gap_m
    stop that direction. Two tracks that merge are excluded in summaries.
    """
    cfg = config or AssociationConfig(target_phase_rad=candidates.target_phase_rad)
    if not np.isclose(
        np.angle(np.exp(1j * (cfg.target_phase_rad - candidates.target_phase_rad))), 0
    ):
        raise ValueError("Association target phase differs from candidate extraction")
    if not 0 <= seed_channel < len(candidates.channels):
        raise ValueError("seed_channel is outside the data")

    # 1. Use explicit seed crossings or all crossings inside the requested time window.
    if seed_indices is None:
        if (
            time_window_s is None
            or len(time_window_s) != 2
            or not np.isfinite(time_window_s).all()
            or time_window_s[0] >= time_window_s[1]
        ):
            raise ValueError("Supply seed_indices or an increasing time_window_s")
        t = candidates.channels[seed_channel].time_s
        seed_indices = np.flatnonzero((t >= time_window_s[0]) & (t <= time_window_s[1]))
    elif time_window_s is not None:
        raise ValueError("Choose seed_indices or time_window_s, not both")
    seeds = np.asarray(seed_indices)
    if (
        seeds.ndim != 1
        or seeds.dtype.kind not in "iu"
        or not len(seeds)
        or len(np.unique(seeds)) != len(seeds)
    ):
        raise ValueError(
            "seed_indices must be a nonempty set of unique integer indices"
        )
    if np.any(seeds < 0) or np.any(
        seeds >= len(candidates.channels[seed_channel].time_s)
    ):
        raise ValueError("Seed index outside candidate list")
    seeds = np.sort(seeds)

    # 2. Initialise missing tracks; rejected or unvisited samples remain NaN.
    shape = (len(seeds), len(candidates.distance_m))
    arrays = [np.full(shape, np.nan) for _ in range(7)]
    tracks = PhaseTracks(
        *arrays,
        np.full(shape, -1, int),
        candidates.distance_m.copy(),
        cfg,
        seed_channel,
        seeds,
        candidates.dt_s,
        candidates.time_start_s,
    )
    names = (
        "time_s",
        "period_s",
        "slowness_s_per_m",
        "orientation_rad",
        "coherency",
        "amplitude",
    )

    def accept(c: Crossings) -> np.ndarray:
        return (
            (c.coherency >= cfg.coherency_min)
            & (c.amplitude >= cfg.amplitude_min)
            & (c.period_s > 2 * candidates.dt_s)
            & np.isfinite(c.slowness_s_per_m)
        )

    def assign(j: int, i: int, k: int, correction: float = 0.0) -> None:
        for name in names:
            getattr(tracks, name)[j, i] = getattr(candidates.channels[i], name)[k]
        tracks.candidate_index[j, i] = k
        tracks.correction_periods[j, i] = correction

    # 3. Grow each admissible seed independently in both cable directions.
    for j, seed in enumerate(seeds):
        if not accept(candidates.channels[seed_channel])[seed]:
            continue
        assign(j, seed_channel, seed)
        for direction in (-1, 1):
            last, missing_distance = seed_channel, 0.0
            stop = len(candidates.channels) if direction > 0 else -1
            for i in range(seed_channel + direction, stop, direction):
                c = candidates.channels[i]
                ds = candidates.distance_m[i] - candidates.distance_m[last]
                # Predict each candidate using the mean of its and the last accepted slowness.
                predicted = (
                    tracks.time_s[j, last]
                    + 0.5 * (tracks.slowness_s_per_m[j, last] + c.slowness_s_per_m) * ds
                )
                residual = c.time_s - predicted
                period = np.minimum(c.period_s, tracks.period_s[j, last])
                turn = 0.5 * np.abs(
                    np.angle(
                        np.exp(
                            2j * (c.orientation_rad - tracks.orientation_rad[j, last])
                        )
                    )
                )
                # Gate coherency and continuity BEFORE choosing the nearest candidate.
                valid = (
                    accept(c)
                    & (
                        np.abs(residual)
                        <= cfg.max_period_fraction * period + candidates.dt_s / 2
                    )
                    & (turn <= np.deg2rad(cfg.max_orientation_change_degrees))
                )
                options = np.flatnonzero(valid)
                if not len(options):
                    # Bridge only short missing runs, without filling them or reseeding.
                    missing_distance += abs(
                        candidates.distance_m[i] - candidates.distance_m[i - direction]
                    )
                    if missing_distance > cfg.max_gap_m:
                        break
                    continue
                # Rank surviving crossings by timing error as a fraction of the local period.
                k = options[np.argmin(np.abs(residual[options]) / period[options])]
                assign(j, i, k, residual[k] / period[k])
                last, missing_distance = i, 0.0
    return tracks


def _track_time(result: RieszResult, tracks: PhaseTracks) -> np.ndarray:
    mono = result.monogenic
    if (
        tracks.time_s.shape[1] != mono.even.shape[0]
        or not np.isclose(tracks.dt_s, mono.dt_s)
        or not np.allclose(np.diff(tracks.distance_m), mono.dx_m)
    ):
        raise ValueError("Tracks and result must share sampling and channel count")
    return tracks.time_start_s + np.arange(mono.even.shape[1]) * mono.dt_s


def cycle_power(
    result: RieszResult,
    tracks: PhaseTracks,
    *,
    aperture_m: tuple[float, float],
    min_channels: int = 10,
) -> np.ndarray:
    """Median across channels of one-period mean squared strain; no direction ranking."""
    if (
        len(aperture_m) != 2
        or not np.isfinite(aperture_m).all()
        or aperture_m[0] >= aperture_m[1]
    ):
        raise ValueError("aperture_m must be an increasing finite pair")
    if not isinstance(min_channels, (int, np.integer)) or min_channels < 1:
        raise ValueError("min_channels must be a positive integer")

    # 1. Measure original analysed strain power over one period around each tracked crossing.
    t = _track_time(result, tracks)
    aperture = (tracks.distance_m >= aperture_m[0]) & (
        tracks.distance_m <= aperture_m[1]
    )
    scores = np.full(len(tracks.time_s), np.nan)
    for j in range(len(scores)):
        powers = []
        for i in np.flatnonzero(aperture & np.isfinite(tracks.time_s[j])):
            use = abs(t - tracks.time_s[j, i]) <= tracks.period_s[j, i] / 2
            if use.any():
                u = result.monogenic.even[i, use].astype(float)
                powers.append(np.mean(u * u))
        # 2. Rank cycles by median channel power, requiring sufficient aperture support.
        if len(powers) >= min_channels:
            scores[j] = np.median(powers)
    return scores


def select_cycles(
    tracks: PhaseTracks, scores: ArrayLike, *, count: int = 5, consecutive: bool = True
) -> PhaseTracks:
    """Highest-mean-power consecutive group, or strongest distinct cycles anywhere."""
    score = np.asarray(scores, float)
    if (
        score.shape != (len(tracks.time_s),)
        or not isinstance(count, (int, np.integer))
        or count < 1
    ):
        raise ValueError(
            "One score per cycle and a positive integer count are required"
        )
    # 1. Only cycles with a finite power score are eligible for selection.
    eligible = np.flatnonzero(np.isfinite(score))
    if consecutive:
        # 2a. Rank groups of adjacent original seed crossings by their mean power.
        groups = [
            np.arange(i, i + count)
            for i in range(len(score) - count + 1)
            if np.isfinite(score[i : i + count]).all()
            and np.all(np.diff(tracks.seed_indices[i : i + count]) == 1)
        ]
        if not groups:
            raise ValueError("No consecutive group with the requested count")
        ids = groups[int(np.argmax([np.mean(score[g]) for g in groups]))]
    else:
        # 2b. Select the strongest individual cycles, then restore their time order.
        if len(eligible) < count:
            raise ValueError("Too few scored cycles for the requested count")
        ids = np.sort(eligible[np.argsort(-score[eligible], kind="stable")[:count]])

    # 3. Copy selected tracks and scores without rerunning association.
    arrays = {
        f.name: getattr(tracks, f.name)[ids].copy()
        for f in fields(PhaseTracks)
        if isinstance(getattr(tracks, f.name), np.ndarray)
        and f.name not in ("distance_m", "cycle_scores")
    }
    return replace(tracks, **arrays, cycle_scores=score[ids].copy())


def summarize_tracks(
    result: RieszResult,
    tracks: PhaseTracks,
    *,
    half_period_fraction: float = 0.125,
    coherency_min: float = 0.0,
    min_cycles: int = 1,
    quantiles: tuple[float, float] = (0.25, 0.75),
) -> SlownessSummary:
    """Median p in a local phase neighbourhood, then median/quantiles over cycles.

    Sample gating is separate from the association gate. Quantiles measure
    observed spread, not confidence in a true slowness. Samples/cycles are
    correlated. Shared crossings and crossing-order conflicts are excluded.
    """
    if not 0 < half_period_fraction <= 0.5 or not 0 <= coherency_min <= 1:
        raise ValueError("Invalid phase neighbourhood or coherency threshold")
    if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError("quantiles must be an ordered pair in [0,1]")
    if result.coherency is None:
        raise ValueError("Cycle summaries require tensor coherency; supply tensor=...")
    if not isinstance(min_cycles, (int, np.integer)) or min_cycles < 1:
        raise ValueError("min_cycles must be a positive integer")

    # 1. Gate samples within each phase neighbourhood, then take within-cycle medians.
    t = _track_time(result, tracks)
    values = np.full_like(tracks.time_s, np.nan)
    coherencies = np.full_like(values, np.nan)
    sample_counts = np.zeros(values.shape, int)
    for j, i in zip(*np.where(np.isfinite(tracks.time_s))):
        use = (
            (
                abs(t - tracks.time_s[j, i])
                <= half_period_fraction * tracks.period_s[j, i]
            )
            & (result.coherency[i] >= coherency_min)
            & result.valid[i]
        )
        if use.any():
            values[j, i] = np.median(result.slowness_s_per_m[i, use])
            coherencies[j, i] = np.median(result.coherency[i, use])
            sample_counts[j, i] = use.sum()

    # 2. Exclude tracks sharing a crossing or reversing their relative temporal order.
    for i in range(values.shape[1]):
        roots = tracks.candidate_index[:, i]
        for root in np.unique(roots[roots >= 0]):
            if np.sum(roots == root) > 1:
                values[roots == root, i] = np.nan
        valid = np.flatnonzero(np.isfinite(tracks.time_s[:, i]))
        for a, b in zip(valid[:-1], valid[1:]):
            if tracks.time_s[a, i] >= tracks.time_s[b, i]:
                values[[a, b], i] = np.nan

    # 3. Summarise across distinct cycles where min_cycles is met; quantiles describe spread.
    median, lower, upper = [np.full(values.shape[1], np.nan) for _ in range(3)]
    count = np.isfinite(values).sum(axis=0)
    coherencies[~np.isfinite(values)] = np.nan
    sample_counts[~np.isfinite(values)] = 0
    for i in np.flatnonzero(count >= min_cycles):
        v = values[:, i][np.isfinite(values[:, i])]
        median[i], (lower[i], upper[i]) = np.median(v), np.quantile(v, quantiles)
    return SlownessSummary(
        values,
        median,
        lower,
        upper,
        count,
        tuple(quantiles),
        coherencies,
        sample_counts,
        min_cycles,
    )
