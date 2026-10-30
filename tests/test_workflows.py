"""Geometry, association, preprocessing and inversion checks on known fields.

Copyright (c) 2026 Aleix Seguí Ugalde.
Affiliation: University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rieszlab import RieszConfig, GaussianTensorConfig, monogenic
from rieszlab.association import (
    AssociationConfig,
    associate_cycles,
    cycle_power,
    select_cycles,
    phase_candidates,
    summarize_tracks,
)
from rieszlab.geometry import GeometryConfig, adaptive_tensor_widths, cable_geometry
from rieszlab.inversion import FitConfig, bin_slowness, bootstrap_direction, fit_propagation
from rieszlab.preprocessing import (
    butterworth,
    interpolate_bad_channels,
    remove_common_mode,
    ricker_space,
    ricker_time,
)


def test_common_mode_does_not_normalise_channels():
    t = np.arange(200) * 0.01
    common = 0.3 * np.cos(2 * np.pi * 4 * t)
    wave = np.sin(2 * np.pi * 7 * t)
    gains = np.array([-3, -1, 1, 3])
    x = common + gains[:, None] * wave
    result = remove_common_mode(x, energy_mad_multiplier=None)
    assert_allclose(result.template, common, atol=3e-16)
    assert_allclose(result.data, gains[:, None] * wave, atol=1e-15)
    assert_allclose(result.gains, 1)
    filtered = butterworth(x, 0.01, band_hz=(3, 20))
    assert_allclose(butterworth(5 * x, 0.01, band_hz=(3, 20)), 5 * filtered, atol=3e-13)


def test_common_mode_rejection_is_not_channel_repair():
    x = np.ones((6, 20))
    x[-1] = 100
    result = remove_common_mode(x)
    assert not result.selected_channels[-1]
    assert_allclose(result.data[-1], 99)
    bad = np.zeros(6, bool)
    bad[2:4] = True
    x = np.arange(6, dtype=float)[:, None] * np.ones((1, 20))
    x[bad] = np.nan
    repaired = interpolate_bad_channels(x, bad)
    assert_allclose(repaired, np.arange(6)[:, None] * np.ones((1, 20)))
    bad[0] = True
    with pytest.raises(ValueError):
        interpolate_bad_channels(x, bad)


def test_ricker_peak_gain_and_no_velocity_rescaling():
    ns, nt, dx, dt = 100, 500, 10, 0.01
    k, f = 4 / (ns * dx), 25 / (nt * dt)
    x = np.cos(2 * np.pi * (k * np.arange(ns)[:, None] * dx - f * np.arange(nt)[None, :] * dt))
    for y in [
        ricker_time(x, dt, peak_frequency_hz=f, pad_samples=0),
        ricker_space(x, dx, peak_wavelength_m=1 / k, pad_samples=0),
    ]:
        assert_allclose(y, x, atol=3e-14)
        result = monogenic(
            y,
            dx,
            dt,
            config=RieszConfig(pad_samples=(0, 0), dtype="float64"),
            tensor=GaussianTensorConfig(20, 0.03),
        )
        assert_allclose(result.slowness_s_per_m, k / f, atol=5e-17)


def test_geometry_straight_and_circular_limits_in_3d():
    s = np.arange(401) * 10.0
    radius = 1000.0
    cfg = GeometryConfig(smoothing_fwhm_m=0, max_sigma_m=100, max_turn_degrees=12)
    straight = cable_geometry(s, np.column_stack((s, 0 * s, 0.2 * s)), smoothing_fwhm_m=0)
    widths = adaptive_tensor_widths(straight, config=cfg)
    assert_allclose(widths.sigma_m[100:-100], 100)
    assert widths.sigma_m[0] == widths.sigma_m[-1] == 0
    curve = cable_geometry(
        s,
        np.column_stack((radius * np.sin(s / radius), 0 * s, radius * (1 - np.cos(s / radius)))),
        smoothing_fwhm_m=0,
    )
    curved = adaptive_tensor_widths(curve, config=cfg)
    expected = np.deg2rad(12) * radius / 3
    assert_allclose(curved.sigma_m[100:-100], expected, rtol=2e-12)
    assert np.max(curved.max_turn_degrees) <= 12 + 1e-12


def _plane_result():
    ns, nt, dx, dt = 64, 600, 10.0, 0.01
    x = np.cos(
        2 * np.pi * (2 * np.arange(ns)[:, None] / ns - 24 * np.arange(nt)[None, :] / nt) + 0.123
    )
    return monogenic(
        x,
        dx,
        dt,
        config=RieszConfig(pad_samples=(0, 0), dtype="float64"),
        tensor=GaussianTensorConfig(20, 0.03, tau=0.001),
    ), (2 / (ns * dx)) / (24 / (nt * dt))


def test_cycle_tracking_and_missing_runs():
    result, p = _plane_result()
    candidates = phase_candidates(result)
    all_tracks = associate_cycles(candidates, 32, time_window_s=(2, 3))
    power = cycle_power(result, all_tracks, aperture_m=(200, 450))
    tracks = select_cycles(all_tracks, power, count=3)
    seeds = tracks.seed_indices
    assert np.isfinite(tracks.time_s).all()
    assert_allclose(np.diff(tracks.time_s, axis=1), 10 * p, atol=5e-5)
    summary = summarize_tracks(result, tracks, coherency_min=0.7)
    assert_allclose(summary.median_s_per_m, p, atol=1e-17)
    assert (summary.count == 3).all()
    for i in range(36, 40):
        candidates.channels[i].coherency[:] = 0
    tracks = associate_cycles(candidates, 32, seeds, config=AssociationConfig(max_gap_m=20))
    assert np.isnan(tracks.time_s[:, 36:]).all()
    assert np.isfinite(tracks.time_s[:, :36]).all()


def test_inversion_projection_sign_and_robust_fit():
    angle = np.linspace(-1.2, 1.2, 80)
    tangent = np.column_stack((0.8 * np.sin(angle), 0.8 * np.cos(angle), np.full(80, 0.6)))
    vector = np.array([0.0003, -0.0004])
    p = tangent[:, :2] @ vector
    fit = fit_propagation(tangent, p, config=FitConfig(loss="mse"))
    assert_allclose(fit.slowness_en_s_per_m, vector, atol=1e-18)
    expected = np.degrees(np.arctan2(-vector[0], -vector[1])) % 360
    assert_allclose(fit.backazimuth_degrees, expected)
    damaged = p.copy()
    damaged[:3] += 0.003
    mse = fit_propagation(tangent, damaged, config=FitConfig(loss="mse"))
    robust = fit_propagation(tangent, damaged, config=FitConfig(loss="huber"))
    assert (
        np.linalg.norm(robust.slowness_en_s_per_m - vector)
        < np.linalg.norm(mse.slowness_en_s_per_m - vector) / 5
    )
    with pytest.raises(ValueError, match="not resolved"):
        fit_propagation(np.tile([1, 0], (80, 1)), p)


def test_block_interval_reproducible_and_conditioned():
    s = np.arange(240) * 10.0
    angle = np.linspace(-1.1, 1.1, len(s))
    tangent = np.column_stack((np.sin(angle), np.cos(angle)))
    p = tangent @ np.array([0.0003, -0.0004]) + np.random.default_rng(1).normal(0, 1e-6, len(s))
    bins = bin_slowness(s, p, tangent, bin_width_m=100, min_points=5)
    cfg = FitConfig(loss="mse")
    a = bootstrap_direction(bins, config=cfg, block_length_m=200, draws=30)
    b = bootstrap_direction(bins, config=cfg, block_length_m=200, draws=30)
    assert_allclose(a.interval_degrees, b.interval_degrees)
    assert a.accepted_draws == 30
    assert a.interval_degrees[1] - a.interval_degrees[0] < 1


def test_hdf5_reader_preserves_units_and_sampling(tmp_path):
    h5py = pytest.importorskip("h5py")
    from rieszlab.io import read_hdf5

    path = tmp_path / "tiny.h5"
    x = np.arange(200, dtype=np.float64).reshape(10, 20) + 1e-10
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("data", data=x)
        ds.attrs.update(dx_m=10, dt_s=0.01, units="strain")
    record = read_hdf5(path, channels=slice(2, 9, 2), time_range_s=(0.02, 0.1), units="nanostrain")
    assert record.data.dtype == np.float64
    assert_allclose(record.data, x[2:9:2, 2:10], rtol=0, atol=0)
    assert record.dx_m == 20 and record.dt_s == 0.01
    assert record.distance_start_m == 20 and record.time_start_s == 0.02
    assert record.units == "nanostrain" and record.metadata["header_units"] == "strain"
    single = read_hdf5(path, dtype="float32")
    assert single.data.dtype == np.float32
    assert_allclose(single.data, x.astype(np.float32), rtol=0, atol=0)
