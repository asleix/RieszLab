"""Numerical contracts for transform, tensors, units and enhancement.

Copyright (c) 2026 Aleix Seguí Ugalde.
Affiliation: University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import gaussian_filter

from rieszlab import (
    RieszConfig,
    GaussianTensorConfig,
    enhance_coherent,
    reconstruct,
    monogenic,
    structure_tensor,
)
from rieszlab.tensors import _smooth_axis


@pytest.mark.parametrize("dtype", [None, "float32"])
def test_default_double_precision_and_single_precision_opt_in(dtype):
    x = np.random.default_rng(18).normal(size=(12, 32)).astype(np.float32)
    kwargs = {} if dtype is None else {"config": RieszConfig(dtype=dtype)}
    result = monogenic(x, 10, .01, tensor=GaussianTensorConfig(), **kwargs)
    expected = np.dtype("float64" if dtype is None else dtype)
    m = result.monogenic
    arrays = (m.even, m.riesz_space, m.riesz_time, result.amplitude,
              result.phase_directional, result.slowness_s_per_m,
              result.tensor.j_ss, result.coherency, enhance_coherent(result).data)
    assert RieszConfig().dtype == "float64"
    assert m.config.dtype == expected.name
    assert all(a.dtype == expected for a in arrays)
    assert x.dtype == np.float32


@pytest.mark.parametrize("metric", [200.0, 1000.0, 6000.0])
@pytest.mark.parametrize("spatial_mode", [-3, 3])
def test_plane_wave_sign_amplitude_and_metric(metric, spatial_mode):
    ns, nt, dx, dt = 64, 256, 10.0, 0.01
    k, f = spatial_mode / (ns * dx), -12 / (nt * dt)
    phase = 2 * np.pi * (k * np.arange(ns)[:, None] * dx + f * np.arange(nt)[None, :] * dt) + 0.17
    result = monogenic(
        2 * np.cos(phase),
        dx,
        dt,
        config=RieszConfig(metric, (0, 0), dtype="float64"),
        tensor=GaussianTensorConfig(20, 0.04),
    )
    assert_allclose(result.amplitude, 2, atol=3e-14)
    assert_allclose(result.slowness_s_per_m, -k / f, atol=3e-17)
    assert_allclose(result.slowness_s_per_km, -1000 * k / f, atol=3e-14)
    assert_allclose(result.coherency, 1, atol=5e-15)
    assert_allclose(result.velocity_mps(), -f / k, rtol=2e-13)
    m = result.monogenic
    radius = np.hypot(k, f / metric)
    assert_allclose(m.riesz_space, 2 * np.sin(phase) * k / radius, atol=4e-14)
    assert_allclose(m.riesz_time, 2 * np.sin(phase) * (f / metric) / radius, atol=4e-14)


def test_tensor_matches_independent_eigendecomposition():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(2, 15, 29))
    cfg = GaussianTensorConfig(20, 0.03, tau=0.7)
    out = structure_tensor(a, b, 10, 0.01, config=cfg)
    j = np.empty((*a.shape, 2, 2))
    j[..., 0, 0] = gaussian_filter(a * a, (2, 3))
    j[..., 1, 1] = gaussian_filter(b * b, (2, 3))
    j[..., 0, 1] = j[..., 1, 0] = gaussian_filter(a * b, (2, 3))
    eigenvalues, eigenvectors = np.linalg.eigh(j)
    assert_allclose(out.lambda_max, eigenvalues[..., 1], atol=1e-15)
    assert_allclose(out.lambda_min, eigenvalues[..., 0], atol=1e-15)
    normal = np.stack((np.cos(out.orientation), np.sin(out.orientation)), axis=-1)
    assert_allclose(abs(np.sum(normal * eigenvectors[..., :, 1], axis=-1)), 1, atol=1e-14)
    isotropic = np.linalg.eigvalsh(j + 0.35 * np.eye(2))
    expected = np.diff(isotropic, axis=-1)[..., 0] / isotropic.sum(axis=-1)
    assert_allclose(out.coherency_regularized, expected, atol=1e-15)


def test_exact_variable_gaussian_and_constant_profile():
    data = np.arange(9 * 13, dtype=float).reshape(9, 13) ** 2
    widths = np.array([0, 0.1, 1, 2, 1.3, 3, 2, 1, 0])
    out = _smooth_axis(data, widths, 0, 4)
    for i, w in enumerate(widths):
        radius = int(4 * w + 0.5)
        lag = np.arange(-radius, radius + 1)
        weight = np.exp(-0.5 * (lag / w) ** 2) if w else np.ones(1)
        weight /= weight.sum()
        padded = np.pad(data, ((radius, radius), (0, 0)), mode="symmetric")
        assert_allclose(out[i], weight @ padded[i : i + 2 * radius + 1], rtol=1e-15)
    assert_allclose(_smooth_axis(data, np.full(9, 2.0), 0, 4), _smooth_axis(data, 2.0, 0, 4))


def test_tau_scales_with_amplitude_squared_and_zero_data():
    x = np.random.default_rng(10).normal(size=(32, 81))
    config = RieszConfig(pad_samples=(3, 7), dtype="float64")
    tensor = GaussianTensorConfig(10, 0.02, tau=0.5)
    original = monogenic(x, 10, 0.01, config=config, tensor=tensor)
    scaled = monogenic(4 * x, 10, 0.01, config=config, tensor=replace(tensor, tau=16 * tensor.tau))
    assert_allclose(scaled.amplitude, 4 * original.amplitude)
    assert_allclose(scaled.coherency, original.coherency)
    assert_allclose(scaled.slowness_s_per_m, original.slowness_s_per_m)
    zero = monogenic(np.zeros((8, 21)), 10, 0.01, config=config, tensor=tensor)
    assert not zero.valid.any()
    assert_allclose(zero.coherency, 0)
    assert_allclose(zero.tensor.coherency, 0)
    assert np.isnan(zero.phase_directional).all()


@pytest.mark.parametrize("shape", [(17, 31), (18, 32)])
@pytest.mark.parametrize("pad_mode", ["reflect", "constant", "edge"])
def test_enhancement_unity_and_constant_weights(shape, pad_mode):
    x = np.random.default_rng(4).normal(size=shape) + 2
    result = monogenic(
        x,
        10,
        0.01,
        config=RieszConfig(pad_samples=(5, 8), pad_mode=pad_mode, dtype="float64"),
        tensor=GaussianTensorConfig(),
    )
    for weight in (0.0, 0.4, 1.0):
        reconstructed = reconstruct(result.monogenic, weights=weight)
        assert_allclose(reconstructed.data, weight * x, atol=4e-15)
    assert_allclose(enhance_coherent(result).weights, result.coherency)


def test_custom_weights_match_explicit_least_squares():
    shape = (5, 6)
    config = RieszConfig(pad_samples=(0, 0))
    x = np.random.default_rng(20).normal(size=shape)
    result = monogenic(x, 10, 0.01, config=config)
    weights = np.linspace(0, 1, x.size).reshape(shape)

    # Build the analysis matrix independently of the FFT synthesis.
    columns = []
    for impulse in np.eye(x.size):
        mono = monogenic(impulse.reshape(shape), 10, 0.01, config=config).monogenic
        columns.append(np.concatenate([
            mono.even.ravel(), mono.riesz_space.ravel(), mono.riesz_time.ravel(),
        ]))
    analysis = np.column_stack(columns)
    target = np.tile(weights.ravel(), 3) * (analysis @ x.ravel())
    expected = np.linalg.lstsq(analysis, target, rcond=None)[0].reshape(shape)
    enhanced = reconstruct(result.monogenic, weights=weights)
    assert_allclose(enhanced.data, expected, atol=5e-15)
    assert_allclose(enhanced.weights, weights)
    assert_allclose(result.monogenic.even, x)


def test_custom_weights_broadcast_and_validation():
    x = np.random.default_rng(21).normal(size=(7, 13))
    result = monogenic(x, 10, 0.01, config=RieszConfig(pad_samples=(3, 4)))
    weights = np.linspace(0, 1, x.shape[0])[:, None]
    full = np.broadcast_to(weights, x.shape)
    enhanced = reconstruct(result.monogenic, weights=weights)
    assert_allclose(enhanced.data, reconstruct(result.monogenic, full).data)
    assert_allclose(enhanced.weights, full)
    for invalid in (-0.1, 1.1, np.nan, np.inf, np.ones((2, 3))):
        with pytest.raises(ValueError):
            reconstruct(result.monogenic, weights=invalid)


def test_estimates_are_unmasked_and_inputs_are_unchanged():
    x = np.random.default_rng(2).normal(size=(12, 32)).astype(np.float32)
    before = x.copy()
    result = monogenic(x, 10, 0.01, tensor=GaussianTensorConfig())
    assert np.any(result.valid & (result.coherency < 0.7))
    assert np.isfinite(result.slowness_s_per_m[result.valid]).all()
    assert_allclose(x, before)
    with pytest.raises(ValueError):
        monogenic(x + np.nan, 10, 0.01)
    with pytest.raises(ValueError):
        monogenic(x, 10, 0.01, tensor=GaussianTensorConfig(np.ones(3), 0.1))


def test_core_and_plotting_module_do_not_require_optional_dependencies():
    code = """
import sys
sys.path.insert(0, sys.argv[1])
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'h5py', 'cupy'}:
            raise ImportError('blocked optional dependency')
sys.meta_path.insert(0, BlockOptional())
import rieszlab
import rieszlab.plotting
import rieszlab.io
import numpy as np
r = rieszlab.monogenic(np.zeros((4, 8)), 10, .01)
assert not r.valid.any()
"""
    subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve().parents[1])], check=True
    )
