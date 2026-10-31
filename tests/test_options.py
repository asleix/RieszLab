"""Optional tensors, steering, cycle selection and plotting contracts.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import convolve2d

from rieszlab import (
    DirectionSelectionConfig,
    GaussianTensorConfig,
    RieszConfig,
    SteeredTensorConfig,
    TensorConfig,
    direction_weight,
    enhance_coherent,
    enhance_directional,
    reconstruct,
    monogenic,
)
from rieszlab.association import (
    associate_cycles,
    phase_candidates,
    select_cycles,
    summarize_tracks,
)
from rieszlab.inversion import FitConfig, bin_slowness, bootstrap_direction


def plane(tensor=None):
    ns, nt = 48, 240
    phi = 2 * np.pi * (2 * np.arange(ns)[:, None] / ns - 12 * np.arange(nt)[None, :] / nt) + 0.123
    return monogenic(
        np.cos(phi),
        10,
        0.01,
        tensor=tensor,
        config=RieszConfig(pad_samples=(0, 0), dtype="float64"),
    )


def test_optional_tensor_and_explicit_sampling():
    result = plane()
    assert result.tensor is None and result.coherence is None
    assert result.monogenic.config.reference_velocity_mps == 1000
    assert_allclose(result.amplitude, 1, atol=2e-14)
    assert_allclose(result.slowness_s_per_m, (2 / 480) / 5, atol=1e-15)
    assert_allclose(reconstruct(result.monogenic, 1).data, result.monogenic.even, atol=1e-14)
    x = result.monogenic.even
    with pytest.raises(ValueError, match="Supply dx_m and dt_s"):
        monogenic(x)
    with pytest.raises(ValueError, match="sampling grid"):
        monogenic(x, reference_velocity_mps=1000)
    inferred = monogenic(x, 12, 0.02)
    assert inferred.monogenic.config.reference_velocity_mps == 600
    assert (
        monogenic(x, 12, 0.02, reference_velocity_mps=2500).monogenic.config.reference_velocity_mps
        == 2500
    )
    with pytest.raises(ValueError, match="Conflicting"):
        monogenic(x, 12, 0.02, reference_velocity_mps=2500, config=RieszConfig(600))
    for operation in (enhance_coherent, phase_candidates):
        with pytest.raises(ValueError, match="tensor"):
            operation(result)
    with pytest.raises(ValueError, match="tensor"):
        enhance_directional(result, selection=DirectionSelectionConfig(0.0005))


def test_tensor_reuses_monogenic_coefficients():
    original = plane()
    result = original.with_tensor(GaussianTensorConfig(20, 0.04, tau=0.01))
    assert result.monogenic is original.monogenic
    assert result.coherence is not None
    assert_allclose(result.slowness_s_per_m, original.slowness_s_per_m, atol=1e-15)
    instant = result.with_tensor(None)
    assert instant.coherence is None and instant.monogenic is result.monogenic


@pytest.mark.parametrize(
    "tensor",
    [
        None,
        GaussianTensorConfig(20, 0.04, tau=0.02),
        GaussianTensorConfig(np.linspace(10, 40, 24), 0.04, tau=0.02),
        SteeredTensorConfig(0.0005, 60, 20, tau=0.02),
    ],
)
def test_with_tensor_matches_fresh_analysis_without_recomputing(monkeypatch, tensor):
    x = np.random.default_rng(19).normal(size=(24, 80))
    config = RieszConfig(750, (5, 9), dtype="float64")
    original = monogenic(x, 10, 0.01, config=config, tensor=GaussianTensorConfig())
    expected = monogenic(x, 10, 0.01, config=config, tensor=tensor)
    old_tensor = original.tensor
    snapshots = [
        a.copy() for a in (
            original.orientation, original.slowness_s_per_m,
            original.phase_directional, original.coherence,
        )
    ]

    def no_transform(*args, **kwargs):
        pytest.fail("with_tensor must not repeat the Riesz transform")

    module = import_module("rieszlab.riesz")
    monkeypatch.setattr(module, "_monogenic_signal", no_transform)
    monkeypatch.setattr(module.fft, "rfft2", no_transform)
    actual = original.with_tensor(tensor)
    assert actual is not original and actual.monogenic is original.monogenic
    assert original.tensor is old_tensor
    for before, after in zip(
        snapshots,
        (original.orientation, original.slowness_s_per_m,
         original.phase_directional, original.coherence),
    ):
        assert_allclose(after, before, rtol=0, atol=0)
    for name in ("amplitude", "orientation", "slowness_s_per_m", "phase_directional", "valid"):
        assert_allclose(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    if tensor is None:
        assert actual.coherence is None and actual.tensor is None
    else:
        assert_allclose(actual.coherence, expected.coherence, rtol=0, atol=0)


def test_single_public_entry_point():
    import rieszlab as dm

    assert callable(dm.monogenic) and "monogenic" in dm.__all__
    assert not callable(dm.riesz)  # Internal module, not a second entry point.
    assert "riesz" not in dm.__all__
    assert not hasattr(dm, "analyse_monogenic")
    assert not hasattr(dm, "monogenic_signal")


def test_steered_kernel_matches_direct_linear_convolution():
    cfg = SteeredTensorConfig(0.0005, 45, 20, truncate=3)
    assert isinstance(cfg, TensorConfig)
    kernel = cfg.kernel(10, 0.01, 1000)
    assert_allclose(kernel.sum(), 1)
    assert (kernel >= 0).all()
    assert_allclose(kernel, kernel[::-1, ::-1])
    assert_allclose(kernel, replace(cfg, slowness_s_per_m=-0.0005).kernel(10, 0.01, 1000)[:, ::-1])
    x = np.random.default_rng(19).normal(size=(19, 37))
    padded = np.pad(x, [(n // 2, n // 2) for n in kernel.shape], mode="symmetric")
    expected = convolve2d(padded, kernel, mode="valid")
    assert_allclose(cfg.average(x, 10, 0.01, 1000), expected, atol=6e-16)


def test_direction_selection_plane_wave_and_axial_symmetry():
    p = (2 / 480) / 5
    selection = DirectionSelectionConfig(p, angular_sigma_degrees=5)
    result = plane(SteeredTensorConfig(p, 60, 20))
    assert_allclose(result.slowness_s_per_m, p, atol=1e-17)
    assert_allclose(
        direction_weight(result.orientation + np.pi, 1000, selection=selection), 1, atol=1e-14
    )
    matching = enhance_directional(result, selection=selection)
    opposite = enhance_directional(result, selection=replace(selection, slowness_s_per_m=-p))
    assert_allclose(matching.data, result.monogenic.even, atol=1e-14)
    assert np.max(np.abs(opposite.data)) < 1e-8
    assert_allclose(matching.weights, result.coherence, atol=1e-14)


def test_cycle_selection_minimum_and_nonconsecutive():
    result = plane(GaussianTensorConfig(20, 0.03))
    candidates = phase_candidates(result, distance_start_m=1000, time_start_s=10)
    tracks = associate_cycles(candidates, 24, time_window_s=(10.6, 11.8))
    score = np.arange(len(tracks.seed_indices), dtype=float)
    score[1] = 100
    selected = select_cycles(tracks, score, count=2, consecutive=False)
    assert_allclose(selected.seed_indices, tracks.seed_indices[[1, len(score) - 1]])
    assert selected.time_start_s == 10
    summary = summarize_tracks(result, selected, min_cycles=2)
    assert np.isfinite(summary.median_s_per_m).all()
    assert summary.min_cycles == 2 and (summary.count == 2).all()
    too_few = summarize_tracks(result, selected, min_cycles=3)
    assert np.isnan(too_few.median_s_per_m).all()
    gapped = replace(tracks, seed_indices=2 * tracks.seed_indices)
    with pytest.raises(ValueError, match="consecutive group"):
        select_cycles(gapped, score, count=2)


def test_bootstrap_does_not_bridge_missing_bin():
    s = np.array([240.0, 510.0, 760.0, 1010.0])
    angle = np.linspace(-1, 1, len(s))
    tangent = np.column_stack((np.sin(angle), np.cos(angle)))
    bins = bin_slowness(s, tangent @ np.array([0.0003, -0.0004]), tangent, min_points=1)
    assert_allclose(bins.bin_ids, [0, 2, 3, 4])
    with pytest.raises(ValueError, match="Too few contiguous blocks"):
        bootstrap_direction(bins, config=FitConfig(loss="mse"), block_length_m=750, min_bins=3)


def test_plot_helpers_optional_tensor_and_small_views(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rieszlab.plotting import (
        PlotConfig,
        plot_association,
        plot_attributes,
        plot_direction_selection,
        plot_enhancement,
    )

    result = plane(GaussianTensorConfig(20, 0.03))
    cfg = PlotConfig(time_range_s=(0.5, 1.5), distance_range_m=(100, 300), max_pixels=400)
    tracks = associate_cycles(phase_candidates(result), 24, time_window_s=(0.6, 1.2))
    a = enhance_directional(result, selection=DirectionSelectionConfig(0.0005))
    b = enhance_directional(result, selection=DirectionSelectionConfig(-0.0005))
    figures = [
        plot_attributes(plane(), config=cfg),
        plot_enhancement(result, enhance_coherent(result), config=cfg),
        plot_association(result, tracks, summary=summarize_tracks(result, tracks), config=cfg),
        plot_direction_selection(result, a, b, config=cfg),
    ]
    for i, fig in enumerate(figures):
        fig.savefig(tmp_path / f"{i}.png", dpi=40)
        for ax in fig.axes:
            for im in ax.images:
                assert np.prod(im.get_array().shape[:2]) <= cfg.max_pixels
        plt.close(fig)
