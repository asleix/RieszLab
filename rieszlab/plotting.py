"""Optional plotting helpers. Display crops never change the numerical analysis.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import TYPE_CHECKING

import numpy as np

from .association import PhaseTracks, SlownessSummary
from .enhancement import EnhancementResult
from .geometry import AdaptiveWidths, CableGeometry
from .inversion import BinnedSlowness, DirectionInterval
from .riesz import RieszResult, METRES_PER_KILOMETRE as KM

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class PlotConfig:
    accent: str = "#006D77"
    contrast: str = "#C96F4F"
    font_size: float = 9.0
    font_family: str = "DejaVu Serif"
    interpolation: str = "none"
    strain_cmap: str = "seismic"
    slowness_cmap: str = "RdBu_r"
    phase_cmap: str = "twilight"
    association_color: str = "#FFB000"
    amplitude_percentile: float = 99.0
    strain_limits: tuple[float, float] | None = None
    strain_percentiles: tuple[float, float] = (25.0, 75.0)
    slowness_limits_s_per_km: tuple[float, float] = (-1.5, 1.5)
    coherency_min: float = 0.3
    distance_range_m: tuple[float, float] | None = None
    time_range_s: tuple[float, float] | None = None
    max_pixels: int = 1_000_000

    def __post_init__(self) -> None:
        if not 0 < self.amplitude_percentile <= 100 or not 0 <= self.coherency_min <= 1:
            raise ValueError("Invalid percentile or coherency threshold")
        if not np.isfinite(self.font_size) or self.font_size <= 0:
            raise ValueError("font_size must be positive")
        if not isinstance(self.max_pixels, int) or self.max_pixels < 4:
            raise ValueError("max_pixels must be an integer >=4")
        for limits in (
            self.strain_limits,
            self.strain_percentiles,
            self.slowness_limits_s_per_km,
            self.distance_range_m,
            self.time_range_s,
        ):
            if limits is not None and (
                len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]
            ):
                raise ValueError("Plot limits must be increasing finite pairs")
        if not 0 <= self.strain_percentiles[0] < self.strain_percentiles[1] <= 100:
            raise ValueError("strain_percentiles must lie in [0,100]")


def _mpl(config: PlotConfig):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:
        raise ImportError("Install rieszlab[plot] to make figures") from exc
    cmap = LinearSegmentedColormap.from_list("coherency", ["white", config.accent])
    return (
        plt,
        cmap,
        {
            "font.size": config.font_size,
            "font.family": config.font_family,
            "mathtext.fontset": "stix",
            "legend.framealpha": 1,
            "legend.fancybox": False,
        },
    )


def _view(result: RieszResult, cfg: PlotConfig, distance_start_m: float, time_start_s: float):
    """Bound the displayed pixels before creating derived plotting arrays."""
    m = result.monogenic
    slices, extent = [], []
    for n, step, start, limits in zip(
        m.even.shape,
        (m.dx_m, m.dt_s),
        (distance_start_m, time_start_s),
        (cfg.distance_range_m, cfg.time_range_s),
    ):
        first, last = 0, n
        if limits is not None:
            first = max(0, int(np.ceil((limits[0] - start) / step)))
            last = min(n, int(np.floor((limits[1] - start) / step)) + 1)
        if last <= first:
            raise ValueError("Display window does not intersect the record")
        slices.append(slice(first, last))
        extent += [start + (first - 0.5) * step, start + (last - 0.5) * step]
    size = np.prod([s.stop - s.start for s in slices])
    stride = max(1, int(np.ceil(np.sqrt(size / cfg.max_pixels))))
    # Display thinning only; no averaging or resampling of numerical results.
    slices = tuple(slice(s.start, s.stop, stride) for s in slices)
    extent = [extent[2], extent[3], extent[0] / KM, extent[1] / KM]
    return slices, extent


def _strain_limits(data: np.ndarray, cfg: PlotConfig) -> tuple[float, float]:
    lo, hi = cfg.strain_limits or np.percentile(data, cfg.strain_percentiles)
    return (lo, hi) if hi > lo else (lo - 1, hi + 1)


def _image(ax, data, extent, cfg, cmap, limits, *, vertical=False):
    if vertical:
        data, extent = data.T, [extent[2], extent[3], extent[1], extent[0]]
    return ax.imshow(
        data,
        extent=extent,
        origin="upper" if vertical else "lower",
        aspect="auto",
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        interpolation=cfg.interpolation,
        rasterized=True,
    )


def plot_attributes(
    result: RieszResult,
    *,
    config: PlotConfig | None = None,
    distance_start_m: float = 0.0,
    time_start_s: float = 0.0,
    units: str = "input units",
    figsize=(7, 6),
) -> Figure:
    """Six primary/derived maps; missing tensor coherency is explicitly labelled."""
    cfg = config or PlotConfig()
    plt, cmap, style = _mpl(cfg)
    view, extent = _view(result, cfg, distance_start_m, time_start_s)
    m = result.monogenic
    u, a = m.even[view], m.amplitude[view]
    chi = None if result.coherence is None else result.coherence[view]
    keep = result.valid[view] & (True if chi is None else chi >= cfg.coherency_min)
    fields = [
        (u, "Strain", cfg.strain_cmap, _strain_limits(u, cfg), units),
        (
            a,
            "Monogenic amplitude",
            "gray",
            (0, max(np.percentile(a, cfg.amplitude_percentile), 1e-30)),
            units,
        ),
        (
            result.phase_directional[view],
            "Directional phase",
            cfg.phase_cmap,
            (-np.pi, np.pi),
            "rad",
        ),
        (
            np.degrees(result.orientation[view]),
            "Instantaneous orientation" if chi is None else "Tensor orientation",
            cfg.phase_cmap,
            (0, 180),
            "degrees",
        ),
        (chi, "Directional coherency", cmap, (0, 1), r"$\mathcal{X}$"),
        (
            np.where(keep, KM * result.slowness_s_per_m[view], np.nan),
            "Apparent slowness",
            cfg.slowness_cmap,
            cfg.slowness_limits_s_per_km,
            r"$p_{\rm app}$ (s/km)",
        ),
    ]
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            2, 3, figsize=figsize, sharex=True, sharey=True, layout="constrained"
        )
        for j, (ax, (data, title, colors, limits, label)) in enumerate(zip(axes.flat, fields)):
            ax.set_title(f"({chr(97+j)}) {title}", loc="left", fontsize=cfg.font_size)
            if data is None:
                ax.text(0.5, 0.5, "No tensor computed", ha="center", transform=ax.transAxes)
            else:
                im = _image(ax, data, extent, cfg, colors, limits)
                fig.colorbar(im, ax=ax, label=label, fraction=0.045, pad=0.02)
            ax.set_xlabel("Time (s)")
            if j % 3 == 0:
                ax.set_ylabel("Cable distance (km)")
    return fig


def plot_enhancement(
    result: RieszResult,
    enhanced: EnhancementResult,
    *,
    config: PlotConfig | None = None,
    distance_start_m: float = 0.0,
    time_start_s: float = 0.0,
    units: str = "input units",
    figsize=(7, 4.5),
) -> Figure:
    """Original/coherency/enhanced maps, with shared strain limits."""
    if result.coherence is None:
        raise ValueError("This figure requires tensor coherency")
    cfg = config or PlotConfig()
    plt, cmap, style = _mpl(cfg)
    view, extent = _view(result, cfg, distance_start_m, time_start_s)
    limits = _strain_limits(result.monogenic.even[view], cfg)
    fields = [
        (result.monogenic.even, "Strain", cfg.strain_cmap, limits, units),
        (result.coherence, "Directional coherency", cmap, (0, 1), r"$\mathcal{X}$"),
        (enhanced.data, "Coherent enhancement", cfg.strain_cmap, limits, units),
    ]
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            1, 3, figsize=figsize, sharex=True, sharey=True, layout="constrained"
        )
        for j, (ax, (data, title, colors, bounds, label)) in enumerate(zip(axes, fields)):
            im = _image(ax, data[view], extent, cfg, colors, bounds, vertical=True)
            ax.set_title(f"({chr(97+j)}) {title}", loc="left", fontsize=cfg.font_size)
            ax.set_xlabel("Cable distance (km)")
            fig.colorbar(im, ax=ax, label=label, orientation="horizontal", fraction=0.04, pad=0.08)
        axes[0].set_ylabel("Time (s)")
    return fig


def plot_association(
    result: RieszResult,
    tracks: PhaseTracks,
    *,
    summary: SlownessSummary | None = None,
    config: PlotConfig | None = None,
    figsize=(7, 5),
) -> Figure:
    """Equal-phase crossings and per-cycle slowness (local medians if supplied)."""
    cfg = config or PlotConfig()
    plt, cmap, style = _mpl(cfg)
    view, extent = _view(result, cfg, tracks.distance_m[0], tracks.time_start_s)
    with plt.rc_context(style):
        fig = plt.figure(figsize=figsize, layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=[2, 1])
        axes = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :]),
        ]
        for ax, data, colors, limits, title in zip(
            axes[:2],
            [result.amplitude[view], result.phase_directional[view]],
            ["gray", cfg.phase_cmap],
            [
                (0, max(np.percentile(result.amplitude[view], cfg.amplitude_percentile), 1e-30)),
                (-np.pi, np.pi),
            ],
            ["Monogenic amplitude", "Directional phase"],
        ):
            _image(ax, data, extent, cfg, colors, limits)
            for track in tracks.time_s:
                ax.scatter(
                    track,
                    tracks.distance_m / KM,
                    s=3,
                    color=cfg.association_color,
                    edgecolors="none",
                )
            ax.set(
                xlabel="Time (s)",
                ylabel="Cable distance (km)",
                title=title,
                xlim=extent[:2],
                ylim=extent[2:],
            )
        p = tracks.slowness_s_per_m if summary is None else summary.per_cycle_s_per_m
        chi = tracks.coherency if summary is None else summary.per_cycle_coherence
        points = axes[2].scatter(
            np.broadcast_to(tracks.distance_m / KM, p.shape),
            p * KM,
            c=chi,
            s=6,
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        axes[2].set(
            xlabel="Cable distance (km)",
            ylabel=r"$p_{\rm app}$ (s/km)",
            ylim=cfg.slowness_limits_s_per_km,
            xlim=extent[2:],
        )
        fig.colorbar(points, ax=axes[2], label=r"$\mathcal{X}$", fraction=0.025)
    return fig


def plot_profiles(
    cable: CableGeometry,
    widths: AdaptiveWidths,
    summaries: dict[str, SlownessSummary],
    *,
    config: PlotConfig | None = None,
    figsize=(7, 4),
) -> Figure:
    """Cable geometry/width and median–quantile profiles (not confidence intervals)."""
    cfg = config or PlotConfig()
    plt, _, style = _mpl(cfg)
    with plt.rc_context(style):
        fig, (top, bottom) = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [1, 2]},
            layout="constrained",
        )
        distance = cable.distance_m / KM
        top.plot(distance, cable.azimuth_degrees, color=".25", lw=0.8)
        top.set_ylabel("Cable azimuth (°)")
        right = top.twinx()
        right.plot(distance, widths.sigma_m, color=cfg.accent, lw=0.8)
        right.set_ylabel(r"Tensor width $\sigma$ (m)")
        for color, (label, summary) in zip(
            cycle([cfg.accent, cfg.contrast, ".4"]), summaries.items()
        ):
            bottom.plot(distance, KM * summary.median_s_per_m, color=color, lw=1, label=label)
            bottom.fill_between(
                distance,
                KM * summary.lower_s_per_m,
                KM * summary.upper_s_per_m,
                color=color,
                alpha=0.18,
            )
        bottom.set(
            xlabel="Cable distance (km)",
            ylabel="Apparent slowness (s/km)",
            ylim=cfg.slowness_limits_s_per_km,
        )
        bottom.legend()
    return fig


def plot_direction_fit(
    bins: BinnedSlowness,
    interval: DirectionInterval,
    *,
    config: PlotConfig | None = None,
    figsize=(7, 2.6),
) -> Figure:
    """Projection fit and conditional bootstrap bearing distribution."""
    cfg = config or PlotConfig()
    plt, _, style = _mpl(cfg)
    fit = interval.fit
    with plt.rc_context(style):
        fig, (left, right) = plt.subplots(
            1, 2, figsize=figsize, layout="constrained", gridspec_kw={"width_ratios": [2, 1]}
        )
        left.scatter(
            bins.distance_m / KM,
            bins.slowness_s_per_m * KM,
            s=10,
            color=cfg.accent,
            label="Bin medians",
        )
        left.plot(
            bins.distance_m / KM,
            fit.predicted_s_per_m * KM,
            color=cfg.contrast,
            label="Cable projection",
        )
        left.set(xlabel="Cable distance (km)", ylabel=r"$p_{\rm app}$ (s/km)")
        left.legend()
        delta = (interval.backazimuth_samples_degrees - fit.backazimuth_degrees + 180) % 360 - 180
        right.hist(fit.backazimuth_degrees + delta, bins=20, color=cfg.accent)
        for bound in interval.interval_degrees:
            right.axvline(bound, color=cfg.contrast, ls="--", lw=1)
        right.axvline(fit.backazimuth_degrees, color=".2", lw=1)
        right.set(xlabel="Incoming backazimuth (°)", ylabel="Bootstrap fits")
    return fig


def plot_direction_selection(
    result: RieszResult,
    first: EnhancementResult,
    second: EnhancementResult,
    *,
    config: PlotConfig | None = None,
    distance_start_m: float = 0.0,
    time_start_s: float = 0.0,
    units: str = "input units",
    figsize=(7, 3.2),
) -> Figure:
    """Signed reconstructions plus a positive-lobe colour overlay for comparison."""
    cfg = config or PlotConfig()
    plt, _, style = _mpl(cfg)
    from matplotlib.colors import to_rgb

    view, extent = _view(result, cfg, distance_start_m, time_start_s)
    strain = result.monogenic.even[view]
    limits = _strain_limits(strain, cfg)
    scale = max(abs(limits[0]), abs(limits[1]))
    rgb = np.ones((*strain.shape, 3))
    for enhanced, color in ((first, cfg.accent), (second, cfg.contrast)):
        strength = np.clip(enhanced.data[view] / scale, 0, 1)
        rgb *= 1 - strength[..., None] * (1 - np.asarray(to_rgb(color)))
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            1, 4, figsize=figsize, sharex=True, sharey=True, layout="constrained"
        )
        for j, (ax, data) in enumerate(
            zip(axes[:3], (strain, first.data[view], second.data[view]))
        ):
            im = _image(ax, data, extent, cfg, cfg.strain_cmap, limits, vertical=True)
            title = (
                "Strain"
                if j == 0
                else f"Selected {KM*(first, second)[j-1].selection.slowness_s_per_m:+.2f} s/km"
            )
            ax.set_title(title, fontsize=cfg.font_size)
            fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.08, label=units)
        axes[3].imshow(
            rgb.transpose(1, 0, 2),
            extent=[extent[2], extent[3], extent[1], extent[0]],
            origin="upper",
            aspect="auto",
            interpolation=cfg.interpolation,
            rasterized=True,
        )
        axes[3].set_title("Positive-lobe overlay", fontsize=cfg.font_size)
        for ax in axes:
            ax.set_xlabel("Cable distance (km)")
        axes[0].set_ylabel("Time (s)")
    return fig
