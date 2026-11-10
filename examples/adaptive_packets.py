"""Geometry-adaptive tensors, associated cycles, profiles and a direction fit.

Copyright (c) 2026 Aleix Seguí Ugalde.
Affiliation: University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

import argparse
from pathlib import Path

import numpy as np

from rieszlab import GaussianTensorConfig, monogenic
from rieszlab.association import (
    AssociationConfig,
    associate_cycles,
    cycle_power,
    select_cycles,
    phase_candidates,
    summarize_tracks,
)
from rieszlab.geometry import GeometryConfig, adaptive_tensor_widths, cable_geometry
from rieszlab.inversion import FitConfig, bin_slowness, bootstrap_direction
from rieszlab.preprocessing import butterworth
from synthetic_data import demo_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("example_output/packets"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw, dx, dt, s, position, truth = demo_data()
    x = butterworth(raw, dt, band_hz=(3, 20), order=2)
    geometry_config = GeometryConfig(smoothing_fwhm_m=20, max_sigma_m=100, max_turn_degrees=20)
    cable = cable_geometry(s, position, smoothing_fwhm_m=geometry_config.smoothing_fwhm_m)
    widths = adaptive_tensor_widths(cable, config=geometry_config)
    result = monogenic(x, dx, dt, tensor=GaussianTensorConfig(widths.sigma_m, 0.09, tau=0.01))
    candidates = phase_candidates(result)
    seed_channel = len(s) // 2
    tracks = associate_cycles(
        candidates, seed_channel, time_window_s=(5, 7), config=AssociationConfig(coherency_min=0.4)
    )
    scores = cycle_power(result, tracks, aperture_m=(950, 1450))
    tracks = select_cycles(tracks, scores, count=3)
    summary = summarize_tracks(result, tracks, coherency_min=0.7)
    # Freeze the selected cycles when comparing averaging widths.
    fixed = result.with_tensor(GaussianTensorConfig(60, 0.09, tau=0.01))
    fixed_summary = summarize_tracks(fixed, tracks, coherency_min=0.7)
    bins = bin_slowness(s, summary.median_s_per_m, cable.tangent_enu, bin_width_m=150, min_points=5)
    interval = bootstrap_direction(bins, config=FitConfig(), block_length_m=300, draws=100)
    expected = np.degrees(np.arctan2(-truth[0], -truth[1])) % 360
    print(
        f"Synthetic backazimuth {expected:.1f}°; {interval.fit.config.loss} fit {interval.fit.backazimuth_degrees:.1f}°"
    )
    print(
        f"Conditional 95% block interval {interval.interval_degrees}; {interval.accepted_draws}/100 fits"
    )
    np.savez_compressed(
        args.output / "packet_estimates.npz",
        distance_m=s,
        median_s_per_m=summary.median_s_per_m,
        sigma_m=widths.sigma_m,
        track_times_s=tracks.time_s,
        backazimuth_samples=interval.backazimuth_samples_degrees,
    )
    if not args.no_plots:
        from rieszlab.plotting import plot_association, plot_profiles
        import matplotlib.pyplot as plt

        figures = [
            ("03_associated_cycles", plot_association(result, tracks, summary=summary)),
            (
                "04_distance_slowness",
                plot_profiles(
                    cable, widths, {"Geometry-adaptive": summary, "Fixed 60 m": fixed_summary}
                ),
            ),
        ]
        for name, fig in figures:
            fig.savefig(args.output / f"{name}.png", dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    main()
