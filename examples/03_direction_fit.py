"""Fit horizontal incoming direction from associated Alboran slowness and geometry.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from pathlib import Path

import numpy as np

from rieszlab import GaussianTensorConfig, RieszConfig, monogenic
from rieszlab.association import (
    AssociationConfig,
    associate_cycles,
    cycle_power,
    phase_candidates,
    select_cycles,
    summarize_tracks,
)
from rieszlab.inversion import FitConfig, bin_slowness, bootstrap_direction
from rieszlab.plotting import PlotConfig, plot_direction_fit
from alboran_io import (
    add_packet_arguments,
    argument_parser,
    load_and_preprocess,
    load_geometry,
    save_figure,
)


def main() -> None:
    p = argument_parser(__doc__)
    add_packet_arguments(p)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--geometry-lonlat", action="store_true")
    p.add_argument("--geometry-fwhm-m", type=float, default=500.0)
    p.add_argument("--fit-distance-m", type=float, nargs=2, default=(21000.0, 26000.0))
    p.add_argument("--loss", choices=("huber", "mse"), default="huber")
    p.add_argument("--draws", type=int, default=400)
    args = p.parse_args()
    record = load_and_preprocess(args)
    result = monogenic(
        record.data,
        dx_m=record.dx_m,
        dt_s=record.dt_s,
        config=RieszConfig(reference_velocity_mps=args.metric_mps),
        tensor=GaussianTensorConfig(
            sigma_space_m=args.sigma_space_m, sigma_time_s=args.sigma_time_s, tau=args.tau
        ),
    )
    # Select and summarise associated cycles before fitting their cable projections.
    if not record.distance_m[0] <= args.reference_m <= record.distance_m[-1]:
        raise ValueError("Reference channel is outside record")
    candidates = phase_candidates(
        result,
        distance_start_m=record.distance_start_m,
        time_start_s=record.time_start_s,
        channel_mask=record.measured_channel,
    )
    reference = int(np.argmin(abs(record.distance_m - args.reference_m)))
    tracks = associate_cycles(
        candidates, reference, time_window_s=tuple(args.search_time),
        config=AssociationConfig(coherency_min=args.association_coherency),
    )
    aperture = (
        args.reference_m - args.power_aperture_m / 2,
        args.reference_m + args.power_aperture_m / 2,
    )
    scores = cycle_power(result, tracks, aperture_m=aperture)
    tracks = select_cycles(tracks, scores, count=args.cycles, consecutive=not args.nonconsecutive)
    summary = summarize_tracks(
        result, tracks, coherency_min=args.summary_coherency, min_cycles=args.min_cycles
    )
    s = record.distance_m
    use = (s >= args.fit_distance_m[0]) & (s <= args.fit_distance_m[1])
    cable = load_geometry(
        args.geometry, s[use], lonlat=args.geometry_lonlat, smoothing_fwhm_m=args.geometry_fwhm_m
    )
    bins = bin_slowness(s[use], summary.median_s_per_m[use], cable.tangent_enu)
    interval = bootstrap_direction(bins, config=FitConfig(loss=args.loss), draws=args.draws)
    print(
        f"Backazimuth {interval.fit.backazimuth_degrees:.1f}°; conditional 95% interval "
        f"{interval.interval_degrees}; accepted {interval.accepted_draws}/{args.draws}"
    )
    fig = plot_direction_fit(
        bins, interval,
        config=PlotConfig(time_range_s=args.plot_time, distance_range_m=args.plot_distance_m),
    )
    save_figure(fig, args, "03_direction_fit")


if __name__ == "__main__":
    main()
