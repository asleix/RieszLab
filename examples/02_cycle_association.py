"""Phase-associated oscillations and local slowness summaries on Alboran data.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

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
from rieszlab.plotting import PlotConfig, plot_association
from alboran_io import add_packet_arguments, argument_parser, load_and_preprocess, save_figure


def main() -> None:
    p = argument_parser(__doc__)
    add_packet_arguments(p)
    p.set_defaults(plot_time=(98.0, 110.0), plot_distance_m=(21000.0, 26000.0))
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
    # Follow phase crossings from a reference channel using the Riesz orientation.
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
    fig = plot_association(
        result, tracks, summary=summary,
        config=PlotConfig(time_range_s=args.plot_time, distance_range_m=args.plot_distance_m),
    )
    save_figure(fig, args, "02_cycle_association")


if __name__ == "__main__":
    main()
