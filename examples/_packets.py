"""Explicit association recipe shared by the two packet demonstrations.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

import numpy as np

from rieszlab.association import (
    AssociationConfig,
    associate_cycles,
    cycle_power,
    phase_candidates,
    select_cycles,
    summarize_tracks,
)


def add_arguments(parser) -> None:
    parser.add_argument("--search-time", type=float, nargs=2, default=(98.0, 110.0))
    parser.add_argument("--reference-m", type=float, default=23000.0)
    parser.add_argument("--power-aperture-m", type=float, default=500.0)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--min-cycles", type=int, default=1)
    parser.add_argument("--nonconsecutive", action="store_true")
    parser.add_argument("--association-coherency", type=float, default=0.4)
    parser.add_argument("--summary-coherency", type=float, default=0.7)


def packet(result, record, args):
    if not record.distance_m[0] <= args.reference_m <= record.distance_m[-1]:
        raise ValueError("Reference channel is outside record")
    bank = phase_candidates(
        result,
        distance_start_m=record.distance_start_m,
        time_start_s=record.time_start_s,
        channel_mask=record.measured_channel,
    )
    reference = int(np.argmin(abs(record.distance_m - args.reference_m)))
    tracks = associate_cycles(
        bank,
        reference,
        time_window_s=tuple(args.search_time),
        config=AssociationConfig(coherency_min=args.association_coherency),
    )
    aperture = (
        args.reference_m - args.power_aperture_m / 2,
        args.reference_m + args.power_aperture_m / 2,
    )
    scores = cycle_power(result, tracks, aperture_m=aperture)
    selected = select_cycles(tracks, scores, count=args.cycles, consecutive=not args.nonconsecutive)
    summary = summarize_tracks(
        result, selected, coherency_min=args.summary_coherency, min_cycles=args.min_cycles
    )
    return selected, summary
