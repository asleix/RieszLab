"""Continuous coherent enhancement of an Alboran record; display crop only.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from rieszlab import GaussianTensorConfig, RieszConfig, enhance_coherent, monogenic
from rieszlab.plotting import PlotConfig, plot_enhancement
from alboran_io import argument_parser, load_and_preprocess, save_figure


def main() -> None:
    p = argument_parser(__doc__)
    p.add_argument("--power", type=float, default=1.0)
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
    enhanced = enhance_coherent(result, power=args.power)
    # Estimates remain those of the input, not the enhanced field.
    fig = plot_enhancement(
        result,
        enhanced,
        config=PlotConfig(time_range_s=args.plot_time, distance_range_m=args.plot_distance_m),
        units=record.units,
        distance_start_m=record.distance_start_m,
        time_start_s=record.time_start_s,
    )
    save_figure(fig, args, "01_enhancement")


if __name__ == "__main__":
    main()
