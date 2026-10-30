"""Two soft direction selections using the SAME unenhanced monogenic field.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from rieszlab import (
    DirectionSelectionConfig,
    RieszConfig,
    SteeredTensorConfig,
    enhance_directional,
    monogenic,
)
from rieszlab.plotting import plot_direction_selection
from _alboran import load, parser, plot_config, save


def main() -> None:
    p = parser(__doc__, gaussian_tensor=False)
    p.add_argument("--slownesses-s-per-km", type=float, nargs=2, default=(-0.5, 0.5))
    p.add_argument("--sigma-parallel-m", type=float, default=300.0)
    p.add_argument("--sigma-normal-m", type=float, default=50.0)
    p.add_argument("--angular-sigma-degrees", type=float, default=10.0)
    p.set_defaults(plot_time=(90.0, 150.0), plot_distance_m=(10000.0, 50000.0))
    args = p.parse_args()
    record = load(args)
    original = monogenic(
        record.data,
        record.dx_m,
        record.dt_s,
        config=RieszConfig(reference_velocity_mps=args.metric_mps),
    )
    outputs = []
    for p_km in args.slownesses_s_per_km:
        p_si = p_km / 1000.0  # Unit conversion, not a metric choice.
        tensor = SteeredTensorConfig(p_si, args.sigma_parallel_m, args.sigma_normal_m, tau=args.tau)
        result = original.with_tensor(tensor)
        outputs.append(
            enhance_directional(
                result,
                selection=DirectionSelectionConfig(
                    p_si, angular_sigma_degrees=args.angular_sigma_degrees
                ),
            )
        )
        del result  # Reuse the FFT, but release each large tensor before the next.
    fig = plot_direction_selection(
        original,
        *outputs,
        config=plot_config(args),
        units=record.units,
        time_start_s=record.time_start_s,
        distance_start_m=record.distance_start_m,
    )
    save(fig, args, "04_direction_selection")


if __name__ == "__main__":
    main()
