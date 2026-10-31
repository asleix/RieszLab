"""Continuous coherent enhancement of an Alboran record; display crop only.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from rieszlab import enhance_coherent
from rieszlab.plotting import plot_enhancement
from _alboran import analyse, load, parser, plot_config, save


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--power", type=float, default=1.0)
    args = p.parse_args()
    record = load(args)
    result = analyse(record, args)
    enhanced = enhance_coherent(result, power=args.power)
    # Estimates remain those of the input, not the enhanced field.
    fig = plot_enhancement(
        result,
        enhanced,
        config=plot_config(args),
        units=record.units,
        distance_start_m=record.distance_start_m,
        time_start_s=record.time_start_s,
    )
    save(fig, args, "01_enhancement")


if __name__ == "__main__":
    main()
