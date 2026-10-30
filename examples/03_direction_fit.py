"""Fit horizontal incoming direction from associated Alboran slowness and geometry.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from pathlib import Path

from rieszlab.inversion import FitConfig, bin_slowness, bootstrap_direction
from rieszlab.plotting import plot_direction_fit
from _packets import add_arguments, packet
from _alboran import analyse, load, load_geometry, parser, plot_config, save


def main() -> None:
    p = parser(__doc__)
    add_arguments(p)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--geometry-lonlat", action="store_true")
    p.add_argument("--geometry-fwhm-m", type=float, default=500.0)
    p.add_argument("--fit-distance-m", type=float, nargs=2, default=(21000.0, 26000.0))
    p.add_argument("--loss", choices=("huber", "mse"), default="huber")
    p.add_argument("--draws", type=int, default=400)
    args = p.parse_args()
    record = load(args)
    result = analyse(record, args)
    _, summary = packet(result, record, args)
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
    save(plot_direction_fit(bins, interval, config=plot_config(args)), args, "03_direction_fit")


if __name__ == "__main__":
    main()
