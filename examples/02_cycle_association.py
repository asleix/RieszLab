"""Phase-associated oscillations and local slowness summaries on Alboran data.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from rieszlab.plotting import plot_association
from _packets import add_arguments, packet
from _alboran import analyse, load, parser, plot_config, save


def main() -> None:
    p = parser(__doc__)
    add_arguments(p)
    p.set_defaults(plot_time=(98.0, 110.0), plot_distance_m=(21000.0, 26000.0))
    args = p.parse_args()
    record = load(args)
    result = analyse(record, args)
    tracks, summary = packet(result, record, args)
    fig = plot_association(result, tracks, summary=summary, config=plot_config(args))
    save(fig, args, "02_cycle_association")


if __name__ == "__main__":
    main()
