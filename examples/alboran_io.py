"""Shared CLI, input conditioning and file output for the Alboran examples.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import numpy as np

from rieszlab.geometry import CableGeometry, cable_geometry
from rieszlab.io import DASRecord, read_hdf5
from rieszlab.preprocessing import (
    butterworth,
    interpolate_bad_channels,
    remove_common_mode,
    ricker_space,
)


def argument_parser(description: str, *, gaussian_tensor: bool = True) -> ArgumentParser:
    p = ArgumentParser(description=description)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("example_output/alboran"))
    p.add_argument("--dx-m", type=float, help="Override header sampling")
    p.add_argument("--dt-s", type=float, help="Override header sampling")
    p.add_argument("--metric-mps", type=float, help="Default: supplied/header dx divided by dt")
    p.add_argument(
        "--origin-utc", help="Optional ISO origin time; otherwise times are record-relative"
    )
    p.add_argument("--bad-channels", type=int, nargs="*", default=[])
    p.add_argument("--band", type=float, nargs=2, default=(3.0, 30.0))
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--no-common-mode", action="store_true")
    p.add_argument("--ricker-wavelength-m", type=float)
    if gaussian_tensor:
        p.add_argument("--sigma-space-m", type=float, default=90.0)
        p.add_argument("--sigma-time-s", type=float, default=0.09)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--plot-time", type=float, nargs=2)
    p.add_argument("--plot-distance-m", type=float, nargs=2)
    return p


def add_packet_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--search-time", type=float, nargs=2, default=(98.0, 110.0))
    parser.add_argument("--reference-m", type=float, default=23000.0)
    parser.add_argument("--power-aperture-m", type=float, default=500.0)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--min-cycles", type=int, default=1)
    parser.add_argument("--nonconsecutive", action="store_true")
    parser.add_argument("--association-coherency", type=float, default=0.4)
    parser.add_argument("--summary-coherency", type=float, default=0.7)


def load_and_preprocess(args: Namespace) -> DASRecord:
    """Condition the complete file, repairing only explicitly identified channels."""
    record = read_hdf5(args.input, dx_m=args.dx_m, dt_s=args.dt_s, units="nanostrain")
    bad = np.zeros(len(record.data), bool)
    if any(i < 0 or i >= len(bad) for i in args.bad_channels):
        raise ValueError("Bad-channel index outside record")
    bad[args.bad_channels] = True
    x = interpolate_bad_channels(record.data, bad) if bad.any() else record.data
    x = butterworth(x, record.dt_s, band_hz=tuple(args.band), order=args.order)
    if not args.no_common_mode:
        x = remove_common_mode(
            x, estimator="median", channel_mask=~bad, energy_mad_multiplier=None
        ).data
    if args.ricker_wavelength_m is not None:
        x = ricker_space(x, record.dx_m, peak_wavelength_m=args.ricker_wavelength_m)
    record.data = x
    record.measured_channel = ~bad
    if args.origin_utc:
        start = record.metadata["recording_start_utc"]
        if not start:
            raise ValueError("Origin-relative time needs the HDF5 begin_time attribute")
        record.time_start_s += (
            datetime.fromisoformat(start.replace("Z", "+00:00"))
            - datetime.fromisoformat(args.origin_utc.replace("Z", "+00:00"))
        ).total_seconds()
    return record


def save_figure(fig, args: Namespace, name: str) -> None:
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output/name}")


def load_geometry(
    path: Path, distance_m: np.ndarray, *, lonlat: bool = False, smoothing_fwhm_m: float = 500.0
) -> CableGeometry:
    """Four columns with one header: distance,E,N,U; optional lon/lat degrees instead of E,N."""
    table = np.loadtxt(path, skiprows=1)
    s, xyz = table[:, 0], table[:, 1:4].copy()
    if xyz.shape[1] != 3 or np.any(np.diff(s) <= 0) or not np.isfinite(table).all():
        raise ValueError("Geometry needs ordered distance and finite E,N,U columns")
    if lonlat:
        # Local tangent-plane approximation, adequate for this few-km demonstration aperture.
        lon, lat = np.deg2rad(xyz[:, 0]), np.deg2rad(xyz[:, 1])
        xyz[:, 0] = 6371008.8 * np.cos(np.mean(lat)) * (lon - lon[0])
        xyz[:, 1] = 6371008.8 * (lat - lat[0])
    if distance_m[0] < s[0] or distance_m[-1] > s[-1]:
        raise ValueError("Fit aperture extends beyond surveyed geometry; do not extrapolate")
    position = np.column_stack([np.interp(distance_m, s, col) for col in xyz.T])
    return cable_geometry(distance_m, position, smoothing_fwhm_m=smoothing_fwhm_m)
