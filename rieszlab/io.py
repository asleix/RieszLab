"""Small optional HDF5 reader; no event paths or processing hidden in loading.

Copyright (c) 2026 Aleix Seguí Ugalde.
Affiliation: University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from typing import Any

import numpy as np

from ._validation import positive as _positive


@dataclass
class DASRecord:
    data: np.ndarray
    dx_m: float
    dt_s: float
    distance_start_m: float = 0.0
    time_start_s: float = 0.0
    units: str = "unspecified"
    metadata: dict[str, Any] = field(default_factory=dict)
    measured_channel: np.ndarray | None = None

    @property
    def distance_m(self) -> np.ndarray:
        return self.distance_start_m + np.arange(len(self.data)) * self.dx_m

    @property
    def time_s(self) -> np.ndarray:
        return self.time_start_s + np.arange(self.data.shape[1]) * self.dt_s


def read_hdf5(
    path: str | PathLike[str],
    *,
    dataset: str = "data",
    channels: slice = slice(None),
    time_range_s: tuple[float, float] | None = None,
    dx_m: float | None = None,
    dt_s: float | None = None,
    units: str | None = None,
    dtype: str = "float64",
) -> DASRecord:
    """Read an Alboran-style (channel,time) dataset, preserving amplitude values.

    Sampling comes from dx_m/dt_s attributes unless explicitly supplied.
    time_range_s is relative to the record start; no origin time is guessed.
    Units are metadata, not a conversion. The three Alboran files are confirmed
    nanostrain: pass units='nanostrain' if their header is absent/incorrect.
    A strided channel read changes dx; temporal decimation is not performed.
    Loading defaults to double precision; dtype='float32' is an explicit opt-in.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install rieszlab[io] to read HDF5 files") from exc
    if not isinstance(channels, slice) or (channels.step or 1) <= 0:
        raise ValueError("channels must be a positive-step slice")
    with h5py.File(path, "r") as handle:
        ds = handle[dataset]
        if ds.ndim != 2:
            raise ValueError("Dataset must be (channel,time)")
        attrs = dict(handle.attrs) | dict(ds.attrs)
        if (dx_m is None and "dx_m" not in attrs) or (dt_s is None and "dt_s" not in attrs):
            raise ValueError("Sampling missing from HDF5: supply dx_m and dt_s explicitly")
        dx = float(attrs["dx_m"] if dx_m is None else dx_m)
        dt = float(attrs["dt_s"] if dt_s is None else dt_s)
        _positive(dx, "dx_m")
        _positive(dt, "dt_s")
        first, last, stride = channels.indices(ds.shape[0])
        start, stop = 0, ds.shape[1]
        if time_range_s is not None:
            if (
                len(time_range_s) != 2
                or not np.isfinite(time_range_s).all()
                or time_range_s[0] >= time_range_s[1]
            ):
                raise ValueError("time_range_s must be an increasing finite pair")
            start = max(0, int(np.ceil(time_range_s[0] / dt)))
            stop = min(stop, int(np.ceil(time_range_s[1] / dt)))
        if last <= first or stop <= start:
            raise ValueError("Requested data window is empty")
        x = np.asarray(ds[first:last:stride, start:stop], dtype=dtype)

    def decode(value: Any) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    return DASRecord(
        x,
        dx * stride,
        dt,
        first * dx,
        start * dt,
        units if units is not None else decode(attrs.get("units", "unspecified")),
        {
            "path": str(path),
            "dataset": dataset,
            "recording_start_utc": decode(attrs.get("begin_time", "")),
            "header_units": decode(attrs.get("units", "unspecified")),
            "channel_slice": (first, last, stride),
            "sample_slice": (start, stop),
        },
    )
