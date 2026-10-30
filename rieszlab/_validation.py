"""Small shared numerical checks.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def matrix(data: ArrayLike, dtype: str | np.dtype | None = None) -> np.ndarray:
    x = np.asarray(data)
    if np.iscomplexobj(x) or x.ndim != 2 or min(x.shape) < 2:
        raise ValueError("Expected a real (channel, time) matrix, both dimensions >= 2")
    x = np.asarray(x, dtype=dtype or np.result_type(x.dtype, np.float32))
    if not np.isfinite(x).all():
        raise ValueError("Input contains NaN/Inf; repair or exclude bad channels explicitly")
    return x


def divide(a: ArrayLike, b: ArrayLike, *, fill: float = np.nan) -> np.ndarray:
    a, b = np.broadcast_arrays(a, b)
    out = np.full(a.shape, fill, dtype=np.result_type(a, b, np.float32))
    np.divide(a, b, out=out, where=np.isfinite(b) & (b != 0))
    return out
