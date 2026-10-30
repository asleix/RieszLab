"""Run the four data examples on a small HDF5 travelling-packet fixture.

Copyright (c) 2026 Aleix Seguí Ugalde. University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def example_input(tmp_path_factory):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    directory = tmp_path_factory.mktemp("alboran_examples")
    s = np.arange(601) * 10.0
    angle = -0.9 + 1.8 * s / s[-1]
    position = np.cumsum(10 * np.column_stack((np.sin(angle), np.cos(angle), 0 * s)), axis=0)
    position -= position[0]
    delay = position[:, :2] @ np.array([-0.0004, 0.0003])
    tau = np.arange(1600)[None, :] * 0.01 - 7 - delay[:, None]
    data = np.exp(-0.5 * (tau / 2) ** 2) * np.cos(2 * np.pi * 6 * tau)
    with h5py.File(directory / "packet.h5", "w") as f:
        ds = f.create_dataset("data", data=data.astype("float32"))
        ds.attrs.update(dx_m=10.0, dt_s=0.01, units="nanostrain")
    np.savetxt(
        directory / "geometry.txt",
        np.column_stack((s, position)),
        header="distance_m east_m north_m up_m",
    )
    return directory


@pytest.mark.parametrize(
    "name", ["01_enhancement", "02_cycle_association", "03_direction_fit", "04_direction_selection"]
)
def test_example_runs(name, example_input):
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-B",
        str(root / "examples" / f"{name}.py"),
        "--input",
        str(example_input / "packet.h5"),
        "--output",
        str(example_input / name),
        "--no-common-mode",
        "--plot-time",
        "5",
        "12",
        "--plot-distance-m",
        "500",
        "5500",
    ]
    if name.startswith(("02", "03")):
        command += ["--search-time", "7", "9", "--reference-m", "3000"]
    if name.startswith("03"):
        command += [
            "--geometry",
            str(example_input / "geometry.txt"),
            "--fit-distance-m",
            "500",
            "5500",
            "--draws",
            "20",
            "--geometry-fwhm-m",
            "20",
        ]
    env = dict(os.environ, PYTHONPATH=str(root), MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (example_input / name / f"{name}.png").stat().st_size > 1000
    assert (example_input / name / f"{name}.pdf").stat().st_size > 1000
