"""Small curved-cable wave packets for examples; not a full elastic DAS simulator.

Copyright (c) 2026 Aleix Seguí Ugalde.
Affiliation: University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

import numpy as np


def demo_data() -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray]:
    dx, dt = 10.0, 0.01
    s, t = np.arange(240) * dx, np.arange(1600) * dt
    angle = np.deg2rad(30 + 45 * np.sin(2 * np.pi * s / 1800))
    tangent = np.column_stack((np.sin(angle), np.cos(angle)))
    position = np.cumsum(tangent * dx, axis=0)
    position -= position[0]
    vector = np.array([4e-4, 3e-4])
    delay = position @ vector
    tau = t[None, :] - 5 - delay[:, None]
    first = 2 * np.exp(-0.5 * (tau / 0.6) ** 2) * np.cos(2 * np.pi * 6 * tau)
    tau2 = t[None, :] - 11 + 0.0004 * s[:, None]
    second = np.exp(-0.5 * (tau2 / 0.8) ** 2) * np.cos(2 * np.pi * 9 * tau2)
    noise = np.random.default_rng(12).normal(0, 0.2, first.shape)
    return first + second + noise, dx, dt, s, position, vector
