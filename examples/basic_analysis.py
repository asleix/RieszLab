"""Primary attributes and continuous enhancement on small synthetic packets.

Copyright (c) 2026 Aleix Seguí Ugalde.
Affiliation: University of Oxford.
Contact: aleix.seguiugalde@eng.ox.ac.uk
"""

import argparse
from pathlib import Path

import numpy as np

from rieszlab import GaussianTensorConfig, enhance_coherent, monogenic
from rieszlab.preprocessing import butterworth, remove_common_mode, ricker_space
from synthetic_data import demo_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("example_output/basic"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--ricker-wavelength-m", type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw, dx, dt, *_ = demo_data()
    x = butterworth(raw, dt, band_hz=(3, 30), order=4)
    x = remove_common_mode(x).data
    if args.ricker_wavelength_m is not None:
        x = ricker_space(x, dx, peak_wavelength_m=args.ricker_wavelength_m)
    result = monogenic(
        x,
        dx,
        dt,
        tensor=GaussianTensorConfig(sigma_space_m=60, sigma_time_s=0.09, tau=0.01),
    )
    enhanced = enhance_coherent(result)
    np.savez_compressed(
        args.output / "attributes.npz",
        amplitude=result.amplitude,
        phase=result.phase_directional,
        coherency=result.coherency,
        slowness_s_per_m=result.slowness_s_per_m,
        enhanced=enhanced.data,
    )
    if not args.no_plots:
        from rieszlab.plotting import plot_attributes, plot_enhancement
        import matplotlib.pyplot as plt

        for name, fig in [
            ("01_attributes", plot_attributes(result, units="arbitrary strain units")),
            ("02_enhancement", plot_enhancement(result, enhanced, units="arbitrary strain units")),
        ]:
            fig.savefig(args.output / f"{name}.png", dpi=200)
            plt.close(fig)
    print(f"Saved {args.output}; enhancement was not fed back into estimation.")


if __name__ == "__main__":
    main()
