#!/usr/bin/env python3
"""
plot_spectral_energy.py

Responsibility:
    Plot the radially binned spectral energy E(|k|, t) of a run at several
    times. This is the figure that shows the mechanism of the spectral solver
    directly: every Fourier mode decays as exp(-alpha |k|^2 t), so its energy
    decays as exp(-2 alpha |k|^2 t). On a semilog-y axis with linear |k|, each
    successive curve is the initial spectrum pushed down by a PARABOLA in k --
    high wavenumbers fall off a cliff while low wavenumbers barely move. This
    is wavenumber-selective damping, plotted.

    For each plotted time a dashed theory curve E(k, 0) * exp(-2 alpha k^2 t)
    is overlaid. Measured and predicted spectra lying on top of each other is
    a mode-by-mode verification of the decay law, independent of (and finer
    grained than) the aggregate L2 error check.

Method:
    Each snapshot is transformed with numpy's FFT (an independent
    implementation from the solver's own FFT -- a free cross-check), mode
    energies |u_hat|^2 are collected into integer radial shells by
    round(|k| / k_unit) where k_unit = 2*pi/L, and shell energies are SUMMED
    (the standard energy-spectrum convention).

Usage:
    python scripts/plot_spectral_energy.py output/data/heat_hot_square.h5
    python scripts/plot_spectral_energy.py output/data/heat_multi_mode.h5 --show

Run from the project root so relative paths resolve.

Notes:
    - The y-axis is clipped at a display floor: modes decay below the
      double-precision range and become exact zeros, which a log axis cannot
      show. Curves ending at the floor mean "decayed to numerical zero".
    - Best on ICs with a rich spectrum (hot_square, gaussian). Fourier-mode
      runs produce a spectrum of isolated spikes, which is legible but sparse.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat2d_io import load_run, provenance_label

# Log axes cannot show zero; energies below this are clipped for display.
DISPLAY_FLOOR = 1e-20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot radially binned spectral energy E(|k|, t).")

    parser.add_argument("path", help="path to the HDF5 run file")
    parser.add_argument("--num-times", type=int, default=5,
                        help="number of snapshot times to plot (default 5)")
    parser.add_argument("--kmax", type=int, default=None,
                        help="largest integer shell to plot (default: where the "
                             "initial spectrum falls below the display floor)")
    parser.add_argument("--outdir", default="output/figures",
                        help="base directory for the saved figure")
    parser.add_argument("--show", action="store_true",
                        help="display the figure interactively")

    return parser.parse_args()


def radial_energy_spectrum(field, Lx, Ly):
    """
    Returns (shells, energy): integer shell indices and the total mode energy
    |u_hat|^2 summed within each shell.

    Shell index for mode (kx, ky) is round(|k| / k_unit) with
    k_unit = 2*pi / Lx, i.e. shells are integer multiples of the fundamental
    wavenumber. Physical wavenumbers are kx = 2*pi*m/Lx, ky = 2*pi*n/Ly with
    integer m, n (negative frequencies included), matching the solver's
    convention. The FFT is normalized by 1/(nx*ny) so mode energies are
    resolution-independent coefficient magnitudes.
    """
    nx, ny = field.shape

    coefficients = np.fft.fft2(field) / (nx * ny)
    energy_2d = np.abs(coefficients) ** 2

    k_unit = 2.0 * np.pi / Lx

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)   # shape (nx,)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)   # shape (ny,)

    k_magnitude = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    shell_index = np.rint(k_magnitude / k_unit).astype(int)

    num_shells = shell_index.max() + 1
    energy = np.bincount(shell_index.ravel(),
                         weights=energy_2d.ravel(),
                         minlength=num_shells)

    return np.arange(num_shells), energy


def main():
    args = parse_args()

    run = load_run(args.path)

    # Log-spaced snapshot indices: the interesting change happens early, so
    # linear spacing wastes most curves on the slow tail. Always include the
    # initial spectrum.
    count = max(2, min(args.num_times, run.nt))
    log_positions = np.unique(np.geomspace(1, run.nt, count).astype(int) - 1)
    indices = sorted(set([0]) | set(int(i) for i in log_positions))

    # Initial spectrum: defines the shells, the theory baseline, and the
    # default k range worth plotting.
    shells, energy_0 = radial_energy_spectrum(run.snapshots[0], run.Lx, run.Ly)

    if args.kmax is not None:
        kmax = args.kmax
    else:
        visible = np.where(energy_0 > DISPLAY_FLOOR)[0]
        kmax = int(visible.max()) if visible.size else shells.max()

    keep = shells <= kmax
    shells = shells[keep]

    k_unit = 2.0 * np.pi / run.Lx
    k_physical = shells * k_unit   # physical |k| for the theory exponent

    colours = cm.viridis(np.linspace(0.0, 0.85, len(indices)))

    fig, ax = plt.subplots(figsize=(8, 5))

    for colour, index in zip(colours, indices):
        time = float(run.times[index])

        _, energy = radial_energy_spectrum(run.snapshots[index], run.Lx, run.Ly)
        measured = np.maximum(energy[keep], DISPLAY_FLOOR)

        ax.plot(shells, measured, marker="o", markersize=3, linewidth=1.2,
                color=colour, label=f"t = {time:.4f}")

        # Theory: the initial spectrum pushed down by exp(-2 alpha |k|^2 t).
        predicted = energy_0[keep] * np.exp(-2.0 * run.alpha
                                            * k_physical ** 2 * time)
        ax.plot(shells, np.maximum(predicted, DISPLAY_FLOOR),
                linestyle="--", linewidth=1.0, color=colour, alpha=0.6)

    ax.set_yscale("log")
    ax.set_ylim(DISPLAY_FLOOR / 2, None)
    ax.set_xlabel("wavenumber shell  |k| / k_unit")
    ax.set_ylabel("shell energy  sum |u_hat|^2")
    ax.set_title(f"{run.ic_type}  {run.nx}x{run.ny}:  "
                 "high wavenumbers decay as exp(-2 alpha |k|^2 t)")

    # One legend entry explaining the dashed overlay, then the time entries.
    handles, labels = ax.get_legend_handles_labels()
    theory_handle = plt.Line2D([], [], linestyle="--", color="0.4",
                               label="theory: E(k,0) exp(-2 alpha k^2 t)")
    ax.legend(handles=handles + [theory_handle], fontsize=8, loc="upper right")

    ax.grid(True, which="both", alpha=0.25)

    fig.text(0.01, 0.01, provenance_label(run), fontsize=7, color="0.4")
    fig.tight_layout()

    outdir = os.path.join(args.outdir, run.ic_type)
    os.makedirs(outdir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.path))[0]
    out_path = os.path.join(outdir, f"{stem}_spectral_energy.png")

    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()