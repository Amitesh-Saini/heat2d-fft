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

    For each plotted time a theory band E(k, 0) * exp(-2 alpha k^2 t) is drawn
    underneath the measured markers. Markers sitting inside their band is a
    mode-by-mode verification of the decay law, independent of (and finer
    grained than) the aggregate L2 error check. The lower panel makes that
    quantitative: it plots |measured / predicted - 1| per shell on a log axis
    against a machine-epsilon reference.

Method:
    Each snapshot is transformed with numpy's FFT (an independent
    implementation from the solver's own FFT -- a free cross-check), mode
    energies |u_hat|^2 are collected into radial shells, and shell energies
    are SUMMED (the standard energy-spectrum convention).

    The theory curve is evaluated PER MODE and then binned with the same
    shells as the measurement, rather than evaluated once per shell at a
    representative |k|. Within a shell the modes do not all share the same
    |k|, so a per-shell evaluation would introduce a binning error into the
    reference curve itself and the comparison would no longer be exact.

Usage:
    python scripts/plot_spectral_energy.py output/data/heat_hot_square.h5
    python scripts/plot_spectral_energy.py output/data/heat_multi_mode.h5 --show

Run from the project root so relative paths resolve.

Notes:
    - The y-axis is clipped at a display floor: modes decay below the
      double-precision range and become exact zeros, which a log axis cannot
      show. Shells at or below the floor are omitted rather than drawn at the
      floor, so no line is ever drawn across an empty shell.
    - Sparse spectra (Fourier-mode ICs occupy a handful of shells) are drawn
      as markers only, since a connecting line between isolated shells implies
      structure that is not there. Rich spectra (gaussian, hot_square) keep
      the connecting line.
    - Shell binning assumes a square domain. On Lx != Ly the mode lattice is
      not a single integer lattice in |k|, so shells are binned by physical
      |k| in bins of the smaller fundamental instead, and the x-axis is
      labelled accordingly.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat2d_io import load_run, provenance_label

# Log axes cannot show zero; energies at or below this are treated as decayed
# to numerical zero and omitted from the plot.
DISPLAY_FLOOR = 1e-20

# Deviations below this are treated as exact agreement: a log axis cannot
# show zero, and below one epsilon there is nothing left to resolve.
EPSILON = float(np.finfo(float).eps)
RESIDUAL_FLOOR = EPSILON / 2.0

# At or below this many occupied shells the spectrum is treated as sparse and
# drawn without connecting lines.
SPARSE_SHELL_LIMIT = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot radially binned spectral energy E(|k|, t).")

    parser.add_argument("path", help="path to the HDF5 run file")
    parser.add_argument("--num-times", type=int, default=5,
                        help="number of snapshot times to plot (default 5)")
    parser.add_argument("--kmax", type=int, default=None,
                        help="largest shell to plot (default: where the "
                             "initial spectrum falls below the display floor)")
    parser.add_argument("--outdir", default="output/figures",
                        help="base directory for the saved figure")
    parser.add_argument("--no-residual", action="store_true",
                        help="omit the measured/predicted residual panel")
    parser.add_argument("--show", action="store_true",
                        help="display the figure interactively")

    return parser.parse_args()


def mode_geometry(nx, ny, Lx, Ly):
    """
    Returns (k_magnitude, shell_index, num_shells, k_unit, square).

    Physical wavenumbers are kx = 2*pi*m/Lx, ky = 2*pi*n/Ly with integer m, n
    (negative frequencies included), matching the solver's convention.

    On a square domain both axes share one fundamental, so |k| / k_unit lands
    on (near-)integers for the axis modes and integer shells are the natural
    binning. On a rectangular domain the two fundamentals differ and no single
    integer lattice exists, so the same rounding is applied against the
    smaller fundamental -- still a fixed-width radial binning, just without
    the integer-shell interpretation.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)   # shape (nx,)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)   # shape (ny,)

    k_magnitude = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)

    square = np.isclose(Lx, Ly)

    k_unit = 2.0 * np.pi / max(Lx, Ly)

    shell_index = np.rint(k_magnitude / k_unit).astype(int)
    num_shells = int(shell_index.max()) + 1

    return k_magnitude, shell_index, num_shells, k_unit, square


def bin_energy(energy_2d, shell_index, num_shells):
    """Sums per-mode energies into radial shells."""
    return np.bincount(shell_index.ravel(),
                       weights=energy_2d.ravel(),
                       minlength=num_shells)


def mode_energy(field):
    """
    Per-mode energy |u_hat|^2 with the 1/(nx*ny) normalization, so magnitudes
    are resolution-independent coefficient magnitudes.
    """
    nx, ny = field.shape
    coefficients = np.fft.fft2(field) / (nx * ny)
    return np.abs(coefficients) ** 2


def visible(values):
    """Values above the display floor, with the rest as NaN.

    NaN rather than the floor itself: matplotlib breaks a line at NaN, so an
    empty shell interrupts the curve instead of producing a spurious V down to
    the floor and back. A Fourier-mode IC occupies only a few shells, and
    without this the gaps between them dominate the figure.
    """
    return np.where(values > DISPLAY_FLOOR, values, np.nan)


def main():
    args = parse_args()

    run = load_run(args.path)

    # Log-spaced snapshot indices: the interesting change happens early, so
    # linear spacing wastes most curves on the slow tail. Always include the
    # initial spectrum.
    count = max(2, min(args.num_times, run.nt))
    log_positions = np.unique(np.geomspace(1, run.nt, count).astype(int) - 1)
    indices = sorted(set([0]) | set(int(i) for i in log_positions))

    field_0 = np.asarray(run.snapshots[0])
    nx, ny = field_0.shape

    k_magnitude, shell_index, num_shells, k_unit, square = mode_geometry(
        nx, ny, run.Lx, run.Ly)

    # Initial spectrum: defines the theory baseline and the default k range
    # worth plotting. Held per mode, not per shell, because the theory curve
    # is evaluated per mode.
    energy_0_modes = mode_energy(field_0)
    energy_0 = bin_energy(energy_0_modes, shell_index, num_shells)

    shells = np.arange(num_shells)

    if args.kmax is not None:
        kmax = args.kmax
    else:
        occupied = np.where(energy_0 > DISPLAY_FLOOR)[0]
        kmax = int(occupied.max()) if occupied.size else int(shells.max())

    keep = shells <= kmax
    shells = shells[keep]

    # A spectrum occupying a handful of shells is a set of isolated spikes; a
    # line between them would imply structure between the spikes that does not
    # exist. Decided from the initial spectrum so every curve in one figure is
    # styled the same way.
    occupied_count = int(np.count_nonzero(energy_0[keep] > DISPLAY_FLOOR))
    sparse = occupied_count <= SPARSE_SHELL_LIMIT

    colours = cm.viridis(np.linspace(0.0, 0.85, len(indices)))

    show_residual = not args.no_residual

    if show_residual:
        fig, (ax, ax_res) = plt.subplots(
            2, 1, figsize=(8, 6.5), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax_res = None

    worst_deviation = 0.0

    for colour, index in zip(colours, indices):
        time = float(run.times[index])

        measured = bin_energy(mode_energy(np.asarray(run.snapshots[index])),
                              shell_index, num_shells)[keep]

        # Theory per mode, then binned with the same shells as the
        # measurement, so both sides of the comparison see identical binning.
        predicted_modes = energy_0_modes * np.exp(
            -2.0 * run.alpha * k_magnitude ** 2 * time)
        predicted = bin_energy(predicted_modes, shell_index, num_shells)[keep]

        # Theory underneath as a wide pale band, measurement on top as
        # markers. Drawn the other way round -- both as thin lines in the same
        # colour -- perfect agreement hides the reference completely and the
        # figure appears to show one unverified curve.
        # A line cannot render isolated points, and a sparse spectrum is
        # nothing but isolated points, so the band becomes invisible in
        # exactly the case where the reader most needs it. Wide pale markers
        # instead: the measurement dot sits inside its prediction halo.
        if sparse:
            ax.plot(shells, visible(predicted), linestyle="none",
                    marker="o", markersize=12, color=colour, alpha=0.30,
                    zorder=1)
        else:
            ax.plot(shells, visible(predicted), linestyle="-", linewidth=4.0,
                    color=colour, alpha=0.30, zorder=1)

        ax.plot(shells, visible(measured),
                marker="o", markersize=5 if sparse else 3,
                linestyle="none" if sparse else "-",
                linewidth=1.2, color=colour, zorder=2,
                label=f"t = {time:.4f}")

        if show_residual:
            both = (measured > DISPLAY_FLOOR) & (predicted > DISPLAY_FLOOR)

            deviation = np.full(shells.shape, np.nan)
            deviation[both] = np.abs(measured[both] / predicted[both] - 1.0)

            # Exact agreement is not plottable on a log axis; clip it to the
            # reference line rather than dropping the point, so a shell that
            # matches perfectly still appears.
            deviation = np.where(deviation < RESIDUAL_FLOOR,
                                 RESIDUAL_FLOOR, deviation)

            ax_res.plot(shells, deviation, marker="o", markersize=4,
                        linestyle="none", color=colour)

            if np.any(both):
                worst_deviation = max(
                    worst_deviation,
                    float(np.nanmax(deviation[both])))

    ax.set_yscale("log")
    ax.set_ylim(DISPLAY_FLOOR / 2, None)
    ax.set_ylabel("shell energy  sum |u_hat|^2")
    ax.set_title(f"{run.ic_type}  {run.nx}x{run.ny}:  "
                 "high wavenumbers decay as exp(-2 alpha |k|^2 t)")

    # Shell indices are integers; half-integer ticks label nothing.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # One legend entry explaining the band, then the time entries. Outside the
    # axes: a sparse spectrum leaves its occupied shells wherever they fall,
    # and an in-axes legend lands on top of them often enough to matter.
    handles, labels = ax.get_legend_handles_labels()
    if sparse:
        theory_handle = plt.Line2D([], [], linestyle="none", marker="o",
                                   markersize=10, color="0.5", alpha=0.45,
                                   label="theory: E(k,0) exp(-2 alpha k^2 t)")
    else:
        theory_handle = plt.Line2D([], [], linestyle="-", linewidth=4.0,
                                   color="0.5", alpha=0.45,
                                   label="theory: E(k,0) exp(-2 alpha k^2 t)")
    ax.legend(handles=handles + [theory_handle], fontsize=8,
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0)

    ax.grid(True, which="both", alpha=0.25)

    # Fixed to the shell range rather than to the occupied shells: a sparse
    # spectrum would otherwise crop the axis to its first and last spike and
    # hide the empty shells that are part of the story.
    ax.set_xlim(float(shells.min()) - 0.5, float(shells.max()) + 0.5)

    x_axis = ax_res if show_residual else ax

    if square:
        x_axis.set_xlabel("wavenumber shell  |k| / k_unit")
    else:
        x_axis.set_xlabel(f"radial bin  |k| / (2*pi/{max(run.Lx, run.Ly):g})")

    if show_residual:
        # Log scale, not a window around 1. Shell energies span the whole
        # double-precision range, so the relative deviation does too: a linear
        # window is either clipped by the noisiest near-floor shell or blind
        # to the well-resolved ones. The epsilon line gives the reader a fixed
        # reference to read every point against.
        ax_res.axhline(EPSILON, color="0.4", linewidth=1.0,
                       linestyle="--", zorder=1)
        ax_res.set_yscale("log")
        ax_res.set_ylabel("|measured\n/ predicted - 1|", fontsize=8)
        ax_res.grid(True, which="both", alpha=0.25)
        ax_res.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_res.tick_params(labelsize=8)
        ax_res.set_ylim(RESIDUAL_FLOOR / 2, max(worst_deviation * 5, 1e-12))
        # Placed in data coordinates just above the line it labels, so it
        # follows the line when the panel rescales instead of landing on top
        # of it.
        ax_res.text(0.02, EPSILON * 2.5, "machine epsilon", fontsize=7,
                    color="0.4", ha="left", va="bottom",
                    transform=ax_res.get_yaxis_transform())

    fig.text(0.01, 0.01, provenance_label(run), fontsize=7, color="0.4")

    stem = os.path.splitext(os.path.basename(args.path))[0]

    outdir = os.path.join(args.outdir, run.ic_type, stem)
    os.makedirs(outdir, exist_ok=True)

    out_path = os.path.join(outdir, "spectral_energy.png")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")

    if show_residual:
        print(f"worst |measured/predicted - 1| over plotted shells: "
              f"{worst_deviation:.3e}  ({worst_deviation / EPSILON:.1f} eps)")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()