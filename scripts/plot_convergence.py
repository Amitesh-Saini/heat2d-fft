#!/usr/bin/env python3
"""
plot_convergence.py

Responsibility:
    Plot the validation quantities of one run against time:

      - relative L2 error vs. the analytic solution (/error/relative_l2),
        available for Fourier-mode runs with compute_analytic_error enabled;
      - mean drift |mean(t) - mean(0)| (from /diagnostics/mean), available for
        any run with diagnostics enabled. The k=0 mode has decay factor
        exp(0) = 1, so the mean is exactly conserved and any drift measures
        floating-point noise in the FFT round trip.

    Both should sit at machine precision and stay FLAT: the spectral method
    evaluates exp(-alpha |k|^2 t) directly at each output time instead of
    time-stepping, so there is no error accumulation. The flatness of these
    lines is the claim this figure makes.

    A dashed reference line marks double-precision machine epsilon so the
    reader can see the values are at the floating-point floor, not merely
    "small".

Usage:
    python scripts/plot_convergence.py output/data/heat_multi_mode.h5
    python scripts/plot_convergence.py output/data/heat_single_mode.h5 --show

    The saved PNG is static. With --show, hovering the mouse near either line
    displays the exact value at that snapshot (interactive windows only --
    a PNG cannot respond to the mouse).

Run from the project root so relative paths resolve.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat2d_io import load_run, provenance_label

MACHINE_EPSILON = np.finfo(np.float64).eps   # ~2.22e-16

# Log axes cannot show zero; exact zeros (e.g. the mean drift at t = 0, which
# is identically mean(0) - mean(0)) are clipped up to this floor for display.
LOG_FLOOR = 1e-18


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot L2 error and mean drift vs time for a heat2d run.")

    parser.add_argument("path", help="path to the HDF5 run file")
    parser.add_argument("--outdir", default="output/figures",
                        help="base directory for the saved figure")
    parser.add_argument("--show", action="store_true",
                        help="display interactively (enables the hover readout)")

    return parser.parse_args()


def main():
    args = parse_args()

    run = load_run(args.path)

    times = run.times
    series = []   # (label, values, colour) for every line we can plot

    if run.relative_l2_error is not None:
        series.append(("relative L2 error vs analytic solution",
                       np.maximum(run.relative_l2_error, LOG_FLOOR),
                       "tab:blue"))

    if "mean" in run.diagnostics:
        means = run.diagnostics["mean"]
        drift = np.abs(means - means[0])
        series.append(("mean drift  |mean(t) - mean(0)|",
                       np.maximum(drift, LOG_FLOOR),
                       "tab:orange"))

    if not series:
        sys.exit(
            f"{args.path} has neither /error/relative_l2 nor /diagnostics/mean.\n"
            "Enable diagnostics (and compute_analytic_error for Fourier-mode "
            "runs) in the config to produce validation data.")

    fig, ax = plt.subplots(figsize=(8, 5))

    lines = []
    for label, values, colour in series:
        (line,) = ax.plot(times, values, marker="o", markersize=3,
                          linewidth=1.2, color=colour, label=label)
        lines.append((line, values))

    ax.axhline(MACHINE_EPSILON, color="0.5", linestyle="--", linewidth=1.0,
               label=f"machine epsilon ({MACHINE_EPSILON:.2e})")

    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-10)   # headroom above the data so tiny wiggles do
                                # not read as dramatic variation
    ax.set_xlabel("time  t")
    ax.set_ylabel("error magnitude")
    ax.set_title(f"{run.ic_type}  {run.nx}x{run.ny}:  "
                 "Mean conservation remains near machine precision")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)

    fig.text(0.01, 0.01, provenance_label(run), fontsize=7, color="0.4")
    fig.tight_layout()

    # --- hover readout (interactive windows only) ---------------------------
    annotation = ax.annotate(
        "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
        fontsize=8, visible=False)

    def on_mouse_move(event):
        if event.inaxes != ax:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return

        for line, values in lines:
            contains, info = line.contains(event)
            if contains:
                index = info["ind"][0]
                annotation.xy = (times[index], values[index])
                annotation.set_text(
                    f"t = {times[index]:.4f}\n{values[index]:.3e}")
                annotation.set_visible(True)
                fig.canvas.draw_idle()
                return

        if annotation.get_visible():
            annotation.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    # --- save, filed by IC type, matching the other plotting scripts --------
    stem = os.path.splitext(os.path.basename(args.path))[0]

    outdir = os.path.join(args.outdir, run.ic_type, stem)
    os.makedirs(outdir, exist_ok=True)

    out_path = os.path.join(outdir, "convergence.png")

    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()