#!/usr/bin/env python3
"""
plot_diagnostics.py

Responsibility:
    Plot the physical diagnostics of one run against time on a single linear
    axis: L2 norm, max, min, and mean. Together they show the defining
    behaviour of diffusion -- everything collapses onto the conserved mean:

      - mean(t)      : exactly conserved (the k = 0 mode has decay factor 1);
                       plotted as the flat line the others converge toward.
      - max(t)       : non-increasing  \\  the maximum principle: diffusion
      - min(t)       : non-decreasing  /   only smooths, never sharpens.
      - ||u(t)||_2   : monotonically decreasing (every k != 0 mode decays).

    The script checks the monotonicity invariants and prints a WARNING if any
    is violated beyond floating-point tolerance -- it does not abort, because
    a tiny violation on a very sharp initial condition can be a known spectral
    interpolation artifact rather than a solver bug. A large violation, on the
    other hand, means energy appeared from nowhere and IS a bug.

Usage:
    python scripts/plot_diagnostics.py output/data/heat_gaussian.h5
    python scripts/plot_diagnostics.py output/data/heat_hot_square.h5 --show

    With --show, hovering the mouse near a line displays the exact value at
    that snapshot (interactive windows only; the saved PNG is static).

Run from the project root so relative paths resolve.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat2d_io import load_run, provenance_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot L2 norm, min, max, and mean vs time for a heat2d run.")

    parser.add_argument("path", help="path to the HDF5 run file")
    parser.add_argument("--outdir", default="output/figures",
                        help="base directory for the saved figure")
    parser.add_argument("--show", action="store_true",
                        help="display interactively (enables the hover readout)")

    return parser.parse_args()


def check_monotonic(name, values, direction, scale):
    """
    Warns (does not abort) if `values` moves the wrong way beyond a relative
    floating-point tolerance. `direction` is +1 for "must not decrease" (min)
    and -1 for "must not increase" (max, L2 norm). `scale` sets the tolerance:
    steps smaller than 1e-12 * scale are rounding noise, not violations.
    """
    tolerance = 1e-12 * scale
    steps = np.diff(values) * direction   # violations are negative steps

    worst = steps.min() if steps.size else 0.0
    if worst < -tolerance:
        index = int(np.argmin(steps))
        print(f"WARNING: {name} violates the maximum principle at snapshot "
              f"{index + 1}: moved {abs(worst):.3e} in the forbidden direction "
              f"(tolerance {tolerance:.1e}). Small violations on very sharp "
              f"initial conditions can be spectral interpolation artifacts; "
              f"large ones indicate a solver bug.", file=sys.stderr)


def main():
    args = parse_args()

    run = load_run(args.path)

    required = ("l2_norm", "max", "min", "mean")
    missing = [key for key in required if key not in run.diagnostics]
    if missing:
        sys.exit(
            f"{args.path} is missing diagnostics: {', '.join(missing)}.\n"
            "Enable diagnostics in the config to record them.")

    times = run.times
    l2_norm = run.diagnostics["l2_norm"]
    max_values = run.diagnostics["max"]
    min_values = run.diagnostics["min"]
    mean_values = run.diagnostics["mean"]

    # --- invariant checks (warn, never abort) -------------------------------
    field_scale = max(abs(float(max_values[0])), abs(float(min_values[0])), 1e-30)

    check_monotonic("max(t)", max_values, direction=-1, scale=field_scale)
    check_monotonic("min(t)", min_values, direction=+1, scale=field_scale)
    check_monotonic("||u||_2", l2_norm, direction=-1, scale=float(l2_norm[0]))

    # --- plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    lines = []
    for label, values, colour in [
        ("max u(t)", max_values, "tab:red"),
        ("||u(t)||_2", l2_norm, "tab:blue"),
        ("mean u(t)  (conserved)", mean_values, "tab:green"),
        ("min u(t)", min_values, "tab:purple"),
    ]:
        (line,) = ax.plot(times, values, marker="o", markersize=3,
                          linewidth=1.2, color=colour, label=label)
        lines.append((line, values))

    ax.set_xlabel("time  t")
    ax.set_ylabel("value")
    ax.set_title(f"{run.ic_type}  {run.nx}x{run.ny}:  "
                 "extrema and energy collapse onto the conserved mean")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)

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
                    f"t = {times[index]:.4f}\n{values[index]:.6g}")
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

    out_path = os.path.join(outdir, "diagnostics.png")

    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()