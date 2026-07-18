#!/usr/bin/env python3
"""
render_run.py

Responsibility:
    Driver that generates the full default figure suite for one run file, so a
    fresh run can be inspected with a single command instead of eight. It
    invokes the plotting scripts as subprocesses at their default settings:

        plot_snapshot.py         first and last snapshot (2D heatmaps)
        animate_heat.py          2D animation (mp4; gif too with --gif)
        plot_surface.py          first and last snapshot (3D surfaces)
        animate_surface.py       3D animation (mp4; gif too with --gif)
        plot_spectral_energy.py  radially binned energy spectra
        plot_diagnostics.py      L2 norm / min / max / mean vs time
        plot_convergence.py      L2 error + mean drift vs time

    plot_convergence.py runs unconditionally: it plots the analytic error when
    the run has one (Fourier-mode validation runs) and degrades gracefully to
    mean-drift-only for demo runs, so no IC-type conditional is needed here.

    Failures do not stop the suite: each step is attempted, and a summary at
    the end lists what succeeded and what failed. One broken figure should not
    cost the other seven.

Usage:
    python scripts/render_run.py output/data/heat_multi_mode.h5
    python scripts/render_run.py output/data/heat_multi_mode.h5 --gif

Run from the project root so relative paths resolve; the plotting scripts
inherit the working directory, so their default output/figures/... filing
works unchanged.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat2d_io import load_run

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the full default figure suite for one run.")

    parser.add_argument("path", help="path to the HDF5 run file")
    parser.add_argument("--gif", action="store_true",
                        help="also render GIF versions of both animations "
                             "(slow and large; mp4 is always produced)")

    return parser.parse_args()


def run_step(description, script, script_args):
    """
    Runs one plotting script as a subprocess. Returns True on success. Output
    streams through so per-script progress lines stay visible.
    """
    command = [sys.executable, os.path.join(SCRIPTS_DIR, script)] + script_args

    print(f"\n=== {description} ===")
    print("    " + " ".join(command[1:]))

    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"    FAILED (exit code {result.returncode})", file=sys.stderr)
        return False

    return True


def main():
    args = parse_args()

    if not os.path.exists(args.path):
        sys.exit(f"no such file: {args.path}")

    # The only reason the driver reads the file itself: the plotting scripts
    # take explicit indices, and "the last snapshot" requires knowing nt.
    run = load_run(args.path)
    last = run.nt - 1

    steps = [
        ("2D snapshot, first frame",
         "plot_snapshot.py", [args.path, "--index", "0"]),
        ("2D snapshot, last frame",
         "plot_snapshot.py", [args.path, "--index", str(last)]),
        ("2D animation (mp4)",
         "animate_heat.py", [args.path]),
        ("3D surface, first frame",
         "plot_surface.py", [args.path, "--index", "0"]),
        ("3D surface, last frame",
         "plot_surface.py", [args.path, "--index", str(last)]),
        ("3D animation (mp4)",
         "animate_surface.py", [args.path]),
        ("spectral energy",
         "plot_spectral_energy.py", [args.path]),
        ("diagnostics",
         "plot_diagnostics.py", [args.path]),
        ("convergence / conservation",
         "plot_convergence.py", [args.path]),
    ]

    if args.gif:
        steps.insert(3, ("2D animation (gif)",
                         "animate_heat.py", [args.path, "--gif"]))
        steps.append(("3D animation (gif)",
                      "animate_surface.py", [args.path, "--gif"]))

    succeeded = []
    failed = []

    for description, script, script_args in steps:
        if run_step(description, script, script_args):
            succeeded.append(description)
        else:
            failed.append(description)

    print("\n" + "=" * 60)
    print(f"render_run: {len(succeeded)}/{len(steps)} steps succeeded "
          f"for {args.path}")

    if failed:
        print("failed steps:")
        for description in failed:
            print(f"  - {description}")
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(args.path))[0]
    print(f"figures are under output/figures/{run.ic_type}/{stem}/")


if __name__ == "__main__":
    main()