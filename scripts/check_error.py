#!/usr/bin/env python3
"""
check_error.py

Reports the analytic relative L2 error for a validation run (Fourier-mode ICs
with compute_analytic_error enabled). The error should sit at machine precision
(~1e-15) and, critically, should NOT grow with time: a spectral solver
evaluates exp(-alpha |k|^2 t) directly at each output time rather than stepping,
so there is no error accumulation. A rising error would indicate the exactness
claim is false.

Usage:
    python scripts/check_error.py output/data/heat_single_mode.h5
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat2d_io import load_run


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/check_error.py <run.h5>")

    run = load_run(sys.argv[1])

    if run.relative_l2_error is None:
        sys.exit(
            f"{sys.argv[1]} has no /error/relative_l2 dataset.\n"
            "Only Fourier-mode runs with \"compute_analytic_error\": true "
            "record an analytic error."
        )

    error = run.relative_l2_error

    print(f"run          : {run.ic_type}  {run.nx}x{run.ny}  alpha={run.alpha:g}")
    print(f"snapshots    : {run.nt}  (t = {run.times[0]:g} .. {run.times[-1]:g})")
    print(f"error min    : {error.min():.3e}")
    print(f"error max    : {error.max():.3e}")
    print(f"error mean   : {error.mean():.3e}")

    # The key property: flat in time. A ratio near 1 means no accumulation.
    print(f"max/min ratio: {error.max() / error.min():.2f}   "
          f"(near 1 = no error growth in time)")

    # Also report the conservation check, which is free and independent.
    if "mean" in run.diagnostics:
        means = run.diagnostics["mean"]
        drift = np.abs(means - means[0]).max()
        print(f"mean drift   : {drift:.3e}   "
              f"(the k=0 mode is exactly conserved; drift should be ~1e-16)")


if __name__ == "__main__":
    main()