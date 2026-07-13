"""
heat2d_io.py

Responsibility:
    Single point of truth for reading heat2d HDF5 run files. Every plotting
    script loads through here, so the file layout is described in exactly one
    place and the scripts contain no h5py boilerplate.

File layout (written by SnapshotWriter):
    /x                    x-coordinates, shape (nx,)
    /y                    y-coordinates, shape (ny,)
    /times                snapshot times, shape (nt,)
    /snapshots            fields, shape (nt, nx, ny)
    /config_json          verbatim input JSON text
    /diagnostics/{time, mean, l2_norm, min, max}   (if diagnostics enabled)
    /error/relative_l2    analytic error (Fourier-mode validation runs only)
    root attributes       git_commit, compiler, compiler_flags, build_type,
                          timestamp_utc, fft_backend, wall_time_seconds

Array-order convention:
    /snapshots is stored C-order as (nt, nx, ny), matching the C++ Grid2D
    indexing grid(i, j) -> u(x_i, y_j). Matplotlib expects rows to be y and
    columns to be x, so plotting code must transpose a snapshot: u[t].T.
    Do NOT transpose on load -- the stored layout intentionally mirrors the
    solver, and the transpose belongs at the plotting boundary. Use the
    `for_plotting` helper below rather than scattering .T calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import h5py
import numpy as np


@dataclass
class HeatRun:
    """One complete simulation run loaded from an HDF5 file."""

    # Coordinates and time
    x: np.ndarray               # (nx,)  physical x-coordinates
    y: np.ndarray               # (ny,)  physical y-coordinates
    times: np.ndarray           # (nt,)  snapshot times

    # Field data, shape (nt, nx, ny). Index as snapshots[t][i][j] == u(x_i, y_j).
    snapshots: np.ndarray

    # Run identity
    config: dict                # parsed /config_json
    attrs: dict                 # root attributes (provenance, wall time)

    # Optional per-snapshot diagnostics; empty dict if the run disabled them.
    diagnostics: dict = field(default_factory=dict)

    # Optional analytic relative L2 error; None unless the run was a
    # Fourier-mode validation with compute_analytic_error enabled.
    relative_l2_error: Optional[np.ndarray] = None

    # --- convenience -------------------------------------------------------

    @property
    def nt(self) -> int:
        return self.snapshots.shape[0]

    @property
    def nx(self) -> int:
        return self.snapshots.shape[1]

    @property
    def ny(self) -> int:
        return self.snapshots.shape[2]

    @property
    def Lx(self) -> float:
        return float(self.config["solver"]["Lx"])

    @property
    def Ly(self) -> float:
        return float(self.config["solver"]["Ly"])

    @property
    def alpha(self) -> float:
        return float(self.config["solver"]["alpha"])

    @property
    def ic_type(self) -> str:
        return str(self.config["initial_condition"]["type"])

    @property
    def git_commit(self) -> str:
        return str(self.attrs.get("git_commit", "unknown"))

    @property
    def extent(self) -> tuple:
        """(left, right, bottom, top) for imshow, in physical coordinates."""
        return (self.x[0], self.x[-1], self.y[0], self.y[-1])

    def for_plotting(self, index: int) -> np.ndarray:
        """
        Snapshot `index` transposed for matplotlib: shape (ny, nx), so rows are
        y and columns are x. This is the ONLY place the transpose happens.
        """
        return self.snapshots[index].T

    def index_at_time(self, t: float) -> int:
        """Index of the snapshot whose time is closest to `t`."""
        return int(np.argmin(np.abs(self.times - t)))

    def color_limits(self) -> tuple:
        """
        Global (vmin, vmax) across ALL snapshots.

        A fixed color scale is mandatory for animations: if each frame
        autoscales, a decaying field renders its peak as the brightest colour
        in every frame and the animation appears static, hiding the very
        physics it is meant to show.
        """
        return (float(self.snapshots.min()), float(self.snapshots.max()))
    

    def is_signed(self) -> bool:
        
        """
        True if the field takes meaningfully negative values, which decides the
        colormap family.

        The threshold matters: an FFT round-trip leaves floating-point dust of
        order 1e-16 in a field that is mathematically non-negative (a gaussian's
        tails), so a naive `min() < 0` test would misclassify every gaussian run
        as signed and waste half the colour range on temperatures that do not
        exist. Compare against a small fraction of the field's own magnitude
        instead of against exact zero.
        """
        scale = max(abs(float(self.snapshots.min())), abs(float(self.snapshots.max())))
        if scale == 0.0:
            return False
        return bool(float(self.snapshots.min()) < -1e-9 * scale)


def load_run(path: str) -> HeatRun:
    """
    Reads a heat2d HDF5 output file into a HeatRun.

    Everything is read eagerly into memory. For the grid sizes this project
    targets that is fine (256^2 x 50 snapshots is ~26 MB), but note that a
    4096^2 run with many snapshots would not fit comfortably -- in that case
    read /snapshots lazily by index instead of through this function.
    """
    with h5py.File(path, "r") as f:

        x = f["/x"][:]
        y = f["/y"][:]
        times = f["/times"][:]
        snapshots = f["/snapshots"][:]

        # Root attributes: decode any bytes to str so callers get clean values.
        attrs = {}
        for key, value in f.attrs.items():
            attrs[key] = value.decode() if isinstance(value, bytes) else value

        # The verbatim config, which makes the file self-describing.
        raw_config = f["/config_json"][()]
        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode()
        config = json.loads(raw_config)

        # Diagnostics are optional (the run may have disabled them).
        diagnostics = {}
        if "diagnostics" in f:
            for key in f["/diagnostics"]:
                diagnostics[key] = f[f"/diagnostics/{key}"][:]

        # Analytic error only exists for Fourier-mode validation runs.
        relative_l2_error = None
        if "error" in f and "relative_l2" in f["/error"]:
            relative_l2_error = f["/error/relative_l2"][:]

    return HeatRun(
        x=x,
        y=y,
        times=times,
        snapshots=snapshots,
        config=config,
        attrs=attrs,
        diagnostics=diagnostics,
        relative_l2_error=relative_l2_error,
    )


def colormap_for(run: HeatRun) -> tuple:
    """
    Returns (cmap_name, vmin, vmax) appropriate for this run's data.

    Non-negative fields get a perceptually uniform SEQUENTIAL map: `inferno`
    reads intuitively as heat and, unlike rainbow maps such as `jet`, equal
    colour steps correspond to equal temperature steps, so it does not
    manufacture visual features that are not in the data.

    Signed fields get a DIVERGING map centred on zero, with symmetric limits so
    that white sits exactly at u = 0 and the sign of a deviation is readable at
    a glance.

    In both cases the limits are computed once over ALL snapshots, so colours
    mean the same thing in every frame of an animation.
    """
    vmin, vmax = run.color_limits()

    if run.is_signed():
        bound = max(abs(vmin), abs(vmax))
        return ("RdBu_r", -bound, bound)

    return ("inferno", vmin, vmax)


def provenance_label(run: HeatRun) -> str:
    """
    A one-line stamp for figure corners, so any plot is traceable back to the
    exact code and configuration that produced it.
    """
    commit = run.git_commit
    short = commit[:8] if commit != "unknown" else commit
    return f"{run.ic_type}  |  {run.nx}x{run.ny}  |  commit {short}"