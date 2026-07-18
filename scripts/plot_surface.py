"""
plot_surface.py

Responsibility:
    Render one snapshot of a heat2d run as a 3D surface: x and y are the plate
    coordinates and the surface height is the temperature u(x, y, t). For
    localized initial conditions (gaussian, hot_square) this is the most
    intuitive view of diffusion -- the bump visibly slumps and spreads.

    Defaults to the FIRST snapshot (unlike plot_snapshot.py, which defaults to
    the last): the initial bump or box is the dramatic 3D shape, while the
    final diffused state is a nearly flat sheet.

Downsampling:
    matplotlib's plot_surface degrades badly above roughly 200x200 quads, and
    project grids go up to 4096^2. The field is therefore automatically
    downsampled to about --points-per-axis points per axis (default 150) by
    striding; this changes only the rendering density, never the data.

Fixed scales:
    The z-axis limits and colour scale are computed once from ALL snapshots,
    so surfaces at different times are directly comparable: a decaying bump
    genuinely shrinks instead of being rescaled to fill the axes.

Usage:
    python scripts/plot_surface.py output/data/heat_gaussian.h5
    python scripts/plot_surface.py output/data/heat_hot_square.h5 --time 0.005
    python scripts/plot_surface.py output/data/heat_gaussian.h5 --index 30 --show
    python scripts/plot_surface.py output/data/heat_gaussian.h5 --elev 45 --azim 30

Run from the project root so relative paths resolve.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat2d_io import load_run, colormap_for, provenance_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one snapshot of a heat2d run as a 3D surface.")

    parser.add_argument("path", help="path to the HDF5 run file")

    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--time", type=float,
                          help="physical time; the nearest snapshot is used")
    selector.add_argument("--index", type=int,
                          help="snapshot index (0-based)")

    parser.add_argument("--points-per-axis", type=int, default=150,
                        help="target rendering resolution per axis after "
                             "downsampling (default 150)")
    parser.add_argument("--stride", type=int, default=None,
                        help="explicit stride (overrides --points-per-axis); "
                             "plots every Nth point")
    parser.add_argument("--elev", type=float, default=30.0,
                        help="camera elevation angle in degrees (default 30)")
    parser.add_argument("--azim", type=float, default=-60.0,
                        help="camera azimuth angle in degrees (default -60)")
    parser.add_argument("--outdir", default="output/figures",
                        help="base directory for the saved figure")
    parser.add_argument("--show", action="store_true",
                        help="display the figure interactively as well as saving")

    return parser.parse_args()


def main():
    args = parse_args()

    run = load_run(args.path)

    # Snapshot selection: same interface as plot_snapshot.py, but the DEFAULT
    # here is the first frame -- the initial shape is the interesting surface.
    if args.index is not None:
        if not (0 <= args.index < run.nt):
            sys.exit(f"index {args.index} out of range (run has {run.nt} snapshots)")
        index = args.index
    elif args.time is not None:
        index = run.index_at_time(args.time)
    else:
        index = 0

    time = float(run.times[index])

    # --- downsample for rendering -------------------------------------------
    # An explicit --stride wins if given; otherwise a stride is chosen so each
    # axis ends up near the target point count. ceil-style division keeps the
    # stride at least 1 and never overshoots the target; a 4096 grid with
    # target 150 gets stride 28 -> 147 points.
    if args.stride is not None:
        stride_x = stride_y = max(1, args.stride)
    else:
        target = max(2, args.points_per_axis)
        stride_x = max(1, int(np.ceil(run.nx / target)))
        stride_y = max(1, int(np.ceil(run.ny / target)))

    x = run.x[::stride_x]
    y = run.y[::stride_y]

    # for_plotting gives (ny, nx) with rows = y, so row stride is the y stride.
    field = run.for_plotting(index)[::stride_y, ::stride_x]

    X, Y = np.meshgrid(x, y)

    # --- fixed scales, from the whole run -----------------------------------
    cmap, vmin, vmax = colormap_for(run)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")

    surface = ax.plot_surface(
        X, Y, field,
        cmap=cmap, vmin=vmin, vmax=vmax,
        linewidth=0, antialiased=False,   # both matter for speed at 150x150
        rstride=1, cstride=1,
    )

    ax.set_zlim(vmin, vmax)
    ax.view_init(elev=args.elev, azim=args.azim)

    # Box aspect: base proportional to the physical domain so a 2:1 plate
    # renders 2:1, with the height axis kept at a readable fraction.
    ax.set_box_aspect((run.Lx, run.Ly, 0.5 * max(run.Lx, run.Ly)))

    colorbar = fig.colorbar(surface, ax=ax, shrink=0.6, pad=0.1)
    colorbar.set_label("temperature  u(x, y, t)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title(f"{run.ic_type}   t = {time:.4f}   (alpha = {run.alpha:g})")

    fig.text(0.01, 0.01, provenance_label(run), fontsize=7, color="0.4")

    # --- save, filed by IC type, matching the other plotting scripts --------
    stem = os.path.splitext(os.path.basename(args.path))[0]

    outdir = os.path.join(args.outdir, run.ic_type, stem)
    os.makedirs(outdir, exist_ok=True)

    out_path = os.path.join(outdir, f"surface_{index:04d}.png")

    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}   (snapshot {index}, t = {time:.6f}, "
          f"rendered {field.shape[1]}x{field.shape[0]} of {run.nx}x{run.ny})")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()