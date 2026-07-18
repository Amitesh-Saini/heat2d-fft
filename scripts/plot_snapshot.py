#!/usr/bin/env python3
"""
plot_snapshot.py

Responsibility:
    Render one snapshot of a heat2d run as a 2D heatmap. This is the quick-check
    tool and the source of static figures for the README; it is also one frame of
    what animate_heat.py produces, so the rendering logic here is the reference
    the animation reuses.

Usage:
    python scripts/plot_snapshot.py output/data/heat_gaussian.h5
    python scripts/plot_snapshot.py output/data/heat_gaussian.h5 --time 0.1
    python scripts/plot_snapshot.py output/data/heat_gaussian.h5 --index 0 --show

Run from the project root so relative paths resolve.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat2d_io import load_run, colormap_for, provenance_label


def parse_args():
    parser = argparse.ArgumentParser(description="Plot one snapshot of a heat2d run.")

    parser.add_argument("path", help="path to the HDF5 run file")

    # A snapshot is selected either by physical time or by index, not both.
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--time", type=float,
                          help="physical time; the nearest snapshot is used")
    selector.add_argument("--index", type=int,
                          help="snapshot index (0-based)")

    parser.add_argument("--outdir", default="output/figures",
                        help="directory for the saved figure")
    parser.add_argument("--show", action="store_true",
                        help="display the figure interactively as well as saving it")

    return parser.parse_args()


def main():
    args = parse_args()

    run = load_run(args.path)

    # Resolve which snapshot to draw. Default to the final state, which is the
    # most informative single frame for a diffusion run.
    if args.index is not None:
        if not (0 <= args.index < run.nt):
            sys.exit(f"index {args.index} out of range (run has {run.nt} snapshots)")
        index = args.index
    elif args.time is not None:
        index = run.index_at_time(args.time)
    else:
        index = run.nt - 1

    time = float(run.times[index])

    # Colour scale is chosen from the WHOLE run, not this one frame, so a figure
    # made from snapshot k is directly comparable to one made from snapshot j.
    cmap, vmin, vmax = colormap_for(run)

    fig, ax = plt.subplots(figsize=(7, 6))

    # pcolormesh takes the physical coordinate vectors, so the axes are in
    # physical units automatically. for_plotting() supplies the (ny, nx)
    # transpose matplotlib expects.
    mesh = ax.pcolormesh(
        run.x, run.y, run.for_plotting(index),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("temperature  u(x, y, t)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")   # a square domain must render square

    ax.set_title(f"{run.ic_type}   t = {time:.4f}   (alpha = {run.alpha:g})")

    # Stamp the provenance so any figure is traceable to the code that made it.
    fig.text(0.01, 0.01, provenance_label(run), fontsize=7, color="0.4")

    fig.tight_layout()

    stem = os.path.splitext(os.path.basename(args.path))[0]

    outdir = os.path.join(args.outdir, run.ic_type, stem)
    os.makedirs(outdir, exist_ok=True)

    out_path = os.path.join(outdir, f"snapshot_{index:04d}.png")

    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}   (snapshot {index}, t = {time:.6f})")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()