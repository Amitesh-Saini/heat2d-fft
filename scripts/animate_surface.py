"""
animate_surface.py

Responsibility:
    Animate a heat2d run as an evolving 3D surface: the plate in x and y with
    temperature as height. The companion of plot_surface.py in the same way
    animate_heat.py is the companion of plot_snapshot.py -- one frame of this
    animation is one figure from that script.

Performance:
    matplotlib cannot update a 3D surface in place (there is no set_array
    equivalent for Poly3DCollection), so every frame clears the axes and
    rebuilds the surface from scratch. That makes 3D animation roughly an
    order of magnitude slower to encode than the 2D version. The default
    rendering resolution is therefore LOWER here than in plot_surface.py
    (100 points per axis vs 150): motion reads fine at the coarser density,
    and encoding stays tolerable. Raise --points-per-axis for a final
    high-quality render if wanted.

Fixed scales:
    z-limits and the colour scale are computed once over every snapshot and
    held for all frames, so the decaying bump genuinely shrinks instead of
    being rescaled to fill the axes each frame.

Usage:
    python scripts/animate_surface.py output/data/heat_gaussian.h5
    python scripts/animate_surface.py output/data/heat_hot_square.h5 --fps 20
    python scripts/animate_surface.py output/data/heat_gaussian.h5 --gif
    python scripts/animate_surface.py output/data/heat_gaussian.h5 --show

Run from the project root so relative paths resolve.

Note:
    mp4 output requires ffmpeg on PATH (brew install ffmpeg). If ffmpeg is
    not available, pass --gif to fall back to Pillow.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat2d_io import load_run, colormap_for, provenance_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate a heat2d run as an evolving 3D surface.")

    parser.add_argument("path", help="path to the HDF5 run file")

    parser.add_argument("--points-per-axis", type=int, default=100,
                        help="target rendering resolution per axis after "
                             "downsampling (default 100; lower than "
                             "plot_surface.py because every frame is a full "
                             "rebuild)")
    parser.add_argument("--stride", type=int, default=None,
                        help="explicit stride (overrides --points-per-axis); "
                             "plots every Nth point")
    parser.add_argument("--elev", type=float, default=30.0,
                        help="camera elevation angle in degrees (default 30)")
    parser.add_argument("--azim", type=float, default=-60.0,
                        help="camera azimuth angle in degrees (default -60)")
    parser.add_argument("--fps", type=int, default=15,
                        help="frames per second in the output video")
    parser.add_argument("--dpi", type=int, default=120,
                        help="resolution of the rendered frames")
    parser.add_argument("--gif", action="store_true",
                        help="write a GIF via Pillow instead of an mp4 via ffmpeg")
    parser.add_argument("--outdir", default="output/figures",
                        help="base directory for the saved animation")
    parser.add_argument("--show", action="store_true",
                        help="play the animation interactively instead of saving")

    return parser.parse_args()


def main():
    args = parse_args()

    run = load_run(args.path)

    if run.nt < 2:
        sys.exit(f"run has only {run.nt} snapshot(s); nothing to animate")

    # --- downsample for rendering -------------------------------------------
    if args.stride is not None:
        stride_x = stride_y = max(1, args.stride)
    else:
        target = max(2, args.points_per_axis)
        stride_x = max(1, int(np.ceil(run.nx / target)))
        stride_y = max(1, int(np.ceil(run.ny / target)))

    x = run.x[::stride_x]
    y = run.y[::stride_y]
    X, Y = np.meshgrid(x, y)

    # for_plotting gives (ny, nx) with rows = y, so row stride is the y stride.
    def frame_field(index):
        return run.for_plotting(index)[::stride_y, ::stride_x]

    # --- fixed scales, from the whole run -----------------------------------
    cmap, vmin, vmax = colormap_for(run)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")

    # The colorbar is created ONCE from a standalone mappable: the surface
    # artist is destroyed and rebuilt every frame, so it cannot anchor the
    # colorbar without stacking a new bar per frame. The limits never change,
    # so a fixed mappable is exactly right.
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_clim(vmin, vmax)
    colorbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    colorbar.set_label("temperature  u(x, y, t)")

    fig.text(0.01, 0.01, provenance_label(run), fontsize=7, color="0.4")

    box_aspect = (run.Lx, run.Ly, 0.5 * max(run.Lx, run.Ly))

    def draw_frame(index):
        # No in-place update exists for 3D surfaces: clear and rebuild, then
        # re-apply everything the clear wiped (limits, labels, camera, title).
        ax.clear()

        ax.plot_surface(
            X, Y, frame_field(index),
            cmap=cmap, vmin=vmin, vmax=vmax,
            linewidth=0, antialiased=False,
            rstride=1, cstride=1,
        )

        ax.set_zlim(vmin, vmax)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_box_aspect(box_aspect)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        ax.set_title(
            f"{run.ic_type}   t = {run.times[index]:.4f}   "
            f"(alpha = {run.alpha:g},  frame {index + 1}/{run.nt})")

        return ()

    animation = FuncAnimation(
        fig, draw_frame, frames=run.nt, interval=1000 / args.fps, blit=False,
    )

    if args.show:
        plt.show()
        plt.close(fig)
        return

    stem = os.path.splitext(os.path.basename(args.path))[0]

    outdir = os.path.join(args.outdir, run.ic_type, stem)
    os.makedirs(outdir, exist_ok=True)

    if args.gif:
        out_path = os.path.join(outdir, "surface_animation.gif")
        writer = PillowWriter(fps=args.fps)
    else:
        out_path = os.path.join(outdir, "surface_animation.mp4")
        writer = FFMpegWriter(fps=args.fps, bitrate=2400)

    print(f"encoding {run.nt} frames at {len(x)}x{len(y)} rendered "
          "resolution -- 3D redraws are slow, expect this to take a while...")

    try:
        animation.save(out_path, writer=writer, dpi=args.dpi)
    except (FileNotFoundError, RuntimeError) as error:
        sys.exit(
            f"could not write {out_path}: {error}\n"
            "if this is an ffmpeg problem, either install it "
            "(brew install ffmpeg) or re-run with --gif")

    print(f"wrote {out_path}   ({run.nt} frames, {args.fps} fps, "
          f"rendered {len(x)}x{len(y)} of {run.nx}x{run.ny})")

    plt.close(fig)


if __name__ == "__main__":
    main()