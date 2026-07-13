#!/usr/bin/env python3
"""
animate_heat.py

Responsibility:
    Animate the full time evolution of a heat2d run as a 2D heatmap. This is the
    project's headline visual: it shows heat spreading and decaying on the
    periodic domain.

    The rendering matches plot_snapshot.py exactly -- one frame of this animation
    is one figure from that script -- so the two stay visually consistent.

Fixed colour scale:
    vmin/vmax are computed ONCE over every snapshot and held for all frames. This
    is not a stylistic choice. If each frame autoscaled to its own range, a
    decaying field would render its peak as the brightest colour in every frame
    and the animation would appear static, hiding the very physics it exists to
    show. A fixed scale means "dimmer" genuinely means "cooler".

Usage:
    python scripts/animate_heat.py output/data/heat_gaussian.h5
    python scripts/animate_heat.py output/data/heat_gaussian.h5 --fps 20 --gif
    python scripts/animate_heat.py output/data/heat_gaussian.h5 --show

Run from the project root so relative paths resolve.

Note:
    mp4 output requires ffmpeg on PATH (brew install ffmpeg). If ffmpeg is not
    available, pass --gif to fall back to Pillow, which needs no external tool
    but produces files roughly 10-50x larger.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat2d_io import load_run, colormap_for, provenance_label


def parse_args():
    parser = argparse.ArgumentParser(description="Animate a heat2d run.")

    parser.add_argument("path", help="path to the HDF5 run file")

    parser.add_argument("--fps", type=int, default=15,
                        help="frames per second in the output video")
    parser.add_argument("--dpi", type=int, default=120,
                        help="resolution of the rendered frames")
    parser.add_argument("--gif", action="store_true",
                        help="write a GIF via Pillow instead of an mp4 via ffmpeg")
    parser.add_argument("--outdir", default="output/figures",
                        help="directory for the saved animation")
    parser.add_argument("--show", action="store_true",
                        help="play the animation interactively instead of saving")

    return parser.parse_args()


def main():
    args = parse_args()

    run = load_run(args.path)

    if run.nt < 2:
        sys.exit(f"run has only {run.nt} snapshot(s); nothing to animate")

    # Chosen once, from the whole run. See the note in the module docstring.
    cmap, vmin, vmax = colormap_for(run)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw frame 0, then reuse this mesh for every subsequent frame: creating a
    # new pcolormesh per frame would be far slower and would leak artists.
    mesh = ax.pcolormesh(
        run.x, run.y, run.for_plotting(0),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("temperature  u(x, y, t)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    title = ax.set_title("")
    fig.text(0.01, 0.01, provenance_label(run), fontsize=7, color="0.4")

    fig.tight_layout()

    def draw_frame(index):
        # set_array wants the flattened field for "auto"/"gouraud" shading.
        mesh.set_array(run.for_plotting(index).ravel())
        title.set_text(
            f"{run.ic_type}   t = {run.times[index]:.4f}   "
            f"(alpha = {run.alpha:g},  frame {index + 1}/{run.nt})"
        )
        return (mesh, title)

    animation = FuncAnimation(
        fig, draw_frame, frames=run.nt, interval=1000 / args.fps, blit=False,
    )

    if args.show:
        plt.show()
        plt.close(fig)
        return

    # Figures are filed by initial-condition type, read from the run file
    # itself, so a run's output always lands in the right folder without the
    # caller having to specify it.
    outdir = os.path.join(args.outdir, run.ic_type)
    os.makedirs(outdir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.path))[0]

    if args.gif:
        out_path = os.path.join(outdir, f"{stem}_animation.gif")
        writer = PillowWriter(fps=args.fps)
    else:
        out_path = os.path.join(outdir, f"{stem}_animation.mp4")
        writer = FFMpegWriter(fps=args.fps, bitrate=2400)

    try:
        animation.save(out_path, writer=writer, dpi=args.dpi)
    except (FileNotFoundError, RuntimeError) as error:
        # The usual cause is ffmpeg not being installed.
        sys.exit(
            f"could not write {out_path}: {error}\n"
            "if this is an ffmpeg problem, either install it "
            "(brew install ffmpeg) or re-run with --gif"
        )

    print(f"wrote {out_path}   ({run.nt} frames, {args.fps} fps)")

    plt.close(fig)


if __name__ == "__main__":
    main()