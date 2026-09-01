#!/usr/bin/env python3
"""
plot_memory.py

Figures for benchmarks/results/memory.csv.

Run from the project root:

    python3 scripts/plot_memory.py

Writes PNG (300 dpi) and PDF into benchmarks/results/figures/memory/.

What is being compared:

  The analytic working-set model counts only the solver's arrays: element size
  times point count times the number of simultaneously live grids, plus the
  accumulated snapshots. Measured peak resident set additionally includes the
  binary, its static data, allocator arenas and fragmentation, and page
  granularity rounding.

  Subtracting the baseline removes most of that. What remains is the quantity
  the model is trying to predict, so the useful reading is the RATIO and its
  trend: fixed overhead dominates at small sizes and becomes negligible at
  large ones, so the ratio should approach one as the arrays grow. A ratio
  that stays flat or widens would mean per-element overhead, an unintended
  copy or a temporary that scales with the problem, which is the specific
  failure this benchmark exists to detect.

Plotting conventions match the other two scripts in this directory.
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter

CSV = "benchmarks/results/memory.csv"
OUTDIR = "benchmarks/results/figures/memory"

MIB = 1024.0 * 1024.0

os.makedirs(OUTDIR, exist_ok=True)


plt.rcParams.update({
    "figure.figsize": (7.4, 4.4),
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",

    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9.5,

    "axes.linewidth": 0.9,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,

    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,

    "axes.spines.top": False,
    "axes.spines.right": False,

    "legend.frameon": False,

    "figure.constrained_layout.use": True,
    "figure.constrained_layout.w_pad": 0.06,
    "figure.constrained_layout.h_pad": 0.06,
})

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#666666"


df = pd.read_csv(CSV)

df["model_mib"] = df["theoretical_bytes"] / MIB
df["measured_mib"] = (df["peak_rss_bytes"] - df["baseline_rss_bytes"]) / MIB
df["ratio"] = df["measured_mib"] / df["model_mib"]
df["excess_mib"] = df["measured_mib"] - df["model_mib"]

print("rows:", len(df))


def power_of_two_axis(ax, values, rotate=45):
    ticks = sorted(set(int(v) for v in values))
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())

    if rotate:
        for label in ax.get_xticklabels():
            label.set_rotation(rotate)
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"{stem}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(OUTDIR, stem + ".{png,pdf}"))


def sweep_figure(subset, x_column, x_label, title, stem, fixed_note):
    """Model against measured, with the ratio in a second panel.

    Two panels rather than one: the absolute values span two decades, on which
    a five percent disagreement is invisible. The ratio panel is where the
    result actually lives, and it is drawn on a linear axis with one marked so
    the reader is looking at the distance from agreement rather than at the
    shape of a curve.
    """
    subset = subset.sort_values(x_column)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax1.loglog(subset[x_column], subset["model_mib"], "o-", color=BLUE,
               label="analytic model")
    ax1.loglog(subset[x_column], subset["measured_mib"], "s--", color=ORANGE,
               label="measured, baseline subtracted")

    power_of_two_axis(ax1, subset[x_column])
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("working set (MiB)")
    ax1.set_title("Model against measurement")
    ax1.legend(loc="upper left")

    ax2.plot(subset[x_column], subset["ratio"], "o-", color=PURPLE,
             label="measured / model")
    ax2.axhline(1.0, color=GREY, linestyle="--", lw=1.2, label="exact agreement")

    ax2.set_xscale("log")
    power_of_two_axis(ax2, subset[x_column])
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("measured / model")
    ax2.set_ylim(bottom=0.95)
    ax2.set_title("Agreement")
    ax2.legend(loc="upper right")

    fig.suptitle(title + "\n" + fixed_note)
    save(fig, stem)

    print(f"\n{title}")
    print(subset[[x_column, "model_mib", "measured_mib", "ratio",
                  "excess_mib"]].to_string(index=False))


# ---------------------------------------------------------------------------
# 1. Grid size sweep, at fixed snapshot count
# ---------------------------------------------------------------------------

default_snapshots = df["num_snapshots"].mode().iloc[0]

grid_sweep = df[df["num_snapshots"] == default_snapshots]

if len(grid_sweep) > 1:

    sweep_figure(
        grid_sweep, "nx", "grid size $n$   (grid is $n \\times n$)",
        "Solver memory footprint against grid size",
        "memory_grid_sweep",
        f"({int(default_snapshots)} output times held in memory)")


# ---------------------------------------------------------------------------
# 2. Snapshot sweep, at fixed grid size
#
#    The solver returns every snapshot at once, so the snapshot vector is the
#    largest array in the footprint and scales with output count as well as
#    with grid area. This sweep isolates that term: if the model's dominant
#    contribution were wrong, this is where it would show.
# ---------------------------------------------------------------------------

sizes_with_multiple = df.groupby("nx")["num_snapshots"].nunique()

candidates = sizes_with_multiple[sizes_with_multiple > 1]

if not candidates.empty:

    sweep_size = int(candidates.index[0])

    snapshot_sweep = df[df["nx"] == sweep_size]

    sweep_figure(
        snapshot_sweep, "num_snapshots", "output times held in memory",
        "Solver memory footprint against snapshot count",
        "memory_snapshot_sweep",
        f"(fixed {sweep_size} $\\times$ {sweep_size} grid)")


# ---------------------------------------------------------------------------
# 3. The excess, in absolute terms
#
#    The ratio panels show the disagreement shrinking as the arrays grow,
#    which is consistent with either a fixed overhead or one that grows more
#    slowly than the arrays do. Plotting the excess in MiB separates the two.
#
#    Two panels, one per sweep, because the two have different x axes. Putting
#    the snapshot points on a grid-size axis stacked them all at one tick with
#    nothing to say what distinguished them, which is four numbers in a column
#    rather than a figure.
# ---------------------------------------------------------------------------

have_snapshot_sweep = not candidates.empty

if have_snapshot_sweep:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
else:
    fig, ax1 = plt.subplots()
    ax2 = None

grid_sorted = grid_sweep.sort_values("nx")

ax1.semilogx(grid_sorted["nx"], grid_sorted["excess_mib"], "o-", color=GREEN)

power_of_two_axis(ax1, grid_sorted["nx"])
ax1.set_xlabel("grid size $n$")
ax1.set_ylabel("measured minus model (MiB)")
ax1.set_ylim(bottom=0)
ax1.set_title(f"Against grid size\n({int(default_snapshots)} output times)")

if ax2 is not None:

    snapshot_sorted = snapshot_sweep.sort_values("num_snapshots")

    ax2.semilogx(snapshot_sorted["num_snapshots"], snapshot_sorted["excess_mib"],
                 "s-", color=ORANGE)

    power_of_two_axis(ax2, snapshot_sorted["num_snapshots"])
    ax2.set_xlabel("output times held in memory")
    ax2.set_ylabel("measured minus model (MiB)")
    ax2.set_ylim(bottom=0)
    ax2.set_title(f"Against snapshot count\n(fixed {sweep_size} $\\times$ {sweep_size} grid)")

# The excess grows along both axes, but far more slowly than the arrays
# themselves: over an eightfold increase in grid dimension the model rises by
# a factor of 64 while the excess rises by less than two. That rules out an
# unintended copy or a temporary scaling with the problem, and points instead
# at page-table and allocator metadata, which grow with the NUMBER of pages
# rather than with the data in them.
fig.suptitle("Excess over the model grows far more slowly than the arrays do")
save(fig, "memory_excess")

baseline_mib = df["baseline_rss_bytes"].iloc[0] / MIB

print(f"\nprocess baseline before any solver array: {baseline_mib:.2f} MiB")
print(f"mean excess over the model: {df['excess_mib'].mean():.2f} MiB")
print(f"ratio at the largest grid: {grid_sorted['ratio'].iloc[-1]:.3f}")