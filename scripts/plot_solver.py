#!/usr/bin/env python3
"""
plot_solver.py

Figures for benchmarks/results/solver_timings.csv.

Run from the project root:

    python3 scripts/plot_solver.py

Writes PNG (300 dpi) and PDF into benchmarks/results/figures/solver/.

Plotting conventions match scripts/plot_transforms.py:

  - Global rcParams rather than per-call arguments, so every figure across
    both scripts shares one visual identity.
  - Series distinguished by BOTH colour and line style, since colour alone
    fails in greyscale print and for readers with colour vision deficiency.
  - Major gridlines only. On a log axis the minor gridlines sit at 2,3,4...
    within each decade, so their spacing is uneven by construction.
  - Explicit ticks at the measured sizes, all powers of two, rotated so the
    labels clear each other.
  - Reference slopes FITTED rather than anchored at one point.
  - Benchmark enum strings mapped to readable display names.

Reporting note:
  Medians across trials, with min and max shown where the spread matters.
  The 128 and 256 configurations are short enough that a single scheduling
  interrupt is a large fraction of the measurement, so their spread is much
  wider than the larger grids; the spread figure makes that visible rather
  than hiding it behind a median.
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter

CSV = "benchmarks/results/solver_timings.csv"
OUTDIR = "benchmarks/results/figures/solver"

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

# Okabe-Ito, the standard colour-vision-deficiency-safe qualitative palette.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
GREY = "#666666"

IC_DISPLAY = {
    "multi_fourier_mode": "multi-mode",
    "gaussian": "Gaussian",
    "hot_square": "hot square",
}

PHASE_DISPLAY = {
    "forward_transform_time_ns": "forward FFT",
    "spectral_copy_time_ns": "spectral copy",
    "decay_time_ns": "decay pass",
    "inverse_transform_time_ns": "inverse FFTs",
}

PHASE_COLOUR = {
    "forward_transform_time_ns": BLUE,
    "spectral_copy_time_ns": YELLOW,
    "decay_time_ns": ORANGE,
    "inverse_transform_time_ns": GREEN,
}

PHASES = list(PHASE_DISPLAY)


df = pd.read_csv(CSV)

df["total_ms"] = df["total_time_ns"] / 1e6

print("rows:", len(df))
print("benchmarks:", sorted(df["benchmark"].unique()))


def power_of_two_axis(ax, sizes, rotate=45):
    ticks = sorted(set(int(s) for s in sizes))
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())

    if rotate:
        for label in ax.get_xticklabels():
            label.set_rotation(rotate)
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")


def fit_reference(y, model):
    """Scale a complexity model to sit through the data."""
    model = np.asarray(model, dtype=float)
    ratio = np.median(np.asarray(y, dtype=float) / model)
    return ratio * model


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"{stem}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(OUTDIR, stem + ".{png,pdf}"))


# ---------------------------------------------------------------------------
# 1. Solver scaling, with error bars showing the spread across trials
#
#    The bars are min-to-max rather than a standard deviation: with seven
#    trials the extremes are more informative than a spread statistic, and
#    they make the difference between the noisy small grids and the tight
#    large ones immediately visible.
# ---------------------------------------------------------------------------

ladder = df[df["benchmark"] == "Numeric_solver"]

g = ladder.groupby("nx")["total_ms"].agg(["median", "min", "max"]).reset_index()
g = g.sort_values("nx")

fig, ax = plt.subplots()

lower = g["median"] - g["min"]
upper = g["max"] - g["median"]

ax.errorbar(g["nx"], g["median"], yerr=[lower, upper],
            fmt="o-", color=BLUE, capsize=4, label="measured (median, min-max)")

n = np.asarray(g["nx"], dtype=float)

# The reference is fitted on n >= 256 but drawn across the whole range.
#
# At 128 the solve runs for tens of milliseconds, short enough that a single
# scheduling interrupt is a large fraction of it: the spread there is roughly
# 75 percent against under 1 percent at 1024. Including that point drags the
# fitted constant and makes the reference look systematically low everywhere
# else. Excluding it from the fit while still plotting it lets a reader see
# both the trend and the point that does not follow it.
fit_mask = n >= 256

model = n ** 2 * np.log2(n)

ratio = np.median(np.asarray(g["median"])[fit_mask] / model[fit_mask])

ax.plot(n, ratio * model, ":", color="black", lw=1.2,
        label=r"$O(n^2\log_2 n)$ fitted, $n \geq 256$")

ax.set_xscale("log")
ax.set_yscale("log")

power_of_two_axis(ax, g["nx"])
ax.set_xlabel("grid size $n$   (grid is $n \\times n$)")
ax.set_ylabel("solve time (ms)")
ax.set_title("Solver scaling, 10 output times")
ax.legend(loc="upper left")
save(fig, "solver_scaling")

print("\nsolver scaling (ms):")
print(g.to_string(index=False))

ratios = np.asarray(g["median"])[1:] / np.asarray(g["median"])[:-1]
print("successive ratios:", np.round(ratios, 2).tolist(),
      " (n^2 log n predicts ~4.3 to 4.6)")


# ---------------------------------------------------------------------------
# 2. Phase breakdown as a proportion of the solve
#
#    Fractions rather than absolute times: the absolute values span two
#    decades across the ladder, which would compress everything but the
#    inverse transforms into an invisible sliver. The fractions are the
#    result being reported, namely that the inverse transforms dominate and
#    the decay pass is a rounding error.
#
#    These come from the timing registry annotated inside the solver, so they
#    read as zero in a build without HEAT2D_ENABLE_TIMING.
# ---------------------------------------------------------------------------

if ladder[PHASES].to_numpy().sum() == 0:

    print("\nphase columns are all zero: rebuild with -DHEAT2D_ENABLE_TIMING=ON")

else:

    phase_median = ladder.groupby("nx")[PHASES + ["total_time_ns"]].median().reset_index()
    phase_median = phase_median.sort_values("nx")

    fig, ax = plt.subplots()

    x = np.arange(len(phase_median))
    bottom = np.zeros(len(phase_median))

    for phase in PHASES:

        fraction = 100 * phase_median[phase] / phase_median["total_time_ns"]

        ax.bar(x, fraction, 0.6, bottom=bottom,
               color=PHASE_COLOUR[phase], label=PHASE_DISPLAY[phase])

        bottom += np.asarray(fraction)

    # Whatever the annotated regions do not cover: the wavenumber grid, the
    # physical-to-complex conversion, the real-part extraction after each
    # inverse, and the copies into the snapshot vector. Shown rather than
    # dropped, so the bars sum to the whole solve.
    ax.bar(x, 100 - bottom, 0.6, bottom=bottom,
           color=GREY, label="unannotated remainder")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in phase_median["nx"]])
    ax.set_xlabel("grid size $n$")
    ax.set_ylabel("share of solve time (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Where the solve time goes")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3)
    save(fig, "phase_breakdown")

    print("\nphase share of solve time (%):")

    for _, row in phase_median.iterrows():

        parts = "  ".join(
            f"{PHASE_DISPLAY[p]} {100 * row[p] / row['total_time_ns']:.1f}"
            for p in PHASES)

        print(f"  n={int(row['nx']):5d}  {parts}")


# ---------------------------------------------------------------------------
# 3. Initial-condition comparison
#
#    Solver cost is IC-independent by construction: the same transforms on the
#    same array sizes regardless of what the field contains. This is the
#    measurement that confirms it, and it is what licenses the scaling ladder
#    above to use a single initial condition.
#
#    The min-max bars matter more than the bar heights here. The claim is that
#    the three agree WITHIN the spread, so a reader needs to see the spread.
# ---------------------------------------------------------------------------

ic = df[df["benchmark"] == "Solver_ic_compare"]

if not ic.empty:

    gi = ic.groupby("ic")["total_ms"].agg(["median", "min", "max"]).reset_index()

    # Ordered as declared rather than alphabetically.
    order = [name for name in IC_DISPLAY if name in set(gi["ic"])]
    gi = gi.set_index("ic").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    x = np.arange(len(gi))

    lower = gi["median"] - gi["min"]
    upper = gi["max"] - gi["median"]

    ax.bar(x, gi["median"], 0.5, color=[BLUE, ORANGE, PURPLE][:len(gi)],
           yerr=[lower, upper], capsize=5, error_kw={"ecolor": GREY, "lw": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels([IC_DISPLAY[name] for name in gi["ic"]])
    ax.set_xlim(-0.6, len(gi) - 0.4)
    ax.set_ylabel("solve time (ms)")
    ax.set_title("Solve cost is independent of the initial condition\n"
                 "($512 \\times 512$, bars are median, whiskers min to max)")
    ax.grid(axis="x", visible=False)
    save(fig, "ic_comparison")

    span = 100 * (gi["median"].max() - gi["median"].min()) / gi["median"].median()

    print("\ninitial-condition comparison (ms):")
    print(gi.to_string(index=False))
    print(f"medians span {span:.1f}% of the middle value")


# ---------------------------------------------------------------------------
# 4. Compute against I/O
#
#    Two panels. The left shows compute and I/O side by side per grid size at
#    each compression level; the right shows the on-disk size, which is the
#    other half of what compression trades.
#
#    The I/O bars use the MEDIAN, and the caption has to carry a warning:
#    these are page-cache write-back numbers, not device throughput.
#    H5Fflush pushes HDF5's buffers to the operating system but does not
#    fsync, so a write returns as soon as the data reaches the OS cache. When
#    the cache is already full of dirty pages from an earlier job, the write
#    blocks until the OS drains some of it. That produces two widely separated
#    clusters for the same configuration, so the median lands in one of them
#    rather than describing a typical value.
# ---------------------------------------------------------------------------

io = df[df["benchmark"] == "Full_solver"].copy()

if not io.empty:

    io["io_ms"] = io["io_time_ns"] / 1e6
    io["mb"] = io["bytes_written"] / (1024.0 * 1024.0)

    gio = io.groupby(["nx", "gzip_level"])[["total_ms", "io_ms", "mb"]] \
    .median().reset_index()

    levels = sorted(gio["gzip_level"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    sizes = sorted(gio["nx"].unique())
    x = np.arange(len(sizes))

    bars = [("compute", None, BLUE)] + [(f"I/O, gzip {int(level)}", level, colour)
                                         
    for level, colour in zip(levels, [GREY, ORANGE, GREEN])]

    width = 0.8 / len(bars)

    for index, (label, level, colour) in enumerate(bars):

        offset = (index - (len(bars) - 1) / 2) * width

        if level is None:
            sub = gio[gio["gzip_level"] == levels[0]].set_index("nx")
            values = [sub.loc[n, "total_ms"] for n in sizes]
        else:
            sub = gio[gio["gzip_level"] == level].set_index("nx")
            values = [sub.loc[n, "io_ms"] for n in sizes]

        ax1.bar(x + offset, values, width, color=colour, label=label)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in sizes])
    ax1.set_yscale("log")
    ax1.set_xlabel("grid size $n$")
    ax1.set_ylabel("time (ms)")
    ax1.set_title("Compute against I/O\n(medians across seven trials)")
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper left")

    for level, colour, style in zip(levels, [GREY, ORANGE, GREEN], ["o-", "s--", "^:"]):

        sub = gio[gio["gzip_level"] == level].sort_values("nx")
        ax2.loglog(sub["nx"], sub["mb"], style, color=colour,label=f"gzip {int(level)}")

    power_of_two_axis(ax2, sizes)
    ax2.set_xlabel("grid size $n$")
    ax2.set_ylabel("file size (MiB)")
    ax2.set_title("On-disk size")
    ax2.legend(loc="upper left")

    fig.suptitle("gzip costs ~16x the write time for 1.5x compression, at any level")    
    save(fig, "compute_vs_io")

    print("\ncompute against I/O (median ms) and file size (MiB):")
    print(gio.to_string(index=False))

    # The bimodality that the median hides.
    print("\nI/O time spread per configuration (ms), showing the two modes:")

    for (nx, level), group in io.groupby(["nx", "gzip_level"]):
        values = np.sort(np.asarray(group["io_ms"]))
        print(f"  n={nx:5d} gzip{int(level)}:  " + "  ".join(f"{v:.0f}" for v in values))


# ---------------------------------------------------------------------------
# 5. Analytic error against grid size
#
#    The multi-mode initial condition is band-limited, so the spectral solver
#    represents it exactly and evaluates the decay exactly. The only error is
#    floating-point roundoff, which is why this curve rises rather than
#    falling: it is not a convergence plot. A convergence plot would use an
#    initial condition the grid cannot represent exactly.
# ---------------------------------------------------------------------------

err = ladder[ladder["error"].notna()]

if not err.empty:

    ge = err.groupby("nx")["error"].median().reset_index().sort_values("nx")

    fig, ax = plt.subplots()

    ax.loglog(ge["nx"], ge["error"], "o-", color=BLUE, label="measured")

    eps = np.finfo(np.float64).eps

    ax.axhline(eps, color="black", linestyle="--", lw=1.2, label=r"machine $\varepsilon$")

    ne = np.asarray(ge["nx"], dtype=float)
    ax.loglog(ne, fit_reference(ge["error"], np.log2(ne)),
              ":", color=GREY, lw=1.2, label=r"$\varepsilon\log_2 n$ fitted")

    power_of_two_axis(ax, ge["nx"])
    ax.set_xlabel("grid size $n$")
    ax.set_ylabel(r"relative $L_2$ error vs analytic")
    ax.set_title("Solver accuracy, band-limited initial condition\n"
                 "(roundoff only: the solution is represented exactly)")
    ax.legend(loc="upper left")
    save(fig, "solver_error")

    print("\nanalytic error in units of machine epsilon:")
    print("  " + ", ".join(f"{int(n)}:{v / eps:.0f}"
                           for n, v in zip(ge["nx"], ge["error"])))