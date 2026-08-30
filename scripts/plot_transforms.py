#!/usr/bin/env python3
"""
plot_transforms.py

Figures for benchmarks/transform_timings.csv.

Run from the project root:

    python3 scripts/plot_transforms.py

Writes PNG (300 dpi) and PDF into benchmarks/figures/.

Plotting conventions applied here, from standard practice for scientific
figures:

  - Global rcParams rather than per-call arguments, so every figure in the
    set shares one visual identity.
  - Series are distinguished by BOTH colour and line style. Colour alone
    fails in greyscale print and for readers with colour vision deficiency.
  - Major gridlines only. On a log axis the minor gridlines sit at 2,3,4...
    within each decade, so their spacing is uneven by construction.
  - Explicit ticks at the measured sizes, all powers of two, rotated so the
    five-digit labels at the top of the sweep do not collide.
  - Reference slopes are FITTED, not anchored at one point. Anchoring at the
    smallest size pins the line to whichever point is most contaminated by
    fixed overhead.
  - Ratios rather than two curves whenever the interesting quantity is a
    relative difference.
  - Legends placed outside the axes wherever a series or a reference line
    runs through the corner they would otherwise occupy.
  - Benchmark enum strings mapped to readable display names. The raw
    identifiers belong in the CSV, not in a figure legend.
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

CSV = "benchmarks/results/transform_timings.csv"
OUTDIR = "benchmarks/results/figures/transforms"

EPS = np.finfo(np.float64).eps

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
GREY = "#666666"

# Enum identifiers as written by the C++ side, mapped to figure labels. Update
# the keys here if the enum-to-string table changes; nothing else in this
# script depends on the raw names.
DISPLAY = {
    "FFT_1D_time": "custom FFT (1D)",
    "DFT_1D_time": "naive DFT (1D)",
    "FFTW_1d_time": "FFTW (1D)",
    "FFT_2d_time": "custom FFT (2D)",
    "FFTW_2d_time": "FFTW (2D)",
    "FFT_2d_aspect": "custom FFT (2D, rectangular)",
}


df = pd.read_csv(CSV)

df["time_ns"] = df["total_time_ns"] / df["reps_used"]
df["points"] = df["nx"] * df["ny"]

print("rows:", len(df))
print("benchmarks:", sorted(df["benchmark"].unique()))

missing = set(df["benchmark"].unique()) - set(DISPLAY)
if missing:
    print("warning: no display name for", sorted(missing))


def median_by_size(name, value="time_ns", key="nx"):
    sub = df[df["benchmark"] == name]
    if sub.empty:
        return None
    return sub.groupby(key)[value].median().reset_index().sort_values(key)


def fit_reference(y, model):
    """Scale a complexity model to sit through the data.

    Returns model * median(y / model). The median rather than a least-squares
    fit so one outlying point cannot drag the reference off the rest.
    """
    model = np.asarray(model, dtype=float)
    ratio = np.median(np.asarray(y, dtype=float) / model)
    return ratio * model


def power_of_two_axis(ax, sizes, rotate=45):
    """Tick at the measured sizes rather than at decades.

    Rotated because the sweep now reaches 32768: five-digit labels at a
    45-degree angle clear each other where horizontal ones overlap.
    """
    ticks = sorted(set(int(s) for s in sizes))
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


fft = median_by_size("FFT_1D_time")
dft = median_by_size("DFT_1D_time")
fftw = median_by_size("FFTW_1d_time")
f2 = median_by_size("FFT_2d_time")
w2 = median_by_size("FFTW_2d_time")


# ---------------------------------------------------------------------------
# 1. FFT vs DFT vs FFTW, with fitted reference slopes
#
#    Legend outside the axes: with five series there is no interior corner
#    the curves and their reference lines all avoid.
# ---------------------------------------------------------------------------

fig, ax = plt.subplots()

ax.loglog(dft["nx"], dft["time_ns"], "s-", color=RED, label=DISPLAY["DFT_1D_time"])
ax.loglog(fft["nx"], fft["time_ns"], "o-", color=BLUE, label=DISPLAY["FFT_1D_time"])
ax.loglog(fftw["nx"], fftw["time_ns"], "^-", color=GREEN, label=DISPLAY["FFTW_1d_time"])

nd = np.asarray(dft["nx"], dtype=float)
ax.loglog(nd, fit_reference(dft["time_ns"], nd ** 2),
          "--", color=GREY, lw=1.2, label=r"$O(n^2)$ fitted")

nf = np.asarray(fft["nx"], dtype=float)
ax.loglog(nf, fit_reference(fft["time_ns"], nf * np.log2(nf)),
          ":", color="black", lw=1.2, label=r"$O(n\log_2 n)$ fitted")

power_of_two_axis(ax, fft["nx"])
ax.set_xlabel("transform length $n$")
ax.set_ylabel("time per transform (ns)")
ax.set_title("1D transform scaling")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
save(fig, "1d_fft_vs_dft")

for name, d in [("FFT", fft), ("DFT", dft), ("FFTW", fftw)]:
    slope = np.polyfit(np.log(d["nx"]), np.log(d["time_ns"]), 1)[0]
    print(f"fitted exponent {name:5s}: {slope:.3f}")


# ---------------------------------------------------------------------------
# 2. Normalized time.
#
#    Two linear panels rather than one log axis: on a shared log scale the
#    15x vertical gap flattens FFTW's curve and hides that it varies at all.
#
#    Wider figure and every other tick labelled, since two panels of
#    thirteen five-digit labels do not fit side by side.
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

sizes_1d = sorted(set(int(s) for s in fft["nx"]))
sparse_ticks = sizes_1d[::2] + [sizes_1d[-1]]

for ax, d, colour, style, key in [
        (ax1, fft, BLUE, "o-", "FFT_1D_time"),
        (ax2, fftw, GREEN, "^-", "FFTW_1d_time")]:

    nn = np.asarray(d["nx"], dtype=float)
    ax.semilogx(nn, d["time_ns"] / (nn * np.log2(nn)), style, color=colour)

    power_of_two_axis(ax, sparse_ticks)
    ax.set_xlabel("transform length $n$")
    ax.set_ylim(bottom=0)
    ax.set_title(DISPLAY[key])

ax1.set_ylabel(r"time / $(n\log_2 n)$   (ns)")

fig.suptitle("Cost per butterfly-equivalent operation (note differing y scales)")
save(fig, "1d_normalized")


# ---------------------------------------------------------------------------
# 3. Relative to FFTW.
#
#    Gridlines drawn at round factor values that the percentage axis maps
#    onto, so each horizontal line has a labelled value on both sides.
# ---------------------------------------------------------------------------

fig, ax = plt.subplots()

m1 = fft.merge(fftw, on="nx", suffixes=("_fft", "_fftw"))
m2 = f2.merge(w2, on="nx", suffixes=("_fft", "_fftw"))

fac_1d = m1["time_ns_fft"] / m1["time_ns_fftw"]
fac_2d = m2["time_ns_fft"] / m2["time_ns_fftw"]

ax.semilogx(m1["nx"], fac_1d, "o-", color=BLUE, label="1D")
ax.semilogx(m2["nx"], fac_2d, "s--", color=ORANGE, label="2D")

power_of_two_axis(ax, fft["nx"])
ax.set_xlabel("transform length $n$")
ax.set_ylabel("FFTW faster by this factor")
ax.set_title("Custom FFT relative to FFTW (lower is better)")
ax.set_ylim(bottom=0)
ax.legend(loc="lower left")

factor_ticks = [5, 10, 15, 20, 25, 30, 35, 40]
ax.yaxis.set_major_locator(FixedLocator(factor_ticks))
ax.grid(axis="y", alpha=0.25)

secondary = ax.secondary_yaxis(
    "right",
    functions=(lambda f: 100.0 / np.maximum(f, 1e-9),
               lambda p: 100.0 / np.maximum(p, 1e-9)))
secondary.set_ylabel("custom FFT speed as % of FFTW")
secondary.yaxis.set_major_locator(FixedLocator([100.0 / f for f in factor_ticks]))
secondary.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

save(fig, "fraction_of_fftw")

print("\n1D slowdown factor:", fac_1d.round(1).tolist())
print("2D slowdown factor:", fac_2d.round(1).tolist())


# ---------------------------------------------------------------------------
# 4. 2D scaling
# ---------------------------------------------------------------------------

fig, ax = plt.subplots()

ax.loglog(f2["nx"], f2["time_ns"], "o-", color=BLUE, label=DISPLAY["FFT_2d_time"])
ax.loglog(w2["nx"], w2["time_ns"], "^--", color=GREEN, label=DISPLAY["FFTW_2d_time"])

n2 = np.asarray(f2["nx"], dtype=float)
ax.loglog(n2, fit_reference(f2["time_ns"], n2 ** 2 * np.log2(n2)),
          ":", color="black", lw=1.2, label=r"$O(n^2\log_2 n)$ fitted")

power_of_two_axis(ax, f2["nx"])
ax.set_xlabel("grid size $n$   (grid is $n \\times n$)")
ax.set_ylabel("time per transform (ns)")
ax.set_title("2D transform scaling")
ax.legend(loc="upper left")
save(fig, "2d_scaling")


# ---------------------------------------------------------------------------
# 5. Row vs column pass
# ---------------------------------------------------------------------------

sub = df[df["benchmark"] == "FFT_2d_time"].copy()
sub["row_ns"] = sub["row_time_ns"] / sub["reps_used"]
sub["col_ns"] = sub["col_time_ns"] / sub["reps_used"]

g = sub.groupby("nx")[["row_ns", "col_ns", "time_ns"]].median().reset_index()

fig, ax = plt.subplots()

ax.semilogx(g["nx"], g["col_ns"] / g["row_ns"], "o-", color=PURPLE, label="measured")
ax.axhline(1.0, color=GREY, linestyle="--", lw=1.2, label="equal cost")

power_of_two_axis(ax, g["nx"])
ax.set_xlabel("grid size $n$")
ax.set_ylabel("column pass / row pass")
ax.set_title("Stride penalty in the 2D FFT")
ax.legend(loc="lower right")
save(fig, "row_vs_col")

g["residual_pct"] = 100 * (g["time_ns"] - g["row_ns"] - g["col_ns"]) / g["time_ns"]
print("\nresidual (total - row - col) as % of total:")
print(g[["nx", "residual_pct"]].to_string(index=False))


# ---------------------------------------------------------------------------
# 6. Aspect pair.
#
#    The two totals agree to well within the run-to-run spread, which is the
#    claim: identical flops and identical footprint, so any difference is
#    noise rather than a stride effect. The stride effect is in the passes.
# ---------------------------------------------------------------------------

asp = df[df["benchmark"] == "FFT_2d_aspect"].copy()

if not asp.empty:
    asp["row_ns"] = asp["row_time_ns"] / asp["reps_used"]
    asp["col_ns"] = asp["col_time_ns"] / asp["reps_used"]
    asp["shape"] = asp["nx"].astype(str) + "x" + asp["ny"].astype(str)

    ga = asp.groupby("shape")[["time_ns", "row_ns", "col_ns"]].median()

    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    x = np.arange(len(ga))
    width = 0.26

    ax.bar(x - width, ga["time_ns"] / 1e6, width, color=GREY, label="total")
    ax.bar(x, ga["row_ns"] / 1e6, width, color=BLUE, label="row pass (contiguous)")
    ax.bar(x + width, ga["col_ns"] / 1e6, width, color=RED, label="column pass (strided)")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("x", r" $\times$ ") for s in ga.index])
    ax.set_xlim(-0.55, len(ga) - 0.45)
    ax.set_ylim(0, 1.35 * float(ga["time_ns"].max()) / 1e6)
    ax.set_ylabel("time (ms)")
    ax.set_title("Aspect pair: identical flops, only stride differs\n"
                 "(bars are side by side, not stacked)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3)
    save(fig, "aspect_pair")

    print("\naspect pair medians (ns):")
    print(ga.to_string())


# ---------------------------------------------------------------------------
# 7. Roundoff growth.
#
#    Legend outside the axes: the epsilon line spans the full width at the
#    bottom and the reference slopes cross both lower corners, so there is no
#    interior placement that avoids overlapping something.
# ---------------------------------------------------------------------------

fig, ax = plt.subplots()

for name, colour, style in [("FFT_1D_time", BLUE, "o-"),
                            ("DFT_1D_time", RED, "s-"),
                            ("FFT_2d_time", ORANGE, "^-")]:
    d = median_by_size(name, value="error")
    if d is None:
        continue
    ax.loglog(d["nx"], d["error"], style, color=colour, label=DISPLAY[name])

ax.axhline(EPS, color="black", linestyle="--", lw=1.2, label=r"machine $\varepsilon$")

e = median_by_size("FFT_1D_time", value="error")
ne = np.asarray(e["nx"], dtype=float)

ax.loglog(ne, fit_reference(e["error"], np.log2(ne)),
          ":", color=GREY, lw=1.2, label=r"$\varepsilon\log_2 n$ fitted")
ax.loglog(ne, fit_reference(e["error"], ne),
          "-.", color=GREY, lw=1.2, label=r"$\varepsilon n$ fitted")

power_of_two_axis(ax, e["nx"])
ax.set_xlabel("transform length $n$")
ax.set_ylabel(r"relative $L_\infty$ error vs FFTW")
ax.set_title("Roundoff growth")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
save(fig, "error_growth")

print("\nerror in units of machine epsilon:")
for name in ["FFT_1D_time", "DFT_1D_time", "FFT_2d_time"]:
    d = median_by_size(name, value="error")
    if d is None:
        continue
    pairs = ", ".join(f"{int(n)}:{v / EPS:.0f}" for n, v in zip(d["nx"], d["error"]))
    print(f"  {DISPLAY[name]:28s} {pairs}")


# ---------------------------------------------------------------------------
# 8. Forward error against round-trip error.
#
#    The forward curve is the difference from FFTW and grows linearly with n,
#    tracking eps*n. The round-trip curve is self-contained and should stay
#    far lower: twiddle drift partly cancels between the forward and inverse
#    transforms rather than compounding, so the systematic part of the
#    recurrence's error does not survive the round trip.
#
#    The gap between them is the measurement: it separates the systematic
#    component of the twiddle error from the random one.
# ---------------------------------------------------------------------------
 
fig, ax = plt.subplots()
 
fwd = median_by_size("FFT_1D_time", value="error")
rt = median_by_size("FFT_1D_time", value="roundtrip_error")
 
ax.loglog(fwd["nx"], fwd["error"], "o-", color=BLUE,
          label="forward, vs FFTW")
ax.loglog(rt["nx"], rt["roundtrip_error"], "s-", color=GREEN,
          label="round trip, self-contained")
 
ax.axhline(EPS, color="black", linestyle="--", lw=1.2, label=r"machine $\varepsilon$")
 
nr = np.asarray(rt["nx"], dtype=float)
ax.loglog(nr, fit_reference(rt["roundtrip_error"], np.log2(nr)),
          ":", color=GREY, lw=1.2, label=r"$\varepsilon\log_2 n$ fitted")
 
power_of_two_axis(ax, fwd["nx"])
ax.set_xlabel("transform length $n$")
ax.set_ylabel(r"relative $L_\infty$ error")
ax.set_title("Forward error vs round-trip error (custom FFT, 1D)")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
save(fig, "roundtrip_vs_forward")
 
print("\nround-trip error in units of machine epsilon:")
pairs = ", ".join(f"{int(n)}:{v / EPS:.0f}"
                  for n, v in zip(rt["nx"], rt["roundtrip_error"]))
print(" ", pairs)