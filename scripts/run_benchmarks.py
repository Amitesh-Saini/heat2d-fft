#!/usr/bin/env python3
"""
run_benchmarks.py

Drives the whole benchmark suite: one command produces a complete result set.

Run from the project root:

    python3 scripts/run_benchmarks.py

    python3 scripts/run_benchmarks.py --only memory
    python3 scripts/run_benchmarks.py --skip transforms

Why a script rather than one executable:

  Peak resident set size is a whole-process high-water mark that never
  decreases, so the memory benchmark has to be invoked once per configuration.
  No amount of looping inside a single binary can produce that: a process
  sweeping several sizes would report the largest one on every row.

  Once something has to sit above the binaries, it is also the natural place
  for everything else that is session-scoped rather than measurement-scoped:
  generating the run identifier that ties the output files together, clearing
  stale results, and pausing between sweeps.

What it does NOT do:

  It does not configure or build. A benchmark run should measure a binary that
  already exists rather than one it just produced, so that what was measured
  is whatever the developer last chose to build. The script checks the
  binaries are present and refuses to run if they are not.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

BUILD_DIR = "build"
RESULTS_DIR = "benchmarks/results"

# The memory benchmark sweeps two axes. Grid size at fixed snapshot count is
# the headline comparison against the analytic working-set model; snapshot
# count at fixed grid size isolates the model's dominant term, since the
# solver holds every snapshot in memory at once and that vector is the largest
# array in the footprint.
MEMORY_SIZES = [128, 256, 512, 1024]
MEMORY_SNAPSHOTS_DEFAULT = 10

MEMORY_SNAPSHOT_SWEEP_SIZE = 512
MEMORY_SNAPSHOT_SWEEP = [5, 20, 40]

# Pause between sweeps, not between measurements. The individual benchmarks
# handle their own repetition and their spreads are already under a few
# percent; this is only to keep one sweep's tail from overlapping the next
# one's start.
SWEEP_PAUSE_SECONDS = 5


def make_run_id():
    """Timestamp, short git hash, and hostname, joined into one identifier.

    ISO timestamp first so directory listings sort chronologically, and UTC so
    runs from machines in different zones stay orderable.

    The hash carries a -dirty suffix when the working tree has uncommitted
    changes. Without it the identifier would silently claim a commit that does
    not contain the code that was measured, which is the most common way a
    benchmark result becomes unreproducible.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()

        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL).decode().strip()

        if dirty:
            commit += "-dirty"

    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "nogit"

    host = os.uname().nodename.split(".")[0]

    return f"{stamp}_{commit}_{host}"


def binary(name):
    return os.path.join(BUILD_DIR, name)


def require_binaries(names):
    missing = [n for n in names if not os.path.isfile(binary(n))]

    if missing:
        print("missing binaries: " + ", ".join(missing), file=sys.stderr)
        print("build them first:  cmake --build build", file=sys.stderr)
        sys.exit(1)


def run(command):
    """Runs a command, streaming its output, and stops the session on failure.

    A benchmark that aborts has usually failed a correctness check, which
    means the binary computes the wrong answer. Continuing on to the next
    sweep would produce a result set that is partly trustworthy and partly
    not, with nothing in the files to say which is which.
    """
    print("  $ " + " ".join(command))

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\ncommand failed with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def run_transforms(run_id):
    print("\n=== transforms ===")
    require_binaries(["bench_transforms"])
    run([binary("bench_transforms"), run_id])


def run_solver(run_id):
    print("\n=== solver ===")
    require_binaries(["bench_solver"])
    run([binary("bench_solver"), run_id])


def run_memory(run_id):
    print("\n=== memory ===")
    require_binaries(["bench_memory"])

    # Removed once, before the first invocation. The writer opens in append
    # mode so that each of the invocations below adds a row to a file the
    # first one created; without this the rows would accumulate across
    # sessions.
    csv_path = os.path.join(RESULTS_DIR, "memory.csv")

    if os.path.exists(csv_path):
        os.remove(csv_path)

    for size in MEMORY_SIZES:
        run([binary("bench_memory"), str(size),
             str(MEMORY_SNAPSHOTS_DEFAULT), run_id])

    for snapshots in MEMORY_SNAPSHOT_SWEEP:
        run([binary("bench_memory"), str(MEMORY_SNAPSHOT_SWEEP_SIZE),
             str(snapshots), run_id])


def main():
    parser = argparse.ArgumentParser(
        description="Run the heat2d benchmark suite.")

    parser.add_argument("--only", choices=["transforms", "solver", "memory"],
                        help="run just one sweep")

    parser.add_argument("--skip", choices=["transforms", "solver", "memory"],
                        action="append", default=[],
                        help="skip a sweep; may be given more than once")

    parser.add_argument("--run-id",
                        help="override the generated run identifier")

    parser.add_argument("--plot", action="store_true",
                        help="regenerate the figures after the sweeps finish")

    args = parser.parse_args()

    if not os.path.isdir(BUILD_DIR):
        print(f"no {BUILD_DIR}/ directory; configure and build first",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    run_id = args.run_id or make_run_id()

    print(f"run_id: {run_id}")
    print(f"results: {RESULTS_DIR}")

    # A note worth reading before every session. Closing background
    # applications took the solver's spread at the largest grid from 55
    # percent to under one percent, which was a larger effect than any
    # methodological choice in the suite. The conditions of a measurement are
    # part of the measurement.
    print("\nfor clean numbers: no other applications running, on mains power,")
    print("external displays disconnected. Recorded in the README.")

    sweeps = [("transforms", run_transforms),
              ("solver", run_solver),
              ("memory", run_memory)]

    if args.only:
        sweeps = [s for s in sweeps if s[0] == args.only]

    sweeps = [s for s in sweeps if s[0] not in args.skip]

    if not sweeps:
        print("nothing to run", file=sys.stderr)
        sys.exit(1)

    started = time.time()

    for index, (name, function) in enumerate(sweeps):

        function(run_id)

        if index + 1 < len(sweeps):
            time.sleep(SWEEP_PAUSE_SECONDS)

    elapsed = time.time() - started

    print(f"\ndone in {elapsed / 60:.1f} min")

    # Opt in rather than automatic. Plotting is far cheaper than measuring,
    # and figures get regenerated repeatedly while a script is being adjusted;
    # coupling the two would mean either rerunning a twenty-minute sweep to
    # change an axis label, or never being able to run one without also
    # overwriting the figures.
    if args.plot:

        print("\n=== figures ===")

        scripts = [("transforms", "scripts/plot_transforms.py"),
                   ("solver", "scripts/plot_solver.py"),
                   ("memory", "scripts/plot_memory.py")]

        ran = {name for name, _ in sweeps}

        for name, path in scripts:

            if name not in ran:
                continue

            if not os.path.isfile(path):
                print(f"  {path} not found, skipping")
                continue

            run([sys.executable, path])

    else:

        print("\nplot with:")
        print("  python3 scripts/plot_transforms.py")
        print("  python3 scripts/plot_solver.py")
        print("  python3 scripts/plot_memory.py")
        print("\nor rerun with --plot")


if __name__ == "__main__":
    main()