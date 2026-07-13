#!/usr/bin/env python3
"""
Verifies probe.h5 written by the Hdf5File scratch probe in main.cpp.

Checks, in order of what they isolate:
  1. Dataset shape is (nt, nx, ny) = (2, 4, 6), NOT transposed to (2, 6, 4).
  2. Values land where the C++ layout convention says they should:
     f["/snapshots"][0][i][j] == i + 10*j, matching Grid2D's grid(i, j).
     This is asymmetric, so a transposed write cannot pass.
  3. Snapshot 1 differs from snapshot 0, proving the hyperslab offset in
     append_snapshot advanced instead of overwriting slab 0.
  4. The 1-D vectors and the variable-length string round-trip correctly.
"""

import sys
import numpy as np
import h5py

NX, NY, NT = 4, 6, 2
failures = []

with h5py.File("probe.h5", "r") as f:

    # --- 1. shape -----------------------------------------------------------
    shape = f["/snapshots"].shape
    print(f"/snapshots shape: {shape}   (expected ({NT}, {NX}, {NY}))")
    if shape != (NT, NX, NY):
        failures.append(f"shape is {shape}, expected {(NT, NX, NY)} "
                        f"-- axes are transposed or the dataset did not grow")

    # --- 2. layout convention ----------------------------------------------
    snap0 = f["/snapshots"][0]
    expected0 = np.fromfunction(lambda i, j: i + 10 * j, (NX, NY), dtype=float)

    if snap0.shape == expected0.shape and np.allclose(snap0, expected0):
        print("layout OK: snapshots[0][i][j] == i + 10*j")
    else:
        failures.append("layout MISMATCH -- values are not where the C++ "
                        "(i, j) indexing says they should be")
        print("got:\n", snap0)
        print("expected:\n", expected0)

    # Spot check the single most diagnostic element.
    if snap0.shape == (NX, NY):
        print(f"snapshots[0][3][5] = {snap0[3][5]}   (expected 53.0)")

    # --- 3. hyperslab offset advanced --------------------------------------
    snap1 = f["/snapshots"][1]
    if np.allclose(snap0, snap1):
        failures.append("snapshot 1 is identical to snapshot 0 -- the hyperslab "
                        "offset never advanced, so appends overwrote slab 0")
    else:
        print("offset OK: snapshot 1 differs from snapshot 0")

    expected1 = expected0 + 1000.0
    if not np.allclose(snap1, expected1):
        failures.append("snapshot 1 values are wrong")

    # --- 4. vectors, string, attributes ------------------------------------
    print("/x      :", f["/x"][:])
    print("/y      :", f["/y"][:])
    print("/times  :", f["/times"][:])

    if len(f["/x"]) != NX:
        failures.append(f"/x has length {len(f['/x'])}, expected {NX}")
    if len(f["/y"]) != NY:
        failures.append(f"/y has length {len(f['/y'])}, expected {NY}")

    cfg = f["/config_json"][()]
    if isinstance(cfg, bytes):
        cfg = cfg.decode()
    print("/config_json:", cfg)
    if "probe" not in cfg:
        failures.append("/config_json did not round-trip")

    print("attrs   :", dict(f.attrs))
    if "git_commit" not in f.attrs:
        failures.append("root attribute git_commit missing")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for msg in failures:
        print(" -", msg)
    sys.exit(1)

print("all probe checks passed")