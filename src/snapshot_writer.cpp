// snapshot_writer.cpp
// Responsibility:
//   Implement high-level output writing for complete heat-equation runs.
//
//   This file should create the output directory for a run and save:
//     - the configuration that produced the run,
//     - the output times,
//     - the 2D field snapshots,
//     - scalar diagnostics,
//     - optional validation/error data.
//
//   The writer may use lower-level modules such as:
//     - npy_writer for full 2D field snapshots,
//     - io_csv for small scalar tables,
//     - diagnostics for computing mean, L2 norm, min, and max.
//
//   The solver should not write files directly. It should return snapshots,
//   and this module should decide how those snapshots are stored.