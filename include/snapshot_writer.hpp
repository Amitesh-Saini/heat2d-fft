// snapshot_writer.hpp
// Responsibility:
//   Declare the high-level output writer for complete simulation runs.
//
//   The snapshot writer owns the structure of an output run directory. It should
//   save all data needed to inspect, reproduce, and visualize a simulation.
//
//   A typical run directory may contain:
//     - config.json,
//     - times.csv,
//     - snapshot_0000.npy,
//     - snapshot_0001.npy,
//     - diagnostics.csv,
//     - optional validation/error files.
//
//   This module keeps output organization separate from main.cpp and from the
//   numerical solver.