// test_time_grid.cpp
// Responsibility:
//   Unit tests for the output-time grid utilities.
//
//   The heat solver expects a valid vector of output times:
//     - nonempty,
//     - finite,
//     - nonnegative,
//     - sorted,
//     - usually including t = 0 and the requested final time.
//
//   The time_grid module is responsible for converting high-level run settings,
//   such as final_time and num_snapshots, into the concrete output_times vector
//   used by Heat2DConfig.
//
//   These tests verify that time-grid construction behaves correctly before the
//   values are passed into the solver.