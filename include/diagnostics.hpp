// diagnostics.hpp
// Responsibility:
//   Declare diagnostic utilities for analyzing heat-equation snapshots.
//
//   Diagnostics are small scalar quantities computed from each temperature
//   field, such as:
//     - mean temperature,
//     - L2 norm,
//     - minimum temperature,
//     - maximum temperature.
//
//   These values are useful for checking numerical and physical behavior.
//   For the periodic heat equation, the mean temperature should remain
//   constant, while the L2 norm should decrease over time.
//
//   Diagnostics are separate from the solver so that the solver only computes
//   snapshots and does not become responsible for analysis or output.