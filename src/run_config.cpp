// run_config.cpp
// Responsibility:
//   Implement helper functionality for the high-level RunConfig structure.
//
//   This file should contain logic related to complete simulation-run settings,
//   not the numerical heat solver itself.
//
//   Examples of responsibilities that belong here:
//     - default values for run-level settings,
//     - validation of run-level options,
//     - helper functions for interpreting initial-condition choices,
//     - helper functions for output/run naming.
//
//   The goal is to keep main.cpp small and prevent Heat2DConfig from being
//   overloaded with non-solver responsibilities.