// config_io.cpp
// Responsibility:
//   Implement JSON-based configuration input/output for simulation runs.
//
//   This file should read a JSON config file, validate the requested settings,
//   and convert them into the project's C++ configuration objects.
//
//   Expected configuration categories include:
//     - grid and domain settings,
//     - heat-equation parameters,
//     - output-time settings,
//     - initial-condition type and parameters,
//     - output-folder / writer settings,
//     - optional diagnostics or validation flags.
//
//   This module is the bridge between human-editable config files and the
//   strongly typed C++ objects used by the solver pipeline.