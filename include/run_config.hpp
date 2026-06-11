// run_config.hpp
// Responsibility:
//   Define the high-level configuration for a complete heat-equation run.
//
//   Heat2DConfig stores only the numerical solver settings needed by the
//   Fourier heat solver: grid size, domain size, diffusivity, and output times.
//
//   RunConfig stores the larger program-level settings needed by main.cpp,
//   including:
//     - the solver configuration,
//     - the initial-condition type and parameters,
//     - output-directory settings,
//     - optional diagnostics / validation settings.
//
//   This separation keeps the solver configuration clean while still allowing
//   the executable to describe a full reproducible simulation run.