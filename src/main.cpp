#include "config.hpp"
#include "heat2d_fourier.hpp"
// main.cpp
// Responsibility:
//   Thin program entry point for a standard solver run.
// What to do here:
//   - Build the default configuration.
//   - Create/select an initial condition.
//   - Run the solver.
//   - Write outputs and print a short summary.
//   - Do not place heavy numerical logic here.

Heat2DConfig make_default_heat2d_config() {
    Heat2DConfig cfg;
    cfg.grid_size = 256;
    cfg.alpha = 1.0;
    cfg.output_times = {0.0, 0.05, 0.1, 0.5, 1.0};
    return cfg;
}

int main() {
    const auto cfg = make_default_heat2d_config();
    Heat2DFourierSolver solver(cfg);
    solver.solve();
    return 0;
}

