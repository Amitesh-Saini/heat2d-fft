#pragma once
// heat2d_fourier.hpp
// Responsibility:
//   Main 2D Fourier spectral heat-equation solver interface.
// What to do here:
//   - Build the physical grid.
//   - Compute the forward FFT of the initial condition.
//   - Evolve Fourier coefficients exactly in time using exp(-alpha * k^2 * t).
//   - Inverse transform snapshots back to physical space.
//   - Provide hooks for CSV output and error computation.

#include "grid2d.hpp"
#include "types.hpp"
#include <vector>

struct Heat2DConfig {
    std::size_t n = 256;
    Real x_min = -1.0;
    Real x_max = 1.0;
    Real y_min = -1.0;
    Real y_max = 1.0;
    Real alpha = 1.0;
    std::vector<Real> output_times{};
};

class Heat2DFourierSolver {
public:
    explicit Heat2DFourierSolver(const Heat2DConfig& config);

    void set_initial_condition(const Grid2D<Real>& u0);
    void solve();

private:
    Heat2DConfig config_{};
    Grid2D<Real> physical_initial_{};
    Grid2D<Complex> spectral_initial_{};
};

