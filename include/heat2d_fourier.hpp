#pragma once
// heat2d_fourier.hpp
// Responsibility:
//   Public interface for a 2D Fourier spectral solver for the heat equation.
// PDE solved:
//     u_t = alpha * (u_xx + u_yy)
// on a rectangular uniform grid.
// Method:
//   - Sample the initial temperature field in physical space.
//   - Transform it to Fourier space with a 2D FFT.
//   - Evolve each Fourier mode exactly via
//         exp(-alpha * (kx^2 + ky^2) * t).
//   - Inverse transform the result back to physical space.

#include "grid2d.hpp"
#include "types.hpp"
#include <vector>

struct Heat2DConfig {
    std::size_t grid_size = 256;

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

    // Sets the initial temperature field in physical space.
    void set_initial_condition(const Grid2D<Real>& initial_temperature);

    // Solves the heat equation and produces the requested output snapshots.
    void solve();

private:
    Heat2DConfig config_{};

    // Initial temperature field sampled on the physical grid.
    Grid2D<Real> initial_temperature_field_{};

    // Fourier coefficients of the initial temperature field.
    Grid2D<Complex> initial_spectral_coefficients_{};
};