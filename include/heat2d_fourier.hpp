// heat2d_fourier.hpp
// Responsibility:
//   Declare the 2D Fourier spectral solver for the periodic heat equation.
//
//   The solver evolves a real-valued initial temperature field u0(x,y) on
//   the periodic rectangular domain
//       [-Lx/2, Lx/2) x [-Ly/2, Ly/2)
//   using a 2D FFT spectral method.
//
// Method:
//   - Convert the real initial grid to complex physical-space data.
//   - Apply the forward 2D FFT.
//   - Build physical angular wavenumbers Kx and Ky.
//   - Evolve each Fourier coefficient by
//         exp(-alpha * (Kx^2 + Ky^2) * t).
//   - Apply the inverse 2D FFT to recover physical-space snapshots.

#pragma once

#include <cstddef>
#include <vector>

#include "grid2d.hpp"
#include "types.hpp"


struct Heat2DConfig{

    std::size_t nx = 256;
    std::size_t ny = 256;

    Real Lx = Real{2.0};
    Real Ly = Real{2.0};

    Real alpha = Real{1.0};

    std::vector<Real> output_times = {
        Real{0.0},
        Real{0.05},
        Real{0.10},
        Real{0.50},
        Real{1.00}
    };
};


class Heat2DFourierSolver {
    
public:
    explicit Heat2DFourierSolver(const Heat2DConfig& config);

    void set_initial_condition(const Grid2D<Real>& initial_temperature);

    std::vector<Grid2D<Real>> solve();

    const Heat2DConfig& config() const;

private:
    Heat2DConfig config_;
    Grid2D<Real> initial_temperature_field_;
    bool has_initial_condition_ = false;

    void validate_config() const;
    void validate_initial_condition_shape(const Grid2D<Real>& initial_temperature) const;

    Grid2D<Real> make_snapshot_at_time(
        const Grid2D<Complex>& initial_spectral_coefficients, const Grid2D<Real>& squared_wavenumbers, Real time) const;
};