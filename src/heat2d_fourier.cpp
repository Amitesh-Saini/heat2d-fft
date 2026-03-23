#include "heat2d_fourier.hpp"
#include "fft2d.hpp"
#include "wavenumbers.hpp"

// heat2d_fourier.cpp
// Responsibility:
//   Implementation of the 2D Fourier spectral solver for the 2D heat equation.
// Method:
//   - Store the initial temperature field in physical space.
//   - Convert it to complex form and transform it with a 2D FFT.
//   - Evolve each Fourier mode by exp(-alpha * (kx^2 + ky^2) * t).
//   - Inverse transform to recover physical-space temperature snapshots.

Heat2DFourierSolver::Heat2DFourierSolver(const Heat2DConfig& config)
    : config_(config) {}

void Heat2DFourierSolver::set_initial_condition(const Grid2D<Real>& initial_temperature) {
    initial_temperature_field_ = initial_temperature;
    // TODO: validate that the input grid matches config_.grid_size x config_.grid_size.
}

void Heat2DFourierSolver::solve() {
    // TODO:
    // 1. Copy initial_temperature_field_ into a complex-valued grid.
    // 2. Apply the forward 2D FFT to obtain initial_spectral_coefficients_.
    // 3. Build the Fourier wave numbers and the squared wave-number grid.
    // 4. For each output time:
    //      a) copy initial_spectral_coefficients_
    //      b) apply modal decay exp(-alpha * (kx^2 + ky^2) * t)
    //      c) inverse FFT back to physical space
    //      d) save or export the snapshot
}