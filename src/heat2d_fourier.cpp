#include "heat2d_fourier.hpp"
#include "fft2d.hpp"
#include "wavenumbers.hpp"
// heat2d_fourier.cpp
// Responsibility:
//   Implementation of the 2D Fourier spectral heat solver.
// What to do here:
//   - Convert the initial condition to complex form.
//   - Compute the forward 2D FFT.
//   - Apply exact modal heat decay for each requested time.
//   - Inverse transform and store/export physical snapshots.

Heat2DFourierSolver::Heat2DFourierSolver(const Heat2DConfig& config) : config_(config) {}

void Heat2DFourierSolver::set_initial_condition(const Grid2D<Real>& u0) {
    physical_initial_ = u0; // TODO: validate shape and copy into solver state.
}

void Heat2DFourierSolver::solve() {
    // TODO: implement the full spectral solve pipeline.
}

