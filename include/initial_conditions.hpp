#pragma once
// initial_conditions.hpp
// Responsibility:
//   Generates physical-space initial temperature fields for solver tests
//   and visualization runs.
// Notes:
//   - All functions return a real-valued temperature field sampled on the
//     tensor-product grid defined by x and y.
//   - Fourier-mode test cases are useful for validation because their decay
//     is known analytically.

#include "grid2d.hpp"
#include "types.hpp"

// Builds an initial condition consisting of a single Fourier mode sampled
// on the physical grid defined by x and y.
Grid2D<Real> make_single_fourier_mode_ic(const RealVec& x, const RealVec& y);

// Builds an initial condition consisting of a sum of several Fourier modes
// sampled on the physical grid defined by x and y.
Grid2D<Real> make_multi_fourier_mode_ic(const RealVec& x, const RealVec& y);

// Builds a Gaussian initial temperature field sampled on the physical grid.
Grid2D<Real> make_gaussian_ic(const RealVec& x, const RealVec& y);

// Builds a square hot region (box pulse) as an initial temperature field
// sampled on the physical grid.
Grid2D<Real> make_hot_square_ic(const RealVec& x, const RealVec& y);