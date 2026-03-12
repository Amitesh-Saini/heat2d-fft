#pragma once
// wavenumbers.hpp
// Responsibility:
//   Generate discrete Fourier wave numbers and squared-mode grids.
// What to do here:
//   - Build 1D k-vectors for the periodic domain [-1, 1].
//   - Build a 2D k^2 grid for the heat-equation decay factor.
//   - Keep scaling and ordering consistent with your FFT convention.

#include "grid2d.hpp"
#include "types.hpp"

RealVec build_wavenumbers(std::size_t n, Real domain_length);
Grid2D<Real> build_k2_grid(const RealVec& kx, const RealVec& ky);

