#pragma once
// wavenumbers.hpp
// Responsibility:
//   Builds the discrete Fourier wave numbers used by the spectral heat solver.
// Notes:
//   - The 1D wave-number vector must match the ordering used by the FFT.
//   - The squared wave-number grid stores kx(i)^2 + ky(j)^2 at each mode.
//   - These quantities are used in the exact modal decay factor
//         exp(-alpha * (kx^2 + ky^2) * t).

#include "grid2d.hpp"
#include "types.hpp"

// Builds the 1D Fourier wave-number vector associated with an FFT grid of size n
// over a domain of length domain_length.
RealVec build_fourier_wavenumbers(std::size_t n, Real domain_length);

// Builds the 2D grid of squared wave-number magnitudes:
//     k_squared(i,j) = kx(i)^2 + ky(j)^2.
Grid2D<Real> build_squared_wavenumber_grid(const RealVec& kx,
                                           const RealVec& ky);