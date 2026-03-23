#include "wavenumbers.hpp"
// wavenumbers.cpp
// Responsibility:
//   Implementation of discrete Fourier wave-number utilities.
// What to do here:
//   - Construct the ordered frequency vector consistent with your FFT output.
//   - Build k^2 = kx^2 + ky^2 on the tensor-product grid.
//   - Comment the scaling clearly so future you does not forget it.

RealVec build_fourier_wavenumbers(std::size_t n, Real domain_length) {
    (void)n;
    (void)domain_length;
    return {}; // TODO: build frequency vector for periodic domain.
}

Grid2D<Real> build_squared_wavenumber_grid(const RealVec& kx, const RealVec& ky) {
    Grid2D<Real> k2(kx.size(), ky.size());
    // TODO: fill with kx[i]^2 + ky[j]^2.
    return k2;
}

