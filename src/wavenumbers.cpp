#include "wavenumbers.hpp"
// wavenumbers.cpp
// Responsibility:
//   Implementation of discrete Fourier wave-number utilities.
// What to do here:
//   - Construct the ordered frequency vector consistent with your FFT output.
//   - Build k^2 = kx^2 + ky^2 on the tensor-product grid.
//   - Comment the scaling clearly so future you does not forget it.

RealVec build_fourier_wavenumbers(std::size_t n, Real domain_length) {

    if(n < 2) throw std::invalid_argument("build_fourier_wavenumbers: number of grid points must be at least 2");

    if(!std::isfinite(domain_length)) throw std::invalid_argument("build_fourier_wavenumbers: domain length must be finite");

    if(domain_length <= 0) throw std::invalid_argument("build_fourier_wavenumbers: domain length must be positive");

    Real multiplier = Real{2} * PI / domain_length;

    RealVec fourier_wavenumbers(n);

    for(std::size_t i = 0; i < n; ++i){

        if(i < (n + 1) / 2) fourier_wavenumbers[i] = multiplier * static_cast<Real>(i);
        else fourier_wavenumbers[i] = multiplier * static_cast<Real>(static_cast<std::ptrdiff_t>(i) - static_cast<std::ptrdiff_t>(n));
    }

    return fourier_wavenumbers;
}

Grid2D<Real> build_squared_wavenumber_grid(const RealVec& kx, const RealVec& ky) {

    if(kx.empty()) throw std::invalid_argument("build_squared_wavenumber_grid: kx must not be empty");

    if(ky.empty()) throw std::invalid_argument("build_squared_wavenumber_grid: ky must not be empty");

    std::size_t kxn = kx.size();
    std::size_t kyn = ky.size();

    Grid2D<Real> wavenumber_grid(kxn, kyn);

    for(std::size_t i = 0; i < kxn; ++i){

        Real x_part = kx[i] * kx[i];

        for(std::size_t j = 0; j < kyn; ++j){

            Real y_part = ky[j] * ky[j];

            wavenumber_grid(i,j) = x_part + y_part;
        }
    }

    return wavenumber_grid;
}

