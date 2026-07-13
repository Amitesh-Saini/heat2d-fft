#pragma once
// initial_conditions.hpp
// Responsibility:
//   Declares user-selectable initial-condition generators for the 2D periodic
//   FFT heat-equation solver.
//
//   Each function returns a fully populated physical-space Grid2D<Real>
//   sampled on the periodic rectangular domain
//       [-Lx/2, Lx/2) x [-Ly/2, Ly/2)
//   with nx points in x and ny points in y.
//
// Notes:
//   - The periodic endpoint is excluded. The grid points are
//         x_i = -Lx/2 + i * Lx/nx,  i = 0, ..., nx - 1
//         y_j = -Ly/2 + j * Ly/ny,  j = 0, ..., ny - 1.
//   - Fourier-mode initial conditions use integer wavenumbers so they are
//     exactly compatible with the periodic FFT grid.
//   - For a Fourier mode with integer wavevector (kx, ky), the heat equation
//     decay factor is
//         exp(-alpha * ((2*pi*kx/Lx)^2 + (2*pi*ky/Ly)^2) * t).
//   - The IC layer validates physical/domain parameters. FFT-specific
//     restrictions, such as radix-2 power-of-two sizes, belong in the FFT or
//     solver layer.


#include <optional>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cstddef>

#include "grid2d.hpp"
#include "types.hpp"
#include "validation_config.hpp"

// Describes one real plane-wave Fourier mode:
//
//     amplitude * cos(2*pi*(kx*x/Lx + ky*y/Ly) + phase)
//
// where kx and ky are integer wavenumbers and phase is measured in radians.
// This representation is FFT-native: each mode corresponds to a wavevector
// (kx, ky) and decays independently under the heat equation.

struct FourierMode2D {
    std::ptrdiff_t kx;
    std::ptrdiff_t ky;
    Real amplitude;
    Real phase;
};


// Validates the physical grid specification shared by all initial-condition
// generators.
//
// This checks only domain/grid sanity:
//   - nx and ny must be at least 2
//   - Lx and Ly must be finite
//   - Lx and Ly must satisfy the configured minimum domain length
//   - dx = Lx/nx and dy = Ly/ny must satisfy the configured minimum spacing
//
// This function does not check FFT-backend restrictions such as power-of-two
// sizes; those belong in the FFT or solver layer.
void validate_grid_spec_2d(Real Lx, Real Ly, std::size_t nx, std::size_t ny, const ValidationConfig& validation = ValidationConfig{});


// Returns a small default set of low-frequency plane-wave Fourier modes for
// multi-mode demos or for the case where the user selects a multi-mode
// Fourier initial condition without manually specifying modes.
//
// The default modes should remain low-frequency so that they are well resolved
// and do not immediately decay away under heat evolution.
std::vector<FourierMode2D> make_default_fourier_modes();


// Builds the periodic coordinate vector for one axis:
//
//     x_i = -L/2 + i * L/n,   i = 0, ..., n-1
//
// The periodic endpoint is excluded: the grid spans [-L/2, L/2) with spacing
// L/n, so the last point sits one step short of +L/2. This is the same
// convention every initial-condition generator uses internally; exposing it
// here lets the output writer store the physical coordinates alongside the
// field data.
//
// Requires n >= 2 and L finite and positive; throws std::invalid_argument
// otherwise.
RealVec make_periodic_coordinates(Real L, std::size_t n);


// Builds a truncated periodic image-sum Gaussian on the periodic rectangular
// domain [-Lx/2, Lx/2) x [-Ly/2, Ly/2).
//
// The returned field approximates the periodized Gaussian
//
//     u(x,y) = A * sum_m sum_n
//              exp(-((x + mLx)^2 + (y + nLy)^2) / sigma^2),
//
// where the infinite image sum is truncated to
//     m = -image_radius_x, ..., image_radius_x
//     n = -image_radius_y, ..., image_radius_y.
//
// This is more compatible with a periodic FFT spectral solver than a single
// non-periodic Gaussian, because the image sum approximates the periodic
// extension of a localized Gaussian bump.
//
// Defaults:
//   - sigma = 0.10 * min(Lx, Ly)
//   - image_radius_x = image_radius_y = 1
//
// The image radius is capped to avoid accidentally constructing an expensive
// initial condition with too many Gaussian image copies.
Grid2D<Real> make_gaussian_ic(
    Real Lx, Real Ly, std::size_t nx, std::size_t ny, Real amplitude = 1.0, 
    std::optional<Real> sigma = std::nullopt, std::size_t image_radius_x = 1, std::size_t image_radius_y = 1, const ValidationConfig& validation = ValidationConfig{});


// Builds a tanh-smoothed hot square/rectangle on the periodic rectangular
// domain [-Lx/2, Lx/2) x [-Ly/2, Ly/2).
//
// The field has the form
//
//     u(x,y) = A * Bx(x) * By(y),
//
// where Bx and By are smooth box functions:
//
//     Bx(x) = 0.5 * [tanh((x + wx/2)/eps_x)
//                    - tanh((x - wx/2)/eps_x)],
//
//     By(y) = 0.5 * [tanh((y + wy/2)/eps_y)
//                    - tanh((y - wy/2)/eps_y)].
//
// Here wx and wy are the full hot-region widths, while eps_x and eps_y are
// smoothing widths controlling the edge transition thickness.
//
// Defaults:
//   - width_x = 0.20 * Lx
//   - width_y = 0.20 * Ly
//   - smooth_width_x = min(3*dx, 0.10*width_x)
//   - smooth_width_y = min(3*dy, 0.10*width_y)
//
// This is smoother and more FFT-friendly than a discontinuous square pulse,
// while still giving a localized square-like hot region.
Grid2D<Real> make_hot_square_ic(
    Real Lx, Real Ly, std::size_t nx, std::size_t ny, Real amplitude = 1.0, 
    std::optional<Real> width_x = std::nullopt, std::optional<Real> width_y = std::nullopt, 
    std::optional<Real> smooth_width_x = std::nullopt, std::optional<Real> smooth_width_y = std::nullopt, 
    const ValidationConfig& validation = ValidationConfig{});



// Builds a uniform initial temperature field on the periodic rectangular
// domain [-Lx/2, Lx/2) x [-Ly/2, Ly/2).
//
// Every grid point is set to the constant value T0. Under the periodic heat
// equation, this field remains constant because its Laplacian is zero.
Grid2D<Real> make_constant_ic(
    Real Lx, Real Ly, std::size_t nx, std::size_t ny, Real T0 = Real{0.5}, 
    const ValidationConfig& validation = ValidationConfig{});


// Builds a single real plane-wave Fourier mode on the periodic rectangular
// domain [-Lx/2, Lx/2) x [-Ly/2, Ly/2).
//
// The field is
//
//     u(x,y) = A * cos(2*pi*(kx*x/Lx + ky*y/Ly) + phase),
//
// where kx and ky are integer wavenumbers and phase is measured in radians.
//
// Integer wavenumbers make the mode exactly compatible with the periodic FFT
// grid. Under the heat equation, this mode decays by
//
//     exp(-alpha * ((2*pi*kx/Lx)^2 + (2*pi*ky/Ly)^2) * t).
Grid2D<Real> make_custom_single_fourier_mode_ic(
    Real Lx, Real Ly, std::size_t nx, std::size_t ny, std::ptrdiff_t kx = 1, std::ptrdiff_t ky = 1, 
    Real amplitude = Real{1}, Real phase = Real{0}, const ValidationConfig& validation = ValidationConfig{});



// Builds a sum of real plane-wave Fourier modes on the periodic rectangular
// domain [-Lx/2, Lx/2) x [-Ly/2, Ly/2).
//
// Each FourierMode2D entry contributes one term:
//
//     A_m * cos(2*pi*(kx_m*x/Lx + ky_m*y/Ly) + phase_m).
//
// The returned field is the sum of all mode contributions. Each integer
// wavevector (kx_m, ky_m) is checked against the discrete Nyquist range.
// Under the heat equation, each mode decays independently with factor
//
//     exp(-alpha * ((2*pi*kx_m/Lx)^2 + (2*pi*ky_m/Ly)^2) * t).
//
// This provides a controlled multi-frequency initial condition for checking
// independent modal decay and for producing richer periodic demo fields.
Grid2D<Real> make_custom_multi_fourier_mode_ic(
    Real Lx, Real Ly, std::size_t nx, std::size_t ny, const std::vector<FourierMode2D>& modes, 
    const ValidationConfig& validation = ValidationConfig{});