#include "initial_conditions.hpp"
// initial_conditions.cpp
// Responsibility:
//   Implement concrete initial-condition generators.
// What to do here:
//   - Start with a single exact Fourier mode for validation.
//   - Add a multi-mode case to verify independent decay.
//   - Add Gaussian and hot-square cases for plots.

Grid2D<Real> make_single_fourier_mode_ic(const RealVec& x, const RealVec& y) {
    return Grid2D<Real>(x.size(), y.size()); // TODO: fill with exact periodic mode.
}

Grid2D<Real> make_multi_fourier_mode_ic(const RealVec& x, const RealVec& y) {
    return Grid2D<Real>(x.size(), y.size()); // TODO: fill with sum of periodic modes.
}

Grid2D<Real> make_gaussian_ic(const RealVec& x, const RealVec& y) {
    return Grid2D<Real>(x.size(), y.size()); // TODO: fill with Gaussian bump.
}

Grid2D<Real> make_hot_square_ic(const RealVec& x, const RealVec& y) {
    return Grid2D<Real>(x.size(), y.size()); // TODO: fill with centered hot square.
}

