#pragma once
// initial_conditions.hpp
// Responsibility:
//   Initial-condition generators for validation and visualization runs.
// What to do here:
//   - Implement exact Fourier-mode test cases.
//   - Implement multi-mode and Gaussian options.
//   - Add a hot-square / box pulse for visual experiments.

#include "grid2d.hpp"
#include "types.hpp"

Grid2D<Real> make_single_mode_ic(const RealVec& x, const RealVec& y);
Grid2D<Real> make_multi_mode_ic(const RealVec& x, const RealVec& y);
Grid2D<Real> make_gaussian_ic(const RealVec& x, const RealVec& y);
Grid2D<Real> make_hot_square_ic(const RealVec& x, const RealVec& y);

