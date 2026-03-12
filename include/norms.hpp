#pragma once
// norms.hpp
// Responsibility:
//   Discrete norm and error routines for validation.
// What to do here:
//   - Implement discrete L2 and Linf norms.
//   - Implement absolute/relative error helpers.
//   - Keep formulas explicit and documented.

#include "grid2d.hpp"
#include "types.hpp"

Real l2_norm(const Grid2D<Real>& u);
Real linf_norm(const Grid2D<Real>& u);
Real relative_l2_error(const Grid2D<Real>& approx, const Grid2D<Real>& exact);
Real absolute_linf_error(const Grid2D<Real>& approx, const Grid2D<Real>& exact);

