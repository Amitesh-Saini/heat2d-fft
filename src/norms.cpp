#include "norms.hpp"
// norms.cpp
// Responsibility:
//   Implementation of discrete norm/error routines.
// What to do here:
//   - Keep formulas explicit and easy to verify.
//   - Use these routines in tests, validation runs, and benchmark summaries.

Real l2_norm(const Grid2D<Real>& u) {
    (void)u;
    return 0.0; // TODO: compute discrete L2 norm.
}

Real linf_norm(const Grid2D<Real>& u) {
    (void)u;
    return 0.0; // TODO: compute discrete Linf norm.
}

Real relative_l2_error(const Grid2D<Real>& approx, const Grid2D<Real>& exact) {
    (void)approx;
    (void)exact;
    return 0.0; // TODO: compute relative L2 error.
}

Real absolute_linf_error(const Grid2D<Real>& approx, const Grid2D<Real>& exact) {
    (void)approx;
    (void)exact;
    return 0.0; // TODO: compute absolute Linf error.
}

