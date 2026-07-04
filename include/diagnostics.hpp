#pragma once
// diagnostics.hpp
// Responsibility:
//   Declare diagnostic utilities for analyzing heat-equation snapshots.
//
//   Diagnostics are small scalar quantities computed from each temperature
//   field. For the periodic heat equation:
//     - the mean temperature is exactly conserved (it is the k = 0 Fourier
//       mode); any drift indicates a bug,
//     - the L2 norm decreases monotonically in time,
//     - min/max obey a maximum principle (extrema do not grow).
//
//   Diagnostics are separate from the solver so that the solver only
//   computes snapshots and does not become responsible for analysis or
//   output.
//
//   This module also provides the analytic-error diagnostics used by the
//   Fourier-mode validation presets, where the exact solution
//
//       u(x,y,t) = sum_m A_m exp(-alpha |k_m|^2 t)
//                        cos(2*pi*(kx_m*x/Lx + ky_m*y/Ly) + phase_m)
//
//   is known in closed form, so the relative L2 error of the numerical
//   snapshot can be recorded at every output time.

#include <cstddef>
#include <vector>

#include "grid2d.hpp"
#include "initial_conditions.hpp"  // FourierMode2D
#include "types.hpp"

// Per-snapshot scalar diagnostics, in output order.
struct SnapshotDiagnostics {
    Real time = Real{0};
    Real mean = Real{0};
    Real l2_norm = Real{0};
    Real min_value = Real{0};
    Real max_value = Real{0};
};

// Discrete mean: (1 / (nx*ny)) * sum_ij u(i,j).
// Equal to the k = 0 Fourier coefficient divided by nx*ny, so it is exactly
// conserved by the spectral heat evolution.
Real compute_mean(const Grid2D<Real>& field);

// Discrete approximation of the continuous L2 norm:
//
//     ||u||_2 = sqrt( dx * dy * sum_ij u(i,j)^2 ),
//
// with dx = Lx/nx, dy = Ly/ny. The dx*dy weighting makes the value
// resolution-independent, directly comparable to analytic norms, and
// consistent with Parseval's theorem for the FFT normalization used in this
// project.
Real compute_l2_norm(const Grid2D<Real>& field, Real Lx, Real Ly);

// Minimum and maximum field values.
Real compute_min(const Grid2D<Real>& field);
Real compute_max(const Grid2D<Real>& field);

// Convenience: computes all diagnostics for one snapshot.
SnapshotDiagnostics compute_snapshot_diagnostics(
    const Grid2D<Real>& field, Real time, Real Lx, Real Ly);


// ---------------------------------------------------------------------------
// Analytic error for Fourier-mode validation runs
// ---------------------------------------------------------------------------

// Evaluates the exact heat-equation solution at time t for an initial
// condition that is a sum of plane-wave Fourier modes (a single mode is a
// one-element vector). Sampled on the same periodic grid convention as
// initial_conditions.hpp.
Grid2D<Real> make_exact_fourier_mode_solution(
    Real Lx, Real Ly, std::size_t nx, std::size_t ny,
    const std::vector<FourierMode2D>& modes, Real alpha, Real time);

// Relative L2 error of a numerical snapshot against a reference field:
//
//     ||numerical - reference||_2 / ||reference||_2,
//
// using the dx*dy-weighted norm above. If ||reference||_2 falls below a
// small absolute floor (the reference has decayed to numerical zero), the
// absolute error ||numerical - reference||_2 is returned instead, so the
// tail of a validation run does not divide by ~0.
//
// Throws std::invalid_argument if the grid shapes do not match.
Real compute_relative_l2_error(
    const Grid2D<Real>& numerical, const Grid2D<Real>& reference,
    Real Lx, Real Ly);