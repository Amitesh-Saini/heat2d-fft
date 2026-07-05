#pragma once
// time_grid.hpp
// Responsibility:
//   Declare helper functions for constructing and validating output time
//   grids.
//
//   The heat solver expects an explicit sorted list of output times. Users
//   often prefer to specify times indirectly:
//
//       t_start = 0.0, t_end = 0.5, num_snapshots = 100
//
//   This module turns those higher-level settings into the concrete
//   output_times vector used by Heat2DConfig. Keeping this logic separate
//   makes it testable in isolation and keeps both main.cpp and the solver
//   focused on their own responsibilities.

#include <cstddef>

#include "types.hpp"

// Builds num_snapshots uniformly spaced output times covering
// [t_start, t_end], with both endpoints included.
//
// Conventions:
//   - num_snapshots >= 2 produces t_start, ..., t_end with spacing
//     (t_end - t_start) / (num_snapshots - 1).
//   - num_snapshots == 1 returns { t_end } (a single final-state snapshot).
//   - The last entry is set to exactly t_end (no floating-point drift from
//     repeated addition).
//
// Requirements (throws std::invalid_argument otherwise):
//   - t_start and t_end finite, t_start >= 0, t_end >= t_start,
//   - num_snapshots >= 1.
RealVec make_uniform_time_grid(Real t_start, Real t_end, std::size_t num_snapshots);
