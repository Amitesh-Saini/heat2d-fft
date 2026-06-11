// time_grid.hpp
// Responsibility:
//   Declare helper functions for constructing and validating output time grids.
//
//   The heat solver expects an explicit sorted list of output times. However,
//   users often prefer to specify times indirectly, for example:
//
//       final_time = 0.5
//       num_snapshots = 100
//
//   This module turns those higher-level time settings into the concrete
//   output_times vector used by Heat2DConfig.
//
//   Keeping this logic separate makes it easy to test and keeps both main.cpp
//   and the solver focused on their own responsibilities.