// time_grid.cpp
// Responsibility:
//   Implement output-time grid construction and validation utilities.
//
//   This file should handle logic such as:
//     - creating evenly spaced output times from final_time and num_snapshots,
//     - ensuring time values are finite,
//     - ensuring time values are nonnegative,
//     - ensuring output times are sorted,
//     - ensuring the list of output times is not empty.
//
//   The solver should receive a valid output_times vector and should not need
//   to know how that vector was generated.