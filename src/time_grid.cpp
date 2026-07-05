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

#include "time_grid.hpp"

#include <vector>
#include <iomanip>


RealVec make_uniform_time_grid(Real t_start, Real t_end, std::size_t num_snapshots){

    if (!std::isfinite(t_start) || t_start < Real{0})
        throw std::invalid_argument("make_uniform_time_grid: t_start must be finite and nonnegative");

    if (!std::isfinite(t_end) || t_end < t_start)
        throw std::invalid_argument("make_uniform_time_grid: t_end must be finite and >= t_start");

    if (num_snapshots == 0)
        throw std::invalid_argument("make_uniform_time_grid: num_snapshots must be at least 1");
    
    if (num_snapshots == 1) return RealVec{ t_end };
        
    RealVec output_times(num_snapshots);

    Real step_size = (t_end - t_start) / (static_cast<Real>(num_snapshots) - Real{1});

    for(std::size_t i = 0; i < output_times.size(); ++i){

        output_times[i] = t_start + step_size * static_cast<Real>(i);
    }

    output_times.back() = t_end;

    return output_times;
}