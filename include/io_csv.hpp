#pragma once
// io_csv.hpp
// Responsibility:
//   CSV output helpers for grids, fields, and summary values.
// What to do here:
//   - Write x/y coordinate vectors.
//   - Write 2D field snapshots.
//   - Keep file-format logic here rather than inside the solver.

#include "grid2d.hpp"
#include "types.hpp"
#include <string>

void write_vector_csv(const std::string& path, const RealVec& values);
void write_grid_csv(const std::string& path, const Grid2D<Real>& field);

