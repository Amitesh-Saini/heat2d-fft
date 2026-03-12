#include "io_csv.hpp"
// io_csv.cpp
// Responsibility:
//   Implementation of CSV writing helpers.
// What to do here:
//   - Serialize coordinate vectors and 2D fields.
//   - Keep output formatting stable so Python scripts stay simple.
//   - Prefer one helper per output shape/type.

void write_vector_csv(const std::string& path, const RealVec& values) {
    (void)path;
    (void)values;
    // TODO: write one value per line or one comma-separated row; pick one convention and document it.
}

void write_grid_csv(const std::string& path, const Grid2D<Real>& field) {
    (void)path;
    (void)field;
    // TODO: write the 2D field row by row as CSV.
}

