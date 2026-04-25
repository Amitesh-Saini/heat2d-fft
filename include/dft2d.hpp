#pragma once
// dft2d.hpp
// Responsibility:
//   2D DFT/IDFT built from repeated 1D DFTs on rows and columns.
// What to do here:
//   - Apply 1D DFT to each row.
//   - Gather/scatter columns into temporary vectors and transform them.
//   - Reuse the 1D DFT implementation rather than duplaicating logic.


#include "grid2d.hpp"
#include "types.hpp"

Grid2D<Complex> dft_2d(const Grid2D<Complex>& field);
Grid2D<Complex> idft_2d(const Grid2D<Complex>& field);

