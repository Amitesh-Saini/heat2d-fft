#pragma once
// fft2d.hpp
// Responsibility:
//   2D FFT/IFFT built from repeated 1D FFTs on rows and columns.
// What to do here:
//   - Apply 1D FFT to each row.
//   - Gather/scatter columns into temporary vectors and transform them.
//   - Reuse the 1D FFT implementation rather than duplicating logic.

#include "grid2d.hpp"
#include "types.hpp"

void fft_2d_inplace(Grid2D<Complex>& field);
void ifft_2d_inplace(Grid2D<Complex>& field);

void fft_2d_row_inplace(Grid2D<Complex>& field);
void fft_2d_col_inplace(Grid2D<Complex>& field);

void ifft_2d_col_inplace(Grid2D<Complex>& field);
void ifft_2d_row_inplace(Grid2D<Complex>& field);

