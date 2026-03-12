#include "fft2d.hpp"
#include "fft1d.hpp"
// fft2d.cpp
// Responsibility:
//   Implementation of row-column 2D FFT/IFFT.
// What to do here:
//   - Transform rows first because they are contiguous in row-major storage.
//   - Gather each column into a temporary vector, transform it, and scatter back.
//   - Keep this file focused only on multidimensional transform orchestration.

void fft_2d_inplace(Grid2D<Complex>& field) {
    (void)field; // TODO: apply 1D FFT across rows and columns.
}

void ifft_2d_inplace(Grid2D<Complex>& field) {
    (void)field; // TODO: apply inverse 1D FFT across rows and columns.
}

