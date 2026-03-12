#include "fft1d.hpp"
// fft1d.cpp
// Responsibility:
//   Implementation of the radix-2 1D FFT/IFFT.
// What to do here:
//   - Implement recursive or iterative Cooley-Tukey.
//   - Reuse helpers for twiddle factors and base cases.
//   - Make the inverse consistent with the forward transform convention.

bool is_power_of_two(std::size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

void fft_1d_inplace(ComplexVec& a) {
    (void)a; // TODO: implement forward radix-2 FFT.
}

void ifft_1d_inplace(ComplexVec& a) {
    (void)a; // TODO: implement inverse radix-2 FFT.
}

