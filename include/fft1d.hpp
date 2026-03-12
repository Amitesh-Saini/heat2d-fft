#pragma once
// fft1d.hpp
// Responsibility:
//   From-scratch radix-2 Cooley-Tukey FFT and inverse FFT.
// What to do here:
//   - Implement forward and inverse 1D FFT.
//   - Validate against dft1d for powers of two.
//   - Keep normalization conventions consistent.
//   - Start with recursive radix-2; optimize later only after correctness is proven.

#include "types.hpp"

bool is_power_of_two(std::size_t n);
void fft_1d_inplace(ComplexVec& a);
void ifft_1d_inplace(ComplexVec& a);

