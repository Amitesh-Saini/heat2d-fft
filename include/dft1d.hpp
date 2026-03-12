#pragma once
// dft1d.hpp
// Responsibility:
//   Naive O(N^2) 1D DFT/IDFT reference implementation.
// What to do here:
//   - Implement direct DFT and inverse DFT.
//   - Use this for correctness checks against FFT at small sizes.
//   - Do not use this as the main production transform for large runs.

#include "types.hpp"

ComplexVec dft_1d(const ComplexVec& input);
ComplexVec idft_1d(const ComplexVec& input);

