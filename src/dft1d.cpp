#include "dft1d.hpp"
#include <cmath>

constexpr Real PI = 3.14159265358979323846;

// dft1d.cpp
// Responsibility:
//   Implementation file for the direct 1D DFT/IDFT.
// What to do here:
//   - Write the O(N^2) summation formulas.
//   - Match the exact sign and normalization conventions used by your FFT.
//   - Use this as a trusted small-N reference.

ComplexVec dft_1d(const ComplexVec& input) {
    
    std::size_t N = input.size();
    ComplexVec output(N);
    
    for(std::size_t k = 0; k < N; k++){

        Complex omega = {std::cos(2*PI*k/N), (-std::sin(2*PI*k/N))};
        Complex phase = {1.0 , 0.0};

        for(std::size_t j = 0; j < N; j++){
            
            output[k] += input[j] * phase;
            phase *= omega;
        }
    }

    return output;
}

ComplexVec idft_1d(const ComplexVec& input) {

    std::size_t N = input.size();
    ComplexVec output(N);
    
    for(std::size_t j = 0; j < N; j++){

        Complex sum = {0.0 , 0.0};
        Complex omega = {std::cos(2*PI*j/N), std::sin(2*PI*j/N)};
        Complex phase = {1.0 , 0.0};

        for(std::size_t k = 0; k < N; k++){
            
            sum += input[k] * phase;
            phase *= omega;
        }

        output[j] = sum / static_cast<Real>(N);
    }

    return output;
}

