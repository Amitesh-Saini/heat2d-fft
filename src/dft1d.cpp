#include "dft1d.hpp"
#include <cmath>


// dft1d.cpp
// Responsibility:
//   Implementation file for the direct 1D DFT/IDFT.
// What to do here:
//   - Write the O(N^2) summation formulas.
//   - Match the exact sign and normalization conventions used by your FFT.
//   - Use this as a trusted small-N reference.



namespace{

    enum class DFTDirection {
        Forward,
        Inverse
    };

    ComplexVec dft_1d_kernel(const ComplexVec& input, DFTDirection dir) {

        std::size_t N = input.size();

        if(N == 0) throw std::invalid_argument("dft_1d: input array size is 0");

        ComplexVec output(N);

        Real sign = (dir == DFTDirection::Forward) ? -1.0 : 1.0;
        
        for(std::size_t k = 0; k < N; k++){

            Complex omega = {std::cos(2*PI*k/N), (sign * std::sin(2*PI*k/N))};
            Complex phase = {1.0 , 0.0};

            for(std::size_t j = 0; j < N; j++){
                
                output[k] += input[j] * phase;
                phase *= omega;
            }
        }
        
        return output;
    }
}


ComplexVec dft_1d(const ComplexVec& input) {

    return dft_1d_kernel(input, DFTDirection::Forward);
}

ComplexVec idft_1d(const ComplexVec& input) {

    std::size_t N = input.size();

    ComplexVec output = dft_1d_kernel(input, DFTDirection::Inverse);
    
    for(std::size_t a = 0; a < N; a++){
        output[a] = output[a] / static_cast<Real>(N);
    }

    return output;
}
