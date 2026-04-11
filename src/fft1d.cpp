#include "fft1d.hpp"
#include <cmath>
#include <stdexcept>

constexpr Real PI = 3.14159265358979323846;

// fft1d.cpp
// Responsibility:
//   Implementation of the radix-2 1D FFT/IFFT.
// What to do here:
//   - Implement recursive or iterative Cooley-Tukey.
//   - Reuse helpers for twiddle factors and base cases.
//   - Make the inverse consistent with the forward transform convention.




namespace {

    enum class FFTDirection {
        Forward,
        Inverse
    };
    
    void fft_1d_rec(ComplexVec& a, FFTDirection dir){

    std::size_t N = a.size();

    if(N == 1) return;

    else if(N == 2){

        Complex first_entry  = a[0];
        Complex second_entry = a[1];

        a[0] = first_entry + second_entry;
        a[1] = first_entry - second_entry;

        return;
    }

    else{

        Real sign = (dir == FFTDirection::Forward) ? -1.0 : 1.0;
        
        ComplexVec even(N/2);
        ComplexVec odd(N/2);
        Complex phase = {1.0, 0.0};
        Complex omega = {std::cos(2*PI/N), (sign * std::sin(2*PI/N))};

        for(std::size_t b = 0; b < N/2; b++){

            even[b] = a[2*b];
            odd[b] = a[2*b + 1];
        }

        fft_1d_rec(even, dir);
        fft_1d_rec(odd, dir);

        for(std::size_t k = 0; k < N/2; k++){

            a[k] = even[k] + odd[k] * phase;
            a[k + N/2] = even[k] - odd[k] * phase;
            phase *= omega;
        }
    }
    
}
}


bool is_power_of_two(std::size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

void fft_1d_inplace(ComplexVec& a) {

    std::size_t N = a.size();

    if(!is_power_of_two(N)) throw std::invalid_argument("fft_1d_inplace: input size must be a power of two");

    fft_1d_rec(a, FFTDirection::Forward);
}




void ifft_1d_inplace(ComplexVec& a) {

    std::size_t N = a.size();

    if(!is_power_of_two(N)) throw std::invalid_argument("ifft_1d_inplace: input size must be a power of two");

    fft_1d_rec(a, FFTDirection::Inverse);

    for(std::size_t c = 0; c < N; c++){

        a[c] = a[c] / static_cast<Real>(N);
    }

}

