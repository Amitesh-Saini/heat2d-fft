// test_fft1d.cpp
// Responsibility:
//   Test suite for the radix-2 1D FFT/IFFT implementation.
// What to do here:
//   - Verify FFT produces correct known outputs.
//   - Verify FFT/IFFT round-trip consistency.
//   - Verify FFT agrees with the DFT reference for all test sizes.
//   - Verify fundamental Fourier properties hold for the FFT.
//   - Test power-of-two enforcement.
//   - Scale N up to 128 to exercise deeper recursion levels.

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>

#include "fft1d.hpp"
#include "dft1d.hpp"
#include "types.hpp"
#include "1D_test_utils.hpp"


int main() {

    std::vector<std::string> failed_tests;

    const Real abs_tol = 1e-10;
    const Real rel_tol = 1e-10;

    std::cout << "=== Running 1D FFT tests ===\n\n";


    // ------------------------------------------------------------
    // Known-output tests
    // Same analytic cases as the DFT tests but run through the FFT.
    // These establish that the FFT computes the correct transform
    // before any comparison against the DFT reference.
    // ------------------------------------------------------------

    // Zero vector: FFT(0) = 0, N=4
    {
        ComplexVec input = {
            Complex(0.0, 0.0), Complex(0.0, 0.0),
            Complex(0.0, 0.0), Complex(0.0, 0.0)
        };
        ComplexVec expected = {
            Complex(0.0, 0.0), Complex(0.0, 0.0),
            Complex(0.0, 0.0), Complex(0.0, 0.0)
        };

        if(!run_known_output_case("zero_vector_n4", input, expected, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("zero_vector_n4");
        }
    }

    // Zero vector N=8
    {
        ComplexVec input(8, Complex(0.0, 0.0));
        ComplexVec expected(8, Complex(0.0, 0.0));

        if(!run_known_output_case("zero_vector_n8", input, expected, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("zero_vector_n8");
        }
    }

    // Constant vector: FFT([1,1,1,1]) = [4,0,0,0], N=4
    {
        ComplexVec input = {
            Complex(1.0, 0.0), Complex(1.0, 0.0),
            Complex(1.0, 0.0), Complex(1.0, 0.0)
        };
        ComplexVec expected = {
            Complex(4.0, 0.0), Complex(0.0, 0.0),
            Complex(0.0, 0.0), Complex(0.0, 0.0)
        };

        if(!run_known_output_case("constant_vector_n4", input, expected, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("constant_vector_n4");
        }
    }

    // Constant vector N=8: FFT([1,...,1]) = [8,0,...,0]
    {
        ComplexVec input(8, Complex(1.0, 0.0));
        ComplexVec expected(8, Complex(0.0, 0.0));
        expected[0] = Complex(8.0, 0.0);

        if(!run_known_output_case("constant_vector_n8", input, expected, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("constant_vector_n8");
        }
    }

    // Constant vector N=32: FFT([1,...,1]) = [32,0,...,0]
    {
        ComplexVec input(32, Complex(1.0, 0.0));
        ComplexVec expected(32, Complex(0.0, 0.0));
        expected[0] = Complex(32.0, 0.0);

        if(!run_known_output_case("constant_vector_n32", input, expected, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("constant_vector_n32");
        }
    }

    // Impulse at index 0: FFT([1,0,...,0]) = [1,1,...,1], N=4
    {
        ComplexVec input = {
            Complex(1.0, 0.0), Complex(0.0, 0.0),
            Complex(0.0, 0.0), Complex(0.0, 0.0)
        };
        ComplexVec expected = {
            Complex(1.0, 0.0), Complex(1.0, 0.0),
            Complex(1.0, 0.0), Complex(1.0, 0.0)
        };

        if(!run_known_output_case("impulse_index0_n4", input, expected, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("impulse_index0_n4");
        }
    }

    // Impulse at index 0, N=8: spectrum is all ones
    {
        ComplexVec input(8, Complex(0.0, 0.0));
        input[0] = Complex(1.0, 0.0);
        ComplexVec expected(8, Complex(1.0, 0.0));

        if(!run_known_output_case("impulse_index0_n8", input, expected, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("impulse_index0_n8");
        }
    }

    // Shifted impulse at index 1, N=4:
    // FFT([0,1,0,0]) = [1, -i, -1, i]
    {
        ComplexVec input = {
            Complex(0.0, 0.0), Complex(1.0, 0.0),
            Complex(0.0, 0.0), Complex(0.0, 0.0)
        };
        ComplexVec expected = {
            Complex( 1.0,  0.0), Complex( 0.0, -1.0),
            Complex(-1.0,  0.0), Complex( 0.0,  1.0)
        };

        if(!run_known_output_case("shifted_impulse_index1_n4", input, expected, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("shifted_impulse_index1_n4");
        }
    }

    // Alternating real vector N=4: FFT([1,-1,1,-1]) = [0,0,4,0]
    {
        ComplexVec input = {
            Complex( 1.0, 0.0), Complex(-1.0, 0.0),
            Complex( 1.0, 0.0), Complex(-1.0, 0.0)
        };
        ComplexVec expected = {
            Complex(0.0, 0.0), Complex(0.0, 0.0),
            Complex(4.0, 0.0), Complex(0.0, 0.0)
        };

        if(!run_known_output_case("alternating_real_n4", input, expected, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("alternating_real_n4");
        }
    }

    // Alternating real vector N=8: FFT([1,-1,...]) = [0,...,0,8,0,...,0]
    // Energy concentrated at Nyquist bin k=4
    {
        ComplexVec input(8);
        for(std::size_t j = 0; j < 8; j++)
            input[j] = Complex((j % 2 == 0) ? 1.0 : -1.0, 0.0);
        ComplexVec expected(8, Complex(0.0, 0.0));
        expected[4] = Complex(8.0, 0.0);

        if(!run_known_output_case("alternating_real_n8", input, expected, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("alternating_real_n8");
        }
    }

    // Single complex Fourier mode N=4:
    // input[j] = exp(2*pi*i*j/4) = [1,i,-1,-i]
    // FFT should give [0,4,0,0]
    {
        ComplexVec input = {
            Complex( 1.0,  0.0), Complex( 0.0,  1.0),
            Complex(-1.0,  0.0), Complex( 0.0, -1.0)
        };
        ComplexVec expected = {
            Complex(0.0, 0.0), Complex(4.0, 0.0),
            Complex(0.0, 0.0), Complex(0.0, 0.0)
        };

        if(!run_known_output_case("single_complex_mode_n4", input, expected, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("single_complex_mode_n4");
        }
    }

    // Single Fourier mode N=8, mode k=3:
    // input[j] = exp(2*pi*i*3*j/8), FFT gives [0,0,0,8,0,0,0,0]
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * 3.0 * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), std::sin(theta));
        }
        ComplexVec expected(N, Complex(0.0, 0.0));
        expected[3] = Complex(8.0, 0.0);

        if(!run_known_output_case("single_mode_k3_n8", input, expected, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("single_mode_k3_n8");
        }
    }


    // ------------------------------------------------------------
    // IFFT known-output tests
    // Direct correctness check for the inverse transform.
    // Symmetric coverage with the forward known-output tests.
    // ------------------------------------------------------------

    // IFFT([4,0,0,0]) = [1,1,1,1]
    {
        ComplexVec spectrum = {
            Complex(4.0, 0.0), Complex(0.0, 0.0),
            Complex(0.0, 0.0), Complex(0.0, 0.0)
        };
        ComplexVec expected = {
            Complex(1.0, 0.0), Complex(1.0, 0.0),
            Complex(1.0, 0.0), Complex(1.0, 0.0)
        };

        if(!run_inverse_known_output_case("ifft_constant_spectrum_n4", spectrum, expected, ITransform_1d::IFFT, abs_tol, rel_tol)) {
            failed_tests.push_back("ifft_constant_spectrum_n4");
        }
    }

    // IFFT([1,1,1,1]) = [1,0,0,0]
    {
        ComplexVec spectrum = {
            Complex(1.0, 0.0), Complex(1.0, 0.0),
            Complex(1.0, 0.0), Complex(1.0, 0.0)
        };
        ComplexVec expected = {
            Complex(1.0, 0.0), Complex(0.0, 0.0),
            Complex(0.0, 0.0), Complex(0.0, 0.0)
        };

        if(!run_inverse_known_output_case("ifft_flat_spectrum_n4", spectrum, expected, ITransform_1d::IFFT, abs_tol, rel_tol)) {
            failed_tests.push_back("ifft_flat_spectrum_n4");
        }
    }

    // IFFT([8,0,...,0]) = [1,1,...,1], N=8
    {
        ComplexVec spectrum(8, Complex(0.0, 0.0));
        spectrum[0] = Complex(8.0, 0.0);
        ComplexVec expected(8, Complex(1.0, 0.0));

        if(!run_inverse_known_output_case("ifft_constant_spectrum_n8", spectrum, expected, ITransform_1d::IFFT, abs_tol, rel_tol)) {
            failed_tests.push_back("ifft_constant_spectrum_n8");
        }
    }


    // ------------------------------------------------------------
    // Round-trip tests: FFT followed by IFFT
    // Tests forward/inverse consistency including sign convention
    // and 1/N normalization across a range of N values.
    // ------------------------------------------------------------

    // Real-valued round trip N=4
    {
        ComplexVec input = {
            Complex(1.0, 0.0), Complex(2.0, 0.0),
            Complex(3.0, 0.0), Complex(4.0, 0.0)
        };

        if(!run_round_trip_case("round_trip_real_n4", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_n4");
        }
    }

    // Complex-valued round trip N=4
    {
        ComplexVec input = {
            Complex( 1.0,  2.0), Complex(-3.0,  0.5),
            Complex( 0.0, -1.0), Complex( 2.5, -4.0)
        };

        if(!run_round_trip_case("round_trip_complex_n4", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_n4");
        }
    }

    // Real-valued round trip N=8
    {
        ComplexVec input = {
            Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(3.0, 0.0), Complex(4.0, 0.0),
            Complex(5.0, 0.0), Complex(6.0, 0.0), Complex(7.0, 0.0), Complex(8.0, 0.0)
        };

        if(!run_round_trip_case("round_trip_real_n8", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_n8");
        }
    }

    // Complex-valued round trip N=16
    {
        std::size_t N = 16;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++)
            input[j] = Complex(std::cos(static_cast<Real>(j)), std::sin(static_cast<Real>(j)));

        if(!run_round_trip_case("round_trip_complex_n16", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_n16");
        }
    }

    // Real-valued round trip N=32
    {
        std::size_t N = 32;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++)
            input[j] = Complex(static_cast<Real>(j + 1), 0.0);

        if(!run_round_trip_case("round_trip_real_n32", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_n32");
        }
    }

    // Complex-valued round trip N=64
    {
        std::size_t N = 64;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), std::sin(2.0 * theta));
        }

        if(!run_round_trip_case("round_trip_complex_n64", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_n64");
        }
    }

    // Real-valued round trip N=128
    // Exercises 7 levels of recursion
    {
        std::size_t N = 128;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(3.0 * theta) + std::sin(7.0 * theta), 0.0);
        }

        if(!run_round_trip_case("round_trip_real_n128", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_n128");
        }
    }


    // ------------------------------------------------------------
    // Linearity tests
    // FFT(alpha*x + beta*y) == alpha*FFT(x) + beta*FFT(y)
    // ------------------------------------------------------------

    // Real scalars N=4
    {
        ComplexVec x = {
            Complex(1.0, 0.0), Complex(2.0, 0.0),
            Complex(3.0, 0.0), Complex(4.0, 0.0)
        };
        ComplexVec y = {
            Complex(4.0, 0.0), Complex(3.0, 0.0),
            Complex(2.0, 0.0), Complex(1.0, 0.0)
        };
        Complex alpha = {2.0, 0.0};
        Complex beta  = {-1.0, 0.0};

        if(!run_linearity_case("linearity_real_scalars_n4", x, y, alpha, beta, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_real_scalars_n4");
        }
    }

    // Complex scalars N=8
    {
        std::size_t N = 8;
        ComplexVec x(N), y(N);
        for(std::size_t j = 0; j < N; j++){
            x[j] = Complex(static_cast<Real>(j), 1.0);
            y[j] = Complex(static_cast<Real>(N - j), -0.5);
        }
        Complex alpha = {1.0,  2.0};
        Complex beta  = {0.5, -1.0};

        if(!run_linearity_case("linearity_complex_scalars_n8", x, y, alpha, beta, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_complex_scalars_n8");
        }
    }

    // Real scalars N=32
    {
        std::size_t N = 32;
        ComplexVec x(N), y(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            x[j] = Complex(std::cos(theta), 0.0);
            y[j] = Complex(std::sin(theta), 0.0);
        }
        Complex alpha = {3.0, 0.0};
        Complex beta  = {-2.0, 0.0};

        if(!run_linearity_case("linearity_real_scalars_n32", x, y, alpha, beta, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_real_scalars_n32");
        }
    }

    // Complex scalars N=64
    {
        std::size_t N = 64;
        ComplexVec x(N), y(N);
        for(std::size_t j = 0; j < N; j++){
            x[j] = Complex(std::cos(static_cast<Real>(j)), std::sin(static_cast<Real>(j)));
            y[j] = Complex(static_cast<Real>(j) / static_cast<Real>(N), -1.0);
        }
        Complex alpha = {1.5, -0.5};
        Complex beta  = {-1.0,  2.0};

        if(!run_linearity_case("linearity_complex_scalars_n64", x, y, alpha, beta, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_complex_scalars_n64");
        }
    }


    // ------------------------------------------------------------
    // Parseval's identity tests
    // sum_j |x[j]|^2 == (1/N) * sum_k |X[k]|^2
    // Energy conservation must hold at all tested sizes.
    // The spectral heat solver relies on this for correct modal
    // energy dissipation.
    // ------------------------------------------------------------

    // Real input N=4
    {
        ComplexVec input = {
            Complex(1.0, 0.0), Complex(2.0, 0.0),
            Complex(3.0, 0.0), Complex(4.0, 0.0)
        };

        if(!run_parseval_case("parseval_real_n4", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_real_n4");
        }
    }

    // Complex input N=4
    {
        ComplexVec input = {
            Complex( 1.0,  2.0), Complex(-1.0,  3.0),
            Complex( 2.0, -1.0), Complex( 0.0,  1.0)
        };

        if(!run_parseval_case("parseval_complex_n4", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_complex_n4");
        }
    }

    // Single Fourier mode N=8
    // Physical energy = N, spectral energy = (1/N)*N^2 = N
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), std::sin(theta));
        }

        if(!run_parseval_case("parseval_single_mode_n8", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_single_mode_n8");
        }
    }

    // Real sinusoid N=16
    {
        std::size_t N = 16;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), 0.0);
        }

        if(!run_parseval_case("parseval_cosine_n16", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_cosine_n16");
        }
    }

    // Mixed frequency real input N=32
    // x[j] = cos(2*pi*j/32) + 0.5*cos(6*pi*j/32)
    // Two cosine modes, energy split across four bins
    {
        std::size_t N = 32;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta) + 0.5 * std::cos(3.0 * theta), 0.0);
        }

        if(!run_parseval_case("parseval_mixed_freq_n32", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_mixed_freq_n32");
        }
    }

    // Complex input N=64
    {
        std::size_t N = 64;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), std::sin(2.0 * theta));
        }

        if(!run_parseval_case("parseval_complex_n64", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_complex_n64");
        }
    }


    // ------------------------------------------------------------
    // Time-shift property tests
    // FFT(x[j - m])[k] = exp(-2*pi*i*k*m/N) * X[k]
    // ------------------------------------------------------------

    // Impulse shifted by 1, N=4
    {
        ComplexVec input = {
            Complex(1.0, 0.0), Complex(0.0, 0.0),
            Complex(0.0, 0.0), Complex(0.0, 0.0)
        };

        if(!run_time_shift_case("time_shift_impulse_m1_n4", input, 1, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("time_shift_impulse_m1_n4");
        }
    }

    // Cosine shifted by N/2, N=8
    // Shifting by half-period flips sign of all odd-frequency bins
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), 0.0);
        }

        if(!run_time_shift_case("time_shift_cosine_m4_n8", input, 4, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("time_shift_cosine_m4_n8");
        }
    }

    // Complex input shifted by 3, N=8
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++)
            input[j] = Complex(std::cos(static_cast<Real>(j)), std::sin(static_cast<Real>(j)));

        if(!run_time_shift_case("time_shift_complex_m3_n8", input, 3, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("time_shift_complex_m3_n8");
        }
    }

    // Full period shift gives identity, N=8
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++)
            input[j] = Complex(static_cast<Real>(j + 1), 0.0);

        if(!run_time_shift_case("time_shift_full_period_n8", input, 8, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("time_shift_full_period_n8");
        }
    }

    // Negative shift (left shift) by -1, N=8
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++)
            input[j] = Complex(std::cos(static_cast<Real>(j)), 0.0);

        if(!run_time_shift_case("time_shift_negative_m1_n8", input, -1, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("time_shift_negative_m1_n8");
        }
    }

    // Shift by 7, N=16
    {
        std::size_t N = 16;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta) + std::sin(3.0 * theta), 0.0);
        }

        if(!run_time_shift_case("time_shift_m7_n16", input, 7, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("time_shift_m7_n16");
        }
    }

    // Shift by 13, N=32
    {
        std::size_t N = 32;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(5.0 * theta), std::sin(3.0 * theta));
        }

        if(!run_time_shift_case("time_shift_m13_n32", input, 13, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("time_shift_m13_n32");
        }
    }

    // Negative shift by -5, N=64
    {
        std::size_t N = 64;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), 0.0);
        }

        if(!run_time_shift_case("time_shift_negative_m5_n64", input, -5, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("time_shift_negative_m5_n64");
        }
    }


    // ------------------------------------------------------------
    // Conjugate symmetry tests
    // For real-valued input: X[k] = conj(X[(N-k) % N])
    // Critical precondition for the solver to produce real output.
    // ------------------------------------------------------------

    // Constant real input N=4
    {
        ComplexVec input = {
            Complex(1.0, 0.0), Complex(1.0, 0.0),
            Complex(1.0, 0.0), Complex(1.0, 0.0)
        };

        if(!run_conjugate_symmetry_case("conj_sym_constant_n4", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_constant_n4");
        }
    }

    // Arbitrary real input N=8
    {
        ComplexVec input = {
            Complex(3.0, 0.0), Complex(1.0, 0.0), Complex(-2.0, 0.0), Complex(0.5, 0.0),
            Complex(7.0, 0.0), Complex(-1.0, 0.0), Complex(4.0, 0.0), Complex(2.0, 0.0)
        };

        if(!run_conjugate_symmetry_case("conj_sym_arbitrary_n8", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_arbitrary_n8");
        }
    }

    // Cosine input N=16
    {
        std::size_t N = 16;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), 0.0);
        }

        if(!run_conjugate_symmetry_case("conj_sym_cosine_n16", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_cosine_n16");
        }
    }

    // Mixed frequency real input N=32
    {
        std::size_t N = 32;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta) + 0.5 * std::sin(5.0 * theta), 0.0);
        }

        if(!run_conjugate_symmetry_case("conj_sym_mixed_n32", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_mixed_n32");
        }
    }

    // Arbitrary real input N=64
    {
        std::size_t N = 64;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(3.0 * theta) + std::sin(7.0 * theta), 0.0);
        }

        if(!run_conjugate_symmetry_case("conj_sym_real_n64", input, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_real_n64");
        }
    }


    // ------------------------------------------------------------
    // Modulation property tests
    // FFT(x[j] * exp(2*pi*i*k0*j/N))[k] = X[(k - k0) mod N]
    // ------------------------------------------------------------

    // Impulse shifted by 1 frequency bin, N=4
    {
        ComplexVec input = {
            Complex(1.0, 0.0), Complex(0.0, 0.0),
            Complex(0.0, 0.0), Complex(0.0, 0.0)
        };

        if(!run_modulation_case("modulation_impulse_k1_n4", input, 1, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_impulse_k1_n4");
        }
    }

    // Cosine shifted by 1 bin, N=8
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), 0.0);
        }

        if(!run_modulation_case("modulation_cosine_k1_n8", input, 1, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_cosine_k1_n8");
        }
    }

    // Complex input shifted by 3 bins, N=8
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++)
            input[j] = Complex(std::cos(static_cast<Real>(j)), std::sin(static_cast<Real>(j)));

        if(!run_modulation_case("modulation_complex_k3_n8", input, 3, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_complex_k3_n8");
        }
    }

    // Full period shift is identity, N=8
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++)
            input[j] = Complex(static_cast<Real>(j + 1), 0.0);

        if(!run_modulation_case("modulation_full_period_n8", input, 8, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_full_period_n8");
        }
    }

    // Shift by 5 bins, N=16
    {
        std::size_t N = 16;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta) + std::sin(3.0 * theta), 0.0);
        }

        if(!run_modulation_case("modulation_k5_n16", input, 5, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_k5_n16");
        }
    }

    // Complex input shifted by 11 bins, N=32
    {
        std::size_t N = 32;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(5.0 * theta), std::sin(3.0 * theta));
        }

        if(!run_modulation_case("modulation_complex_k11_n32", input, 11, Transform_1d::FFT, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_complex_k11_n32");
        }
    }


    // ------------------------------------------------------------
    // Power-of-two enforcement
    // fft_1d_inplace and ifft_1d_inplace must throw
    // std::invalid_argument for non-power-of-two sizes.
    // This is a contract test, not a numerical test.
    // ------------------------------------------------------------

    // Forward FFT with N=3
    {
        bool threw = false;
        try {
            ComplexVec input(3, Complex(1.0, 0.0));
            fft_1d_inplace(input);
        } catch(const std::invalid_argument&) {
            threw = true;
        }

        if(!threw) {
            std::cout << "FAIL: pow2_enforce_fft_n3 -- expected std::invalid_argument, none thrown\n";
            failed_tests.push_back("pow2_enforce_fft_n3");
        }
    }

    // Forward FFT with N=6
    {
        bool threw = false;
        try {
            ComplexVec input(6, Complex(1.0, 0.0));
            fft_1d_inplace(input);
        } catch(const std::invalid_argument&) {
            threw = true;
        }

        if(!threw) {
            std::cout << "FAIL: pow2_enforce_fft_n6 -- expected std::invalid_argument, none thrown\n";
            failed_tests.push_back("pow2_enforce_fft_n6");
        }
    }

    // Inverse FFT with N=5
    {
        bool threw = false;
        try {
            ComplexVec input(5, Complex(1.0, 0.0));
            ifft_1d_inplace(input);
        } catch(const std::invalid_argument&) {
            threw = true;
        }

        if(!threw) {
            std::cout << "FAIL: pow2_enforce_ifft_n5 -- expected std::invalid_argument, none thrown\n";
            failed_tests.push_back("pow2_enforce_ifft_n5");
        }
    }

    // N=0 should also throw
    {
        bool threw = false;
        try {
            ComplexVec input(0);
            fft_1d_inplace(input);
        } catch(const std::invalid_argument&) {
            threw = true;
        }

        if(!threw) {
            std::cout << "FAIL: pow2_enforce_fft_n0 -- expected std::invalid_argument, none thrown\n";
            failed_tests.push_back("pow2_enforce_fft_n0");
        }
    }


    // ------------------------------------------------------------
    // FFT vs DFT agreement tests
    // Verify that the FFT and DFT produce identical results for
    // the same input. This is the definitive correctness check:
    // the DFT is the mathematical definition, the FFT must match it.
    // Run across all tested sizes up to N=64.
    // ------------------------------------------------------------

    // Zero vector N=4
    {
        ComplexVec input(4, Complex(0.0, 0.0));
        if(!run_fft_vs_dft_case("fft_vs_dft_zero_n4", input, abs_tol, rel_tol))
            failed_tests.push_back("fft_vs_dft_zero_n4");
    }

    // Impulse N=4
    {
        ComplexVec input(4, Complex(0.0, 0.0));
        input[0] = Complex(1.0, 0.0);
        if(!run_fft_vs_dft_case("fft_vs_dft_impulse_n4", input, abs_tol, rel_tol))
            failed_tests.push_back("fft_vs_dft_impulse_n4");
    }

    // Real sinusoid N=8
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), 0.0);
        }
        if(!run_fft_vs_dft_case("fft_vs_dft_cosine_n8", input, abs_tol, rel_tol))
            failed_tests.push_back("fft_vs_dft_cosine_n8");
    }

    // Complex input N=8
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++)
            input[j] = Complex(std::cos(static_cast<Real>(j)), std::sin(static_cast<Real>(j)));
        if(!run_fft_vs_dft_case("fft_vs_dft_complex_n8", input, abs_tol, rel_tol))
            failed_tests.push_back("fft_vs_dft_complex_n8");
    }

    // Mixed frequency real input N=16
    {
        std::size_t N = 16;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta) + std::sin(3.0 * theta), 0.0);
        }
        if(!run_fft_vs_dft_case("fft_vs_dft_mixed_n16", input, abs_tol, rel_tol))
            failed_tests.push_back("fft_vs_dft_mixed_n16");
    }

    // Complex input N=16
    {
        std::size_t N = 16;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(3.0 * theta), std::sin(5.0 * theta));
        }
        if(!run_fft_vs_dft_case("fft_vs_dft_complex_n16", input, abs_tol, rel_tol))
            failed_tests.push_back("fft_vs_dft_complex_n16");
    }

    // Real input N=32
    {
        std::size_t N = 32;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(5.0 * theta) + 0.5 * std::sin(theta), 0.0);
        }
        if(!run_fft_vs_dft_case("fft_vs_dft_real_n32", input, abs_tol, rel_tol))
            failed_tests.push_back("fft_vs_dft_real_n32");
    }

    // Complex input N=64
    {
        std::size_t N = 64;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), std::sin(2.0 * theta));
        }
        if(!run_fft_vs_dft_case("fft_vs_dft_complex_n64", input, abs_tol, rel_tol))
            failed_tests.push_back("fft_vs_dft_complex_n64");
    }


    // ------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------

    if(failed_tests.empty()) {
        std::cout << "All tests passed.\n";
        return 0;
    }

    std::cout << failed_tests.size() << " test(s) failed:\n";
    for(const auto& test_name : failed_tests) {
        std::cout << " - " << test_name << "\n";
    }

    return 1;
}