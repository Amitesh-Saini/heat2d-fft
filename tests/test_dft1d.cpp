// test_dft1d.cpp
// Responsibility:
//   Test scaffold for test_dft1d.
// What to do here:
//   - Add small deterministic tests first.
//   - Compare against exact values or trusted reference routines.
//   - Keep each test focused on one property.


#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include <stdexcept>


#include "dft1d.hpp"
#include "types.hpp"
#include "1D_test_utils.hpp"





int main() {

    std::vector<std::string> failed_tests;

    const Real abs_tol = 1e-12;
    const Real rel_tol = 1e-12;

    std::cout << "=== Running 1D DFT tests ===\n\n";

    
    // Known-output tests
   

    // Zero vector: DFT(0) = 0
    {
        ComplexVec input = {
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0)
        };

        ComplexVec expected = {
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0)
        };

        if(!run_known_output_case("zero_vector_n4", input, expected, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("zero_vector_n4");
        }
    }

    // Constant vector: DFT([1,1,1,1]) = [4,0,0,0]
    {
        ComplexVec input = {
            Complex(1.0, 0.0),
            Complex(1.0, 0.0),
            Complex(1.0, 0.0),
            Complex(1.0, 0.0)
        };

        ComplexVec expected = {
            Complex(4.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0)
        };

        if(!run_known_output_case("constant_vector_n4", input, expected, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("constant_vector_n4");
        }
    }

    // Impulse at index 0: DFT([1,0,0,0]) = [1,1,1,1]
    {
        ComplexVec input = {
            Complex(1.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0)
        };

        ComplexVec expected = {
            Complex(1.0, 0.0),
            Complex(1.0, 0.0),
            Complex(1.0, 0.0),
            Complex(1.0, 0.0)
        };

        if(!run_known_output_case("impulse_index0_n4", input, expected, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("impulse_index0_n4");
        }
    }

    // Shifted impulse at index 1:
    // DFT([0,1,0,0]) = [1, -i, -1, i]
    {
        ComplexVec input = {
            Complex(0.0, 0.0),
            Complex(1.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0)
        };

        ComplexVec expected = {
            Complex( 1.0,  0.0),
            Complex( 0.0, -1.0),
            Complex(-1.0,  0.0),
            Complex( 0.0,  1.0)
        };

        if(!run_known_output_case("shifted_impulse_index1_n4", input, expected, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("shifted_impulse_index1_n4");
        }
    }

    // N=2 hand-checkable case:
    // DFT([3,-1]) = [2,4]
    {
        ComplexVec input = {
            Complex( 3.0, 0.0),
            Complex(-1.0, 0.0)
        };

        ComplexVec expected = {
            Complex(2.0, 0.0),
            Complex(4.0, 0.0)
        };

        if(!run_known_output_case("hand_check_n2_real", input, expected, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("hand_check_n2_real");
        }
    }

    // Alternating real vector:
    // DFT([1,-1,1,-1]) = [0,0,4,0]
    {
        ComplexVec input = {
            Complex( 1.0, 0.0),
            Complex(-1.0, 0.0),
            Complex( 1.0, 0.0),
            Complex(-1.0, 0.0)
        };

        ComplexVec expected = {
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(4.0, 0.0),
            Complex(0.0, 0.0)
        };

        if(!run_known_output_case("alternating_real_n4", input, expected, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("alternating_real_n4");
        }
    }

    // Single complex Fourier mode:
    // input[j] = exp(2*pi*i*j/4) = [1, i, -1, -i]
    // With your forward convention, DFT should be [0,4,0,0]
    {
        ComplexVec input = {
            Complex( 1.0,  0.0),
            Complex( 0.0,  1.0),
            Complex(-1.0,  0.0),
            Complex( 0.0, -1.0)
        };

        ComplexVec expected = {
            Complex(0.0, 0.0),
            Complex(4.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0)
        };

        if(!run_known_output_case("single_complex_mode_n4", input, expected, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("single_complex_mode_n4");
        }
    }

   
    // Round-trip tests
 

    // Real-valued round trip
    {
        ComplexVec input = {
            Complex(1.0, 0.0),
            Complex(2.0, 0.0),
            Complex(3.0, 0.0),
            Complex(4.0, 0.0)
        };

        if(!run_round_trip_case("round_trip_real_n4", input, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_n4");
        }
    }

    // Complex-valued round trip
    {
        ComplexVec input = {
            Complex( 1.0,  2.0),
            Complex(-3.0,  0.5),
            Complex( 0.0, -1.0),
            Complex( 2.5, -4.0)
        };

        if(!run_round_trip_case("round_trip_complex_n4", input, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_n4");
        }
    }

    // Another small round-trip case, N=2
    {
        ComplexVec input = {
            Complex(2.0, -1.0),
            Complex(-0.5, 3.0)
        };

        if(!run_round_trip_case("round_trip_complex_n2", input, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_n2");
        }
    }

    
    // Shifted impulse coverage: m=2 and m=3
    // DFT(delta_{m})[k] = exp(-2*pi*i*m*k/N)
    // For N=4, m=2: X[k] = exp(-i*pi*k) = [1, -1, 1, -1]
    // For N=4, m=3: X[k] = exp(-3*pi*i*k/2) = [1, i, -1, -i]
   
    // Shifted impulse at index 2:
    // DFT([0,0,1,0]) = [1, -1, 1, -1]
    {
        ComplexVec input = {
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(1.0, 0.0),
            Complex(0.0, 0.0)
        };
 
        ComplexVec expected = {
            Complex( 1.0, 0.0),
            Complex(-1.0, 0.0),
            Complex( 1.0, 0.0),
            Complex(-1.0, 0.0)
        };
 
        if(!run_known_output_case("shifted_impulse_index2_n4", input, expected, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("shifted_impulse_index2_n4");
        }
    }


    // Shifted impulse at index 3:
    // DFT([0,0,0,1]) = [1, i, -1, -i]
    {
        ComplexVec input = {
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(1.0, 0.0)
        };
 
        ComplexVec expected = {
            Complex( 1.0,  0.0),
            Complex( 0.0,  1.0),
            Complex(-1.0,  0.0),
            Complex( 0.0, -1.0)
        };
 
        if(!run_known_output_case("shifted_impulse_index3_n4", input, expected, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("shifted_impulse_index3_n4");
        }
    }
 
    
    // IDFT of known spectrum
    // The inverse of the constant-spectrum case:
    //   IDFT([4,0,0,0]) = [1,1,1,1]
    // and the impulse case:
    //   IDFT([1,1,1,1]) = [1,0,0,0]
    // These are the duals of the known-output DFT tests above,
    // providing symmetric coverage of both directions.
   
 
    // IDFT([4,0,0,0]) = [1,1,1,1]
    {
        ComplexVec input = {
            Complex(4.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0)
        };
 
        ComplexVec expected = {
            Complex(1.0, 0.0),
            Complex(1.0, 0.0),
            Complex(1.0, 0.0),
            Complex(1.0, 0.0)
        };
 
        if(!run_inverse_known_output_case("idft_known_constant_spectrum_n4", input, expected, ITransform_1d::IDFT, abs_tol, rel_tol)){
            failed_tests.push_back("idft_known_constant_spectrum_n4");
        }
    }
 
    // IDFT([1,1,1,1]) = [1,0,0,0]
    {
        ComplexVec input = {
            Complex(1.0, 0.0),
            Complex(1.0, 0.0),
            Complex(1.0, 0.0),
            Complex(1.0, 0.0)
        };
 
        ComplexVec expected = {
            Complex(1.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0)
        };
 
        if(!run_inverse_known_output_case("idft_known_flat_spectrum_n4", input, expected, ITransform_1d::IDFT, abs_tol, rel_tol)){
            failed_tests.push_back("idft_known_flat_spectrum_n4");
        }
    }
 
   
    // Larger N round-trip tests
    // Small N tests (N=2, N=4) catch obvious bugs but miss
    // indexing and twiddle errors that only appear at larger sizes.
    // N=8 and N=16 cover the next two power-of-two levels.
   
 
    // Real-valued round trip N=8
    {
        ComplexVec input = {
            Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(3.0, 0.0), Complex(4.0, 0.0),
            Complex(5.0, 0.0), Complex(6.0, 0.0), Complex(7.0, 0.0), Complex(8.0, 0.0)
        };
 
        if(!run_round_trip_case("round_trip_real_n8", input, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_n8");
        }
    }
 
    // Complex-valued round trip N=16
    {
        ComplexVec input(16);
        for(std::size_t j = 0; j < 16; j++){
            // Deterministic non-trivial values: real part = cos(j), imag part = sin(j)
            input[j] = Complex(std::cos(static_cast<Real>(j)), std::sin(static_cast<Real>(j)));
        }
 
        if(!run_round_trip_case("round_trip_complex_n16", input, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_n16");
        }
    }
 
    
    // Linearity tests
    // DFT(alpha*x + beta*y) == alpha*DFT(x) + beta*DFT(y)
    // Tested with real scalars and complex scalars separately.
    
 
    // Real scalar linearity N=4
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
 
        if(!run_linearity_case("linearity_real_scalars_n4", x, y, alpha, beta, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_real_scalars_n4");
        }
    }
 
    // Complex scalar linearity N=8
    {
        ComplexVec x(8), y(8);
        for(std::size_t j = 0; j < 8; j++){
            x[j] = Complex(static_cast<Real>(j),      1.0);
            y[j] = Complex(static_cast<Real>(8 - j), -0.5);
        }
        Complex alpha = {1.0,  2.0};
        Complex beta  = {0.5, -1.0};
 
        if(!run_linearity_case("linearity_complex_scalars_n8", x, y, alpha, beta, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_complex_scalars_n8");
        }
    }


    // ------------------------------------------------------------
    // Parseval's identity tests
    // sum_j |x[j]|^2 == (1/N) * sum_k |X[k]|^2
    // Fundamental energy conservation identity. If this fails,
    // the normalization convention is broken and the spectral
    // solver will produce incorrect energy dissipation rates.
    // ------------------------------------------------------------

    // Parseval: real-valued input N=4
    // Physical energy = 1^2 + 2^2 + 3^2 + 4^2 = 30
    {
        ComplexVec input = {
            Complex(1.0, 0.0), Complex(2.0, 0.0),
            Complex(3.0, 0.0), Complex(4.0, 0.0)
        };

        if(!run_parseval_case("parseval_real_n4", input, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_real_n4");
        }
    }

    // Parseval: complex-valued input N=4
    // Tests that std::norm correctly accumulates |real|^2 + |imag|^2
    {
        ComplexVec input = {
            Complex( 1.0,  2.0), Complex(-1.0,  3.0),
            Complex( 2.0, -1.0), Complex( 0.0,  1.0)
        };

        if(!run_parseval_case("parseval_complex_n4", input, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_complex_n4");
        }
    }

    // Parseval: single Fourier mode N=8
    // input[j] = exp(2*pi*i*j/8), energy = N = 8
    // Spectral energy = (1/8) * N^2 = 8, must be consistent
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), std::sin(theta));
        }

        if(!run_parseval_case("parseval_single_mode_n8", input, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_single_mode_n8");
        }
    }

    // Parseval: real sinusoidal input N=16
    // x[j] = cos(2*pi*j/16), energy split symmetrically
    // between modes k=1 and k=15
    {
        std::size_t N = 16;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), 0.0);
        }

        if(!run_parseval_case("parseval_cosine_n16", input, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_cosine_n16");
        }
    }


    // ------------------------------------------------------------
    // Time-shift property tests
    // DFT(x[j - m])[k] = exp(-2*pi*i*k*m/N) * X[k]
    // A circular right-shift by m in physical space multiplies
    // each frequency bin k by a linear phase ramp. If this fails,
    // the twiddle factor sign convention or circular indexing is wrong.
    // ------------------------------------------------------------

    // Time shift: impulse at 0 shifted by 1, N=4
    // x = [1,0,0,0], shifted = [0,1,0,0]
    // DFT(shifted)[k] = exp(-2*pi*i*k/4) * DFT(x)[k]
    {
        ComplexVec input = {
            Complex(1.0, 0.0), Complex(0.0, 0.0),
            Complex(0.0, 0.0), Complex(0.0, 0.0)
        };

        if(!run_time_shift_case("time_shift_impulse_m1_n4", input, 1, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("time_shift_impulse_m1_n4");
        }
    }

    // Time shift: real sinusoid shifted by 2, N=8
    // Shifting by N/2 flips the sign of all odd-frequency bins
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), 0.0);
        }

        if(!run_time_shift_case("time_shift_cosine_m2_n8", input, 2, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("time_shift_cosine_m2_n8");
        }
    }

    // Time shift: complex input shifted by 3, N=8
    // Non-trivial complex input exercises full phase ramp
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            input[j] = Complex(std::cos(static_cast<Real>(j)), std::sin(static_cast<Real>(j)));
        }

        if(!run_time_shift_case("time_shift_complex_m3_n8", input, 3, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("time_shift_complex_m3_n8");
        }
    }

    // Time shift: shift by N gives identity, N=8
    // Shifting by exactly N is a full period, must return original spectrum
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            input[j] = Complex(static_cast<Real>(j + 1), 0.0);
        }

        if(!run_time_shift_case("time_shift_full_period_n8", input, 8, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("time_shift_full_period_n8");
        }
    }

    // Time shift: negative shift (left shift) by -1, N=8
    // Tests that the wrapping formula handles negative shifts correctly
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            input[j] = Complex(std::cos(static_cast<Real>(j)), 0.0);
        }

        if(!run_time_shift_case("time_shift_negative_m1_n8", input, -1, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("time_shift_negative_m1_n8");
        }
    }


    // ------------------------------------------------------------
    // Conjugate symmetry tests
    // For real-valued input: X[k] = conj(X[(N-k) % N])
    // X[0] and X[N/2] must be purely real.
    // The heat solver always starts from a real temperature field,
    // so this symmetry is a direct precondition for the solver
    // output to also be real-valued.
    // ------------------------------------------------------------

    // Conjugate symmetry: constant real input N=4
    {
        ComplexVec input = {
            Complex(1.0, 0.0), Complex(1.0, 0.0),
            Complex(1.0, 0.0), Complex(1.0, 0.0)
        };

        if(!run_conjugate_symmetry_case("conj_sym_constant_n4", input, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_constant_n4");
        }
    }

    // Conjugate symmetry: arbitrary real input N=8
    {
        ComplexVec input = {
            Complex(3.0, 0.0), Complex(1.0, 0.0), Complex(-2.0, 0.0), Complex(0.5, 0.0),
            Complex(7.0, 0.0), Complex(-1.0, 0.0), Complex(4.0, 0.0), Complex(2.0, 0.0)
        };

        if(!run_conjugate_symmetry_case("conj_sym_arbitrary_n8", input, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_arbitrary_n8");
        }
    }

    // Conjugate symmetry: cosine input N=16
    // x[j] = cos(2*pi*j/16), energy at k=1 and k=15 only
    // These two bins must be complex conjugates of each other
    {
        std::size_t N = 16;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), 0.0);
        }

        if(!run_conjugate_symmetry_case("conj_sym_cosine_n16", input, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_cosine_n16");
        }
    }


    // ------------------------------------------------------------
    // Modulation property tests
    // DFT(x[j] * exp(2*pi*i*k0*j/N))[k] = X[(k - k0) mod N]
    // Multiplying by a complex exponential in physical space
    // circularly shifts the spectrum. This is the dual of the
    // time-shift property. If this fails, frequency-bin indexing
    // or the physical-space phase accumulation is wrong.
    // ------------------------------------------------------------

    // Modulation: impulse input shifted by 1 frequency bin, N=4
    // x = [1,0,0,0], DFT(x) = [1,1,1,1]
    // After modulation by k0=1: spectrum shifts left by 1 -> [1,1,1,1] (invariant here)
    {
        ComplexVec input = {
            Complex(1.0, 0.0), Complex(0.0, 0.0),
            Complex(0.0, 0.0), Complex(0.0, 0.0)
        };

        if(!run_modulation_case("modulation_impulse_k1_n4", input, 1, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_impulse_k1_n4");
        }
    }

    // Modulation: cosine input shifted by 1 frequency bin, N=8
    // x[j] = cos(2*pi*j/8), spectrum has energy at k=1 and k=7
    // After shift by k0=1: energy moves to k=0 and k=6
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            Real theta = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(N);
            input[j] = Complex(std::cos(theta), 0.0);
        }

        if(!run_modulation_case("modulation_cosine_k1_n8", input, 1, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_cosine_k1_n8");
        }
    }

    // Modulation: complex input shifted by 3 frequency bins, N=8
    // Non-trivial complex input, larger shift exercises full circular wrap
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            input[j] = Complex(std::cos(static_cast<Real>(j)), std::sin(static_cast<Real>(j)));
        }

        if(!run_modulation_case("modulation_complex_k3_n8", input, 3, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_complex_k3_n8");
        }
    }

    // Modulation: shift by N gives identity, N=8
    // Full period shift in frequency space must return original spectrum
    {
        std::size_t N = 8;
        ComplexVec input(N);
        for(std::size_t j = 0; j < N; j++){
            input[j] = Complex(static_cast<Real>(j + 1), 0.0);
        }

        if(!run_modulation_case("modulation_full_period_n8", input, 8, Transform_1d::DFT, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_full_period_n8");
        }
    }
 


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

