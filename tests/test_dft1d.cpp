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
#include "dft1d.hpp"
#include "types.hpp"
#include <stdexcept>


#include "dft1d.hpp"
#include <string>

// Returns true if two complex numbers are approximately equal
// within combined absolute/relative tolerances.
//
// What it checks:
//   |a - b| <= abs_tol + rel_tol * max(|a|, |b|)
//
// Why this is needed:
//   Floating-point arithmetic introduces small roundoff error,
//   so exact equality is usually the wrong criterion for
//   transform outputs.


bool approx_equal_complex(const Complex& a, const Complex& b, double abs_tol = 1e-12, double rel_tol = 1e-12){

    return std::abs(a-b) <= abs_tol + rel_tol * std::max(std::abs(a), std::abs(b));
}


// Returns true if two complex vectors are approximately equal
// entry-by-entry within combined absolute/relative tolerances.
//
// What it checks:
//   - same vector length
//   - each entry satisfies approx_equal_complex(...)
//
// Why this is needed:
//   DFT outputs are complex-valued vectors, so this is the main
//   test for checking whether a computed transform matches the
//   expected reference output.


bool approx_equal_vector(const ComplexVec& expected, const ComplexVec& actual, double abs_tol = 1e-12, double rel_tol = 1e-12){

    if(expected.size() != actual.size()) return false;

    for(std::size_t k = 0; k < expected.size(); k++){

        if(!approx_equal_complex(expected[k], actual[k], abs_tol, rel_tol)) return false;

    }
    return true;
}


// Returns the maximum absolute entrywise error between two
// complex vectors.
//
// What it calculates:
//   max_k |expected[k] - actual[k]|
//
// Why this is useful:
//   This gives the worst-case pointwise discrepancy and is
//   often the fastest way to understand how badly a test failed.


double max_abs_error(const ComplexVec& expected, const ComplexVec& actual){
    
    if(expected.size() != actual.size()) throw std::invalid_argument("Expected and actual vector sizes do not match");

    Real max_error = 0.0;

    for(std::size_t k = 0; k < expected.size(); k++){

        Real current_error = std::abs(expected[k] - actual[k]);

        if(current_error > max_error) max_error = current_error;
    }

    return max_error;
}


// Returns the relative L2 error between two complex vectors.
//
// What it calculates:
//   ||expected - actual||_2 / ||expected||_2
//
// Special case:
//   If ||expected||_2 = 0, return ||actual - expected||_2
//   as an absolute error instead.
//
// Why this is useful:
//   This gives a global energy-style error measure and is a
//   standard numerical metric for comparing computed outputs
//   against reference data.


double relative_l2_error(const ComplexVec& expected, const ComplexVec& actual){

    if(expected.size() != actual.size()) throw std::invalid_argument("Expected and actual vector sizes do not match");

    Real numerator = 0.0;
    Real denominator = 0.0;

    for(std::size_t k = 0; k < expected.size(); k++){

        numerator += std::norm(expected[k] - actual[k]);
        denominator += std::norm(expected[k]);
    }

    if(denominator == 0.0) return std::sqrt(numerator);

    return std::sqrt(numerator/denominator);
}


// Returns the relative infinity-norm error between two complex vectors.
//
// What it calculates:
//   ||expected - actual||_inf / ||expected||_inf
//
// where
//   ||v||_inf = max_k |v[k]|
//
// Special case:
//   If ||expected||_inf = 0, return ||actual - expected||_inf
//   as an absolute error instead.
//
// Why this is useful:
//   This measures the worst relative entrywise error and is
//   useful for detecting a single badly wrong Fourier coefficient.


double relative_inf_error(const ComplexVec& expected, const ComplexVec& actual){

    if(expected.size() != actual.size()) throw std::invalid_argument("Expected and actual vector sizes do not match");

    Real max_error = 0.0;
    Real max_ref = 0.0;

    for(std::size_t k = 0; k < expected.size(); k++){

        Real error_k = std::abs(expected[k] - actual[k]);
        Real ref_k = std::abs(expected[k]); 

        if(error_k > max_error) max_error = error_k;
        if(ref_k > max_ref) max_ref = ref_k;
    }

    if(max_ref == 0.0) return max_error;

    return max_error/max_ref;
}


// Prints a detailed failure report for a failed transform test.
//
// What it reports:
//   - test name
//   - expected vector
//   - actual vector
//   - max absolute error
//   - relative L2 error
//   - relative infinity error
//
// Why this is useful:
//   When a DFT test fails, this gives enough numerical detail
//   to debug whether the issue is scaling, sign convention,
//   indexing, or a more general implementation bug.


void print_failure_report(const std::string& test_name, const ComplexVec& expected, const ComplexVec& actual){

    std::cout << "FAIL: " << test_name << "\n";

    if(expected.size() != actual.size()) {
        std::cout << "Size mismatch:\n";
        std::cout << "  expected.size() = " << expected.size() << "\n";
        std::cout << "  actual.size()   = " << actual.size() << "\n";
        return;
    }

    std::cout << std::setprecision(16);

    std::cout << "Expected vector:\n";
    for(std::size_t k = 0; k < expected.size(); ++k) {
        std::cout << "  [" << k << "] = ("
                  << expected[k].real() << ", "
                  << expected[k].imag() << ")\n";
    }

    std::cout << "Actual vector:\n";
    for(std::size_t k = 0; k < actual.size(); ++k) {
        std::cout << "  [" << k << "] = ("
                  << actual[k].real() << ", "
                  << actual[k].imag() << ")\n";
    }

    std::cout << "Error metrics:\n";
    std::cout << "  max_abs_error      = " << max_abs_error(expected, actual) << "\n";
    std::cout << "  relative_l2_error  = " << relative_l2_error(expected, actual) << "\n";
    std::cout << "  relative_inf_error = " << relative_inf_error(expected, actual) << "\n";
}

// Runs a known-output DFT test case.
//
// What it does:
//   - computes dft_1d(input)
//   - compares the computed output against a known expected vector
//   - returns true if the result passes tolerance checks
//   - prints a failure report if the test fails
//
// Why this is useful:
//   This is the main driver for analytic sanity checks such as
//   zero input, constant input, impulse input, and small
//   hand-computable vectors.


bool run_known_output_case(const std::string& test_name, const ComplexVec& input, const ComplexVec& expected, double abs_tol = 1e-12, double rel_tol = 1e-12){

    ComplexVec spectrum = dft_1d(input);

    if(!approx_equal_vector(expected, spectrum, abs_tol, rel_tol)) {
        print_failure_report(test_name, expected, spectrum);
        return false;
    }
    
    return true;
}


// Runs a DFT/IDFT round-trip consistency test.
//
// What it does:
//   - computes dft_1d(input)
//   - computes idft_1d(dft_1d(input))
//   - compares the reconstructed vector against the original input
//   - returns true if reconstruction is accurate within tolerance
//   - prints a failure report if the test fails
//
// Why this is useful:
//   This checks whether the forward and inverse transform
//   conventions are internally consistent, including sign choice
//   and normalization by N in the inverse.


bool run_round_trip_case(const std::string& test_name, const ComplexVec& input, double abs_tol = 1e-12, double rel_tol = 1e-12){

    ComplexVec spectrum = dft_1d(input);
    ComplexVec reconstructed_input = idft_1d(spectrum);

    if(!approx_equal_vector(input, reconstructed_input, abs_tol, rel_tol)){
        print_failure_report(test_name, input, reconstructed_input);
        return false;
    }

    return true;

}





// Runs a linearity test: DFT(alpha*x + beta*y) == alpha*DFT(x) + beta*DFT(y).
//
// What it does:
//   - computes DFT(alpha*x + beta*y) directly
//   - computes alpha*DFT(x) + beta*DFT(y) from separate transforms
//   - compares the two results entry-by-entry
//   - returns true if they agree within tolerance
//   - prints a failure report if the test fails
//
// Why this is important:
//   Linearity is a fundamental algebraic property of the DFT.
//   If this fails, the implementation has a structural bug regardless
//   of whether known-output tests pass.


bool run_linearity_case(const std::string& test_name, const ComplexVec& x, const ComplexVec& y, Complex alpha, Complex beta,
    double abs_tol = 1e-12, double rel_tol = 1e-12) {

    std::size_t N = x.size();
 
    // Build alpha*x + beta*y in physical space
    ComplexVec combined(N);
    for(std::size_t j = 0; j < N; j++){
        combined[j] = alpha * x[j] + beta * y[j];
    }
 
    // DFT of the combined signal
    ComplexVec lhs = dft_1d(combined);
 
    // alpha*DFT(x) + beta*DFT(y) in frequency space
    ComplexVec dx = dft_1d(x);
    ComplexVec dy = dft_1d(y);
    ComplexVec rhs(N);
    for(std::size_t k = 0; k < N; k++){
        rhs[k] = alpha * dx[k] + beta * dy[k];
    }
 
    if(!approx_equal_vector(lhs, rhs, abs_tol, rel_tol)){
        print_failure_report(test_name, lhs, rhs);
        return false;
    }
 
    return true;
}



int main() {

    std::vector<std::string> failed_tests;

    const double abs_tol = 1e-12;
    const double rel_tol = 1e-12;

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

        if(!run_known_output_case("zero_vector_n4", input, expected, abs_tol, rel_tol)) {
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

        if(!run_known_output_case("constant_vector_n4", input, expected, abs_tol, rel_tol)) {
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

        if(!run_known_output_case("impulse_index0_n4", input, expected, abs_tol, rel_tol)) {
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

        if(!run_known_output_case("shifted_impulse_index1_n4", input, expected, abs_tol, rel_tol)) {
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

        if(!run_known_output_case("hand_check_n2_real", input, expected, abs_tol, rel_tol)) {
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

        if(!run_known_output_case("alternating_real_n4", input, expected, abs_tol, rel_tol)) {
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

        if(!run_known_output_case("single_complex_mode_n4", input, expected, abs_tol, rel_tol)) {
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

        if(!run_round_trip_case("round_trip_real_n4", input, abs_tol, rel_tol)) {
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

        if(!run_round_trip_case("round_trip_complex_n4", input, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_n4");
        }
    }

    // Another small round-trip case, N=2
    {
        ComplexVec input = {
            Complex(2.0, -1.0),
            Complex(-0.5, 3.0)
        };

        if(!run_round_trip_case("round_trip_complex_n2", input, abs_tol, rel_tol)) {
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
 
        if(!run_known_output_case("shifted_impulse_index2_n4", input, expected, abs_tol, rel_tol)) {
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
 
        if(!run_known_output_case("shifted_impulse_index3_n4", input, expected, abs_tol, rel_tol)) {
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
 
        ComplexVec actual = idft_1d(input);
 
        if(!approx_equal_vector(expected, actual, abs_tol, rel_tol)){
            print_failure_report("idft_known_constant_spectrum_n4", expected, actual);
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
 
        ComplexVec actual = idft_1d(input);
 
        if(!approx_equal_vector(expected, actual, abs_tol, rel_tol)){
            print_failure_report("idft_known_flat_spectrum_n4", expected, actual);
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
 
        if(!run_round_trip_case("round_trip_real_n8", input, abs_tol, rel_tol)) {
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
 
        if(!run_round_trip_case("round_trip_complex_n16", input, abs_tol, rel_tol)) {
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
 
        if(!run_linearity_case("linearity_real_scalars_n4", x, y, alpha, beta, abs_tol, rel_tol)) {
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
 
        if(!run_linearity_case("linearity_complex_scalars_n8", x, y, alpha, beta, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_complex_scalars_n8");
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

