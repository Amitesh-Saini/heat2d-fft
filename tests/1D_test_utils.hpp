#pragma once

#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include <stdexcept>
#include <algorithm>

#include "types.hpp"

enum class Transform_1d {
        DFT,
        FFT 
    };
    
enum class ITransform_1d {
        IDFT,
        IFFT 
    };


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

bool approx_equal_complex(const Complex& a, const Complex& b, Real abs_tol = 1e-12, Real rel_tol = 1e-12);


// Same as approx_equal_complex but for scalars

bool approx_equal_real(const Real& a, const Real&b, Real abs_tol = 1e-12, Real rel_tol = 1e-12);



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


bool approx_equal_vector(const ComplexVec& expected, const ComplexVec& actual, Real abs_tol = 1e-12,
 Real rel_tol = 1e-12);



// Returns the maximum absolute entrywise error between two
// complex vectors.
//
// What it calculates:
//   max_k |expected[k] - actual[k]|
//
// Why this is useful:
//   This gives the worst-case pointwise discrepancy and is
//   often the fastest way to understand how badly a test failed.


Real max_abs_error(const ComplexVec& expected, const ComplexVec& actual);



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


Real relative_l2_error(const ComplexVec& expected, const ComplexVec& actual, Real abs_tol);


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


Real relative_inf_error(const ComplexVec& expected, const ComplexVec& actual, Real abs_tol);


// Prints a detailed numerical failure report for a failed 1D transform test.
//
// What it reports:
//   - test name
//   - expected output vector
//   - actual computed output vector
//   - maximum absolute error
//   - relative L2 error
//   - relative infinity-norm error
//
// Why this is useful:
//   When a 1D transform test fails, this provides enough numerical
//   detail to diagnose common issues such as scaling mistakes,
//   sign convention errors, indexing bugs, ordering problems,
//   or more general implementation errors.
//
// Notes:
//   This function is transform-agnostic. It can be used for failures
//   from forward DFT, forward FFT, and later inverse transform tests,
//   as long as the test compares an expected ComplexVec against an
//   actual ComplexVec.

void print_failure_report(const std::string& test_name, const ComplexVec& expected, 
 const ComplexVec& actual, Real abs_tol);


// Prints a detailed numerical failure report for a failed scalar test.
//
// What it reports:
//   - test name
//   - expected scalar value
//   - actual scalar value
//   - absolute error
//   - relative error
//
// Why this is useful:
//   Some Fourier-property tests, such as Parseval's identity,
//   compare scalar quantities rather than vectors. This helper
//   provides a clean failure report for those cases.


void print_scalar_failure_report(const std::string& test_name, Real expected, Real actual);






// Prints a failure report for a failed conjugate-symmetry check.
//
// What it reports:
//   - test name
//   - frequency index k
//   - mirror frequency index (N-k) mod N
//   - X[k]
//   - conj(X[mirror])
//   - absolute error
//
// Why this is useful:
//   Conjugate-symmetry tests fail at specific frequency pairs.
//   Reporting the failing pair makes it easier to diagnose indexing,
//   sign-convention, or frequency-ordering bugs.
void print_conjugate_symmetry_failure_report( const std::string& test_name, std::size_t k, std::size_t mirror,
    Complex lhs, Complex rhs);

// Runs a known-output 1D transform test case.
//
// What it does:
//   - applies the selected 1D transform to the input vector
//   - compares the computed output against a known expected vector
//   - returns true if the result passes the tolerance checks
//   - prints a failure report if the test fails
//
// Why this is useful:
//   This is the main driver for analytic sanity checks such as
//   zero input, constant input, impulse input, shifted impulse input,
//   alternating input, and other small hand-computable vectors.
//
// Notes:
//   The transform is selected through the Transform_1d enum.
//   This runner currently supports forward DFT and forward FFT

bool run_known_output_case(const std::string& test_name, const ComplexVec& input, 
 const ComplexVec& expected, Transform_1d transform, Real abs_tol = 1e-12, Real rel_tol = 1e-12);





 // Runs a 1D transform round-trip consistency test.
//
// What it does:
//   - applies the selected forward 1D transform to the input vector
//   - applies the corresponding inverse 1D transform to the transformed data
//   - compares the reconstructed vector against the original input
//   - returns true if reconstruction is accurate within tolerance
//   - prints a failure report if the test fails
//
// Why this is useful:
//   This checks whether the forward and inverse transform pair
//   are internally consistent, including sign convention,
//   normalization, and correct inversion of the transform.
//
// Notes:
//   The transform pair is selected through the Transform_1d enum.
//   This runner currently supports the forward/inverse DFT pair
//   and the forward/inverse FFT pair.


bool run_round_trip_case(const std::string& test_name, const ComplexVec& input, Transform_1d transform, 
 Real abs_tol = 1e-12, Real rel_tol = 1e-12);






// Runs a 1D transform linearity test.
//
// What it does:
//   - builds the combined input alpha*x + beta*y in physical space
//   - applies the selected 1D transform to the combined input
//   - separately applies the same transform to x and y
//   - forms alpha*T(x) + beta*T(y) in transform space
//   - compares the two results entry-by-entry
//   - returns true if they agree within tolerance
//   - prints a failure report if the test fails
//
// Why this is useful:
//   Linearity is a fundamental algebraic property of both the DFT
//   and the FFT. If this test fails, the implementation has a
//   structural error, even if some known-output tests still pass.
//
// Notes:
//   The transform is selected through the Transform_1d enum.
//   This runner currently supports:
//     - DFT
//     - FFT


bool run_linearity_case(const std::string& test_name, const ComplexVec& x, const ComplexVec& y, 
 Complex alpha, Complex beta, Transform_1d transform, Real abs_tol = 1e-12, Real rel_tol = 1e-12);





 // Runs a Parseval / energy-conservation test for a 1D transform.
//
// What it does:
//   - computes the physical-space energy:  E_phys = sum_j |x[j]|^2
//   - applies the selected 1D transform to the input vector
//   - computes the spectral-space energy:  E_spec = (1/N) * sum_k |X[k]|^2
//   - compares E_phys and E_spec within tolerance
//   - returns true if the energy relation holds
//   - prints a numerical failure report if the test fails
//
// Parseval's identity for this convention:
//   sum_j |x[j]|^2  ==  (1/N) * sum_k |X[k]|^2
//
// where X = forward DFT/FFT of x, using the unnormalized forward transform
// and 1/N normalization in the inverse. This is the convention used
// throughout this project and must be consistent with fft_1d_inplace
// and dft_1d.
//
// Why this is useful:
//   Parseval's identity is a fundamental energy-conservation invariant.
//   The spectral heat solver applies modal decay exp(-alpha*(kx^2+ky^2)*t)
//   to each Fourier coefficient. For that to correctly model energy
//   dissipation, the energy accounting between physical and spectral
//   space must be exact. A failure here indicates a normalization or
//   scaling bug that would corrupt the solver output.

bool run_parseval_case(const std::string& test_name, const ComplexVec& input, Transform_1d transform,
 Real abs_tol = 1e-12, Real rel_tol = 1e-12);




// Runs a circular time-shift property test for a 1D transform.
//
// What it does:
//   - checks that the input vector is nonempty
//   - builds a circularly shifted version of the input vector
//   - applies the selected 1D transform to the shifted input
//   - applies the same transform to the original input
//   - multiplies the original transform by the theoretical phase factor
//     predicted by the Fourier time-shift property
//   - compares the direct transform of the shifted input against the
//     phase-shifted transform of the original input
//   - returns true if the identity holds within tolerance
//   - prints a failure report if the test fails
//
// Why this is useful:
//   The time-shift property is a fundamental Fourier identity.
//   This test helps detect sign errors, circular-indexing bugs,
//   phase-factor mistakes, and incorrect frequency-bin handling.
//
// Notes:
//   This test uses a right circular shift:
//     shifted[i] = input[(i - shift) mod N].
//   With the project's unnormalized forward transform convention,
//   the expected phase factor is:
//     exp(-i * 2*pi*k*shift/N).

bool run_time_shift_case(const std::string& test_name, const ComplexVec& input, int shift, Transform_1d transform,
 Real abs_tol = 1e-12, Real rel_tol = 1e-12);



 // Runs a conjugate-symmetry test for a real-valued 1D input.
//
// What it does:
//   - applies the selected 1D transform to a real-valued input vector
//   - checks that the transform output satisfies the expected
//     conjugate-symmetry relation across frequency bins
//   - returns true if the symmetry relation holds within tolerance
//   - prints a failure report if the test fails
//
// Why this is useful:
//   For real-valued input data, Fourier coefficients must satisfy
//   conjugate symmetry. This is especially important for later
//   heat-equation and spectral-solver work, since the physical
//   temperature field is real-valued.

bool run_conjugate_symmetry_case(const std::string& test_name, const ComplexVec& input, Transform_1d transform,
 Real abs_tol = 1e-12, Real rel_tol = 1e-12);



 // Runs a known-output inverse-transform consistency test.
//
// What it does:
//   - applies the selected inverse 1D transform to the input spectrum
//   - compares the reconstructed output against a known expected vector
//   - returns true if the inverse transform output agrees within tolerance
//   - prints a failure report if the test fails
//
// Why this is useful:
//   This provides a direct correctness check for the inverse transform,
//   rather than only verifying it indirectly through round-trip tests.

bool run_inverse_known_output_case(const std::string& test_name, const ComplexVec& spectrum, 
 const ComplexVec& expected, ITransform_1d transform, Real abs_tol = 1e-12, Real rel_tol = 1e-12);




// Runs a modulation / frequency-shift property test for a 1D transform.
//
// What it does:
//   - modulates the input vector by a complex exponential in physical space
//   - applies the selected 1D transform to the modulated input
//   - applies the same transform to the original input
//   - constructs the theoretically shifted transform predicted by the
//     Fourier modulation property
//   - compares the two transform-space results entry-by-entry
//   - returns true if they agree within tolerance
//   - prints a failure report if the test fails
//
// Why this is useful:
//   The modulation property is a core Fourier identity. This test helps
//   detect frequency-indexing mistakes, sign errors, and incorrect phase
//   conventions.

bool run_modulation_case(const std::string& test_name, const ComplexVec& input, int frequency_shift,
 Transform_1d transform, Real abs_tol = 1e-12, Real rel_tol = 1e-12);

    

// Runs an FFT-vs-DFT agreement test for a 1D input.
//
// What it does:
//   - checks that the input vector is nonempty
//   - computes the reference output using dft_1d(input)
//   - computes the FFT output using fft_1d_inplace on a copy of input
//   - compares the FFT result against the DFT result entry-by-entry
//   - returns true if both outputs agree within tolerance
//   - prints a failure report if the test fails
//
// Why this is useful:
//   The DFT is the direct mathematical definition of the transform,
//   while the FFT is the fast algorithm meant to compute the same result.
//   This test verifies that the FFT implementation agrees with the DFT
//   reference for the same input and transform convention.
//
// Notes:
//   This helper is specifically for validating the forward FFT against
//   the forward DFT. It does not use Transform_1d because the comparison
//   always involves both algorithms.
bool run_fft_vs_dft_case(const std::string& test_name, const ComplexVec& input, Real abs_tol = 1e-12, Real rel_tol = 1e-12);







