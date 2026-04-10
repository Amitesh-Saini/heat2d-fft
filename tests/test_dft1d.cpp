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
bool approx_equal_complex(const Complex& a,
                          const Complex& b,
                          double abs_tol = 1e-12,
                          double rel_tol = 1e-12);


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
bool approx_equal_vector(const ComplexVec& expected,
                         const ComplexVec& actual,
                         double abs_tol = 1e-12,
                         double rel_tol = 1e-12);


// Returns the maximum absolute entrywise error between two
// complex vectors.
//
// What it calculates:
//   max_k |expected[k] - actual[k]|
//
// Why this is useful:
//   This gives the worst-case pointwise discrepancy and is
//   often the fastest way to understand how badly a test failed.
double max_abs_error(const ComplexVec& expected,
                     const ComplexVec& actual);


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
double relative_l2_error(const ComplexVec& expected,
                         const ComplexVec& actual);


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
double relative_inf_error(const ComplexVec& expected,
                          const ComplexVec& actual);


// Prints a detailed failure report for a failed transform test.
//
// What it should report:
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
void print_failure_report(const std::string& test_name,
                          const ComplexVec& expected,
                          const ComplexVec& actual);


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
bool run_known_output_case(const std::string& test_name,
                           const ComplexVec& input,
                           const ComplexVec& expected,
                           double abs_tol = 1e-12,
                           double rel_tol = 1e-12);


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
bool run_round_trip_case(const std::string& test_name,
                         const ComplexVec& input,
                         double abs_tol = 1e-12,
                         double rel_tol = 1e-12);

int main() {

    std::vector<std::string> failed_tests;

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

