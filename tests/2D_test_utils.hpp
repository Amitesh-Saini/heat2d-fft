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
#include <cstddef>


#include "types.hpp"
#include "grid2d.hpp"


// Selects which forward 2D transform a test runner should apply.
// DFT2D is the slow mathematical reference transform.
// FFT2D is the fast radix-2 implementation being tested.
enum class Transform_2d {
    DFT2D,
    FFT2D
};


// Selects which inverse 2D transform a test runner should apply.
// IDFT2D is the direct inverse reference transform.
// IFFT2D is the fast inverse radix-2 implementation.
enum class ITransform_2d {
    IDFT2D,
    IFFT2D
};


// ------------------------------------------------------------
// Grid comparison and error helpers
// ------------------------------------------------------------


// Returns true if two complex grids have the same shape and all entries
// agree within the given absolute/relative tolerances.
// This is the main correctness comparison helper used by almost every
// 2D transform test.
bool approx_equal_grid(const Grid2D<Complex>& a, const Grid2D<Complex>& b, Real abs_tol, Real rel_tol);


// Computes the maximum absolute complex-entry error between two grids.
// Useful for failure reports because it tells you the worst pointwise
// error in the transformed or reconstructed grid.
Real max_abs_error_grid(const Grid2D<Complex>& expected, const Grid2D<Complex>& actual);


// Computes the relative L2 error between two grids, treating the grid
// as one flattened vector of complex values.
// Useful for measuring total reconstruction error in round-trip tests.
Real relative_l2_error_grid(const Grid2D<Complex>& expected, const Grid2D<Complex>& actual, Real abs_tol);


// Computes the relative infinity-norm error between two grids.
// Useful for catching the largest local error, especially in known-output
// tests where most coefficients should be exactly zero.
Real relative_inf_error_grid(const Grid2D<Complex>& expected,const Grid2D<Complex>& actual, Real abs_tol);


// Prints a detailed failure report for grid-valued comparisons.
// Should include the test name, grid shape, max absolute error,
// relative L2 error, and relative infinity error.
void print_grid_failure_report(
    const std::string& test_name, const Grid2D<Complex>& expected, const Grid2D<Complex>& actual, Real abs_tol);


// Prints a failure report for scalar-valued checks such as Parseval energy,
// maximum imaginary part, or other norm/property tests.
void print_scalar_failure_report(
    const std::string& test_name, Real expected, Real actual, Real abs_tol, Real rel_tol);


void print_conjugate_symmetry_2d_failure_report(
    const std::string& test_name, std::size_t kx, std::size_t ky, std::size_t mirror_kx, 
    std::size_t mirror_ky, Complex lhs, Complex rhs);


void print_spectral_decay_failure_report(
    const std::string& test_name, Real theoretical_ratio, Real actual_ratio, Real initial_physical_energy, Real final_physical_energy, 
    Real initial_spectral_energy, Real final_spectral_energy, Real max_imag, bool expect_real_output, Real abs_tol, Real rel_tol);


// ------------------------------------------------------------
// Grid construction helpers
// ------------------------------------------------------------


// Creates an nx-by-ny complex grid filled with zero.
// Useful for verifying that DFT/FFT/IDFT/IFFT all map the zero grid
// to the zero grid.
Grid2D<Complex> make_zero_grid(std::size_t nx, std::size_t ny);


// Creates an nx-by-ny complex grid where every entry equals c.
// Useful because the Fourier transform of a constant grid should have
// only the DC mode (0,0) nonzero, with value nx*ny*c.
Grid2D<Complex> make_constant_grid(std::size_t nx, std::size_t ny, Complex c);


// Creates an nx-by-ny complex grid with a single nonzero impulse at (i0,j0).
// Useful because an impulse at (0,0) transforms to an all-ones spectrum,
// while a shifted impulse produces a predictable 2D phase ramp.
Grid2D<Complex> make_impulse_grid(std::size_t nx, std::size_t ny, std::size_t i0, std::size_t j0);


// Creates a pure complex Fourier mode:
//
//   u(i,j) = exp(2*pi*i*(kx*i/nx + ky*j/ny))
//
// Useful because its transform should be zero everywhere except at
// frequency bin (kx,ky), where the coefficient should be nx*ny.
// This is one of the strongest tests for axis ordering, signs, and
// nx/ny indexing.
Grid2D<Complex> make_single_mode_grid(std::size_t nx, std::size_t ny, std::size_t kx, std::size_t ky);


// Creates a deterministic real-valued complex grid from a fixed mixture of
// low-frequency sine/cosine Fourier modes.
//
// The output values have zero imaginary part:
//
//     u(i,j) = real_value(i,j) + 0i
//
// This is useful as a manufactured test input for FFT/DFT round-trip tests,
// Parseval/energy checks, conjugate-symmetry tests for real-valued inputs,
// and smooth heat-equation-style initial conditions.
//
// Because the modes and coefficients are fixed, this helper is best for
// debugging: if a test fails, the input signal is always exactly the same.
Grid2D<Complex> make_real_mixed_mode_grid(std::size_t nx, std::size_t ny);


// Creates a randomized real-valued complex grid with entries sampled
// independently from a uniform distribution.
//
// The output values have zero imaginary part:
//
//     u(i,j) = real_value(i,j) + 0i
//
// Entries are sampled using the fixed-seed test RNG, so generated grids
// are reproducible across test runs. This is useful for stress testing
// FFT/DFT round-trip accuracy, Parseval/energy checks, and conjugate
// symmetry for real inputs.
//
// This helper should be used after deterministic manufactured-mode tests
// pass. The deterministic grid is better for debugging exact failures;
// this randomized version is better for broader coverage.
Grid2D<Complex> make_random_real_grid(std::size_t nx, std::size_t ny, Real lower_bound, Real upper_bound);


// Creates a deterministic complex-valued grid from a fixed mixture of complex
// Fourier modes.
//
// Unlike real-valued test grids, this grid has both nonzero real and imaginary
// parts:
//
//     u(i,j) = real_part(i,j) + i * imag_part(i,j)
//
// This is useful because real-only inputs can hide bugs related to complex
// arithmetic, sign conventions, phase handling, normalization, and inverse
// transform behavior.
//
// Because the modes and coefficients are fixed, this helper is best for
// repeatable debugging of fully complex FFT/DFT behavior.
Grid2D<Complex> make_complex_test_grid(std::size_t nx, std::size_t ny);


// Creates a randomized complex-valued grid with real and imaginary parts
// sampled independently from a uniform distribution.
//
//     u(i,j) = real_part(i,j) + i * imag_part(i,j)
//
// Entries are sampled using the fixed-seed test RNG, so generated grids
// are reproducible across test runs. This is useful for stress testing
// FFT/DFT round-trip accuracy, normalization, phase behavior, and
// complex-valued transform correctness.
//
// This helper should be used after deterministic complex-mode tests pass.
// The deterministic grid is better for debugging exact failures;
// this randomized version is better for broader coverage.
Grid2D<Complex> make_random_complex_grid(std::size_t nx, std::size_t ny, Real lower_bound, Real upper_bound);


// ------------------------------------------------------------
// Energy, realness, and transform-property helpers
// ------------------------------------------------------------


// Computes the physical-space energy:
//
//   sum_{i,j} |u(i,j)|^2
//
// Useful for Parseval tests and later for checking heat-equation
// energy decay.
Real physical_energy_2d(const Grid2D<Complex>& field);


// Computes the scaled spectral-space energy:
//
//   (1 / (nx*ny)) * sum_{k,l} |U(k,l)|^2
//
// for your unnormalized-forward, normalized-inverse convention.
// This should match physical_energy_2d(field) after a correct 2D FFT/DFT.
Real spectral_energy_2d(const Grid2D<Complex>& spectrum);


// Returns a circularly shifted copy of a grid.
// Positive shifts should wrap around modulo nx and ny.
// Useful for testing the 2D shift theorem.
Grid2D<Complex> circular_shift_2d(const Grid2D<Complex>& field, std::ptrdiff_t shift_x, std::ptrdiff_t shift_y);


// Multiplies a grid by a 2D complex exponential:
//
//   exp(2*pi*i*(kx*i/nx + ky*j/ny))
//
// Useful for testing the 2D modulation theorem, where modulation in
// physical space shifts the spectrum in frequency space.
Grid2D<Complex> modulate_2d(const Grid2D<Complex>& field, std::ptrdiff_t kx_shift, std::ptrdiff_t ky_shift);


// Returns the maximum absolute imaginary part over the grid.
// Useful after inverse-transforming real-valued physical data, where
// the imaginary component should be near roundoff error.
Real max_imag_part(const Grid2D<Complex>& field);


// Returns true if every entry has imaginary part below the tolerance.
// Useful for verifying that real-valued heat-equation inputs reconstruct
// to real-valued physical-space fields after inverse transforms.
bool is_real_grid_within_tol(const Grid2D<Complex>& field, Real abs_tol);


// ------------------------------------------------------------
// Test runner helpers
// ------------------------------------------------------------


// Runs a known-output forward transform test.
// Applies the selected forward transform to input and compares the result
// against an analytically known expected spectrum.
// Useful for zero grids, constant grids, impulses, checkerboards, and
// single Fourier modes.
bool run_known_output_2d_case(
    const std::string& test_name, const Grid2D<Complex>& input, const Grid2D<Complex>& expected, 
    Transform_2d transform, Real abs_tol, Real rel_tol);


// Runs a known-output inverse transform test.
// Applies the selected inverse transform to a known spectrum and compares
// against the analytically expected physical-space grid.
// Useful for checking inverse normalization and sign convention directly.
bool run_inverse_known_output_2d_case(
    const std::string& test_name, const Grid2D<Complex>& spectrum, const Grid2D<Complex>& expected, 
    ITransform_2d inverse_transform, Real abs_tol, Real rel_tol);


// Runs a forward-then-inverse round-trip test:
//
//   input -> forward transform -> inverse transform -> reconstructed input
//
// Useful for verifying sign convention, normalization, and row/column
// traversal consistency.
bool run_round_trip_2d_case(
    const std::string& test_name, const Grid2D<Complex>& input, Transform_2d transform, Real abs_tol, Real rel_tol);


// Runs a 2D linearity test:
//
//   T(alpha*x + beta*y) == alpha*T(x) + beta*T(y)
//
// Useful because Fourier transforms are linear operators. This catches
// structural mistakes in grid traversal and transform application.
bool run_linearity_2d_case(
    const std::string& test_name, const Grid2D<Complex>& x, const Grid2D<Complex>& y, Complex alpha, 
    Complex beta, Transform_2d transform, Real abs_tol, Real rel_tol);


// Runs a 2D Parseval test:
//
//   sum |u(i,j)|^2 == (1/(nx*ny)) * sum |U(k,l)|^2
//
// Useful because it checks global energy consistency under your transform
// normalization convention.
bool run_parseval_2d_case(
    const std::string& test_name, const Grid2D<Complex>& input, Transform_2d transform, Real abs_tol, Real rel_tol);


// Runs the 2D circular-shift theorem test.
// A circular shift in physical space should multiply the spectrum by
// a predictable 2D phase factor.
// Useful for catching sign mistakes and axis-ordering bugs.
bool run_shift_2d_case(
    const std::string& test_name, const Grid2D<Complex>& input, std::ptrdiff_t shift_x, std::ptrdiff_t shift_y, 
    Transform_2d transform, Real abs_tol, Real rel_tol);


// Runs the 2D conjugate-symmetry test for real-valued input:
//
//   U(k,l) == conj(U((-k mod nx), (-l mod ny)))
//
// Useful because real heat-equation fields should produce conjugate-symmetric
// spectra. If this fails, inverse transforms may produce nontrivial imaginary
// artifacts.
bool run_conjugate_symmetry_2d_case(
    const std::string& test_name, const Grid2D<Complex>& real_input, Transform_2d transform, 
    Real abs_tol, Real rel_tol);


// Runs the 2D modulation theorem test.
// Multiplying the physical grid by a complex exponential should shift the
// spectrum by the corresponding frequency-bin offset.
// Useful for checking frequency indexing and wraparound behavior.
bool run_modulation_2d_case(
    const std::string& test_name, const Grid2D<Complex>& input, std::ptrdiff_t kx_shift, std::ptrdiff_t ky_shift, 
    Transform_2d transform, Real abs_tol, Real rel_tol);


// Compares the fast 2D FFT result against the slow mathematical 2D DFT
// reference on the same input.
// This is the definitive small-N correctness check for the FFT.
bool run_fft2d_vs_dft2d_case(
    const std::string& test_name, const Grid2D<Complex>& input, Real abs_tol, Real rel_tol);


// Verifies that the 2D FFT rejects invalid dimensions.
// The radix-2 FFT requires both nx and ny to be nonzero powers of two.
// This is a contract test, not a numerical accuracy test.
bool run_power_of_two_enforcement_2d_case(const std::string& test_name, std::size_t nx, std::size_t ny);


// Runs a 2D separability test for the row-column FFT decomposition.
//
// What it does:
//   - applies the selected 2D forward transform to the input grid
//   - independently applies 1D row-wise FFTs to every row, then
//     1D column-wise FFTs to every column of the result
//   - compares the two outputs entry-by-entry within tolerance
//   - returns true if both outputs agree within tolerance
//   - prints a failure report if the test fails
//
// Why this is needed:
//   The 2D FFT is implemented as a row-column decomposition. This test
//   verifies that the 2D transform is structurally consistent with that
//   decomposition, not just numerically close to the DFT reference.
//   A failure here indicates a bug in axis ordering, row/column traversal,
//   or intermediate storage — bugs that may be invisible in round-trip
//   or Parseval tests but would corrupt wavenumber indexing in the
//   heat solver.
bool run_separability_2d_case(const std::string& test_name, const Grid2D<Complex>& input,
    Transform_2d transform, Real abs_tol, Real rel_tol);


// Runs a spectral heat decay test.
//
// What it does:
//   - applies the selected 2D forward transform to the input grid
//   - multiplies each spectral coefficient U(kx, ky) by the heat kernel
//     exp(-alpha * (kx^2 + ky^2) * t)
//   - applies the inverse transform to recover the decayed physical field
//   - computes the physical-space energy of the input and decayed field
//   - verifies that the energy ratio matches the theoretically predicted
//     decay within tolerance
//   - returns true if the energy relation holds
//   - prints a scalar failure report if the test fails
//
// Why this is needed:
//   This is a direct smoke test of the heat solver's core spectral
//   operation. It verifies that wavenumber indexing, normalization, and
//   the modal decay formula are all mutually consistent. A failure here
//   means the heat solver will produce physically wrong energy dissipation
//   even if all other FFT tests pass.
bool run_spectral_decay_2d_case(const std::string& test_name, const Grid2D<Complex>& input,
    Transform_2d transform, Real alpha, Real time, Real Lx, Real Ly, bool expect_real_output, Real abs_tol, Real rel_tol);


// Compares my 2D FFT forward and inverse transforms against FFTW3
// on the same input grid.
//
// What it does:
//   - applies fft_2d_inplace to a copy of the input and records the spectrum
//   - applies the FFTW3 forward transform to the same input and records its spectrum
//   - compares the two forward spectra using relative L2 and infinity-norm errors
//   - applies ifft_2d_inplace to my spectrum
//   - applies the FFTW3 inverse transform to the FFTW3 spectrum
//   - compares the two reconstructed grids
//   - returns true if both the forward spectra and reconstructed grids agree
//     within tolerance
//   - prints a grid failure report at the first failing checkpoint
//
// Why I need this:
//   My round-trip and DFT-agreement tests verify that my implementation is
//   internally consistent, but they cannot prove it is numerically correct
//   in an absolute sense. FFTW3 is the industry-standard reference used in
//   production HPC codes. If my FFT agrees with FFTW3 at large N (512, 1024,
//   2048) I can be confident my heat solver is built on a correct spectral
//   foundation. This is the strongest correctness guarantee I can get for
//   my implementation.
bool run_fft2d_vs_fftw_case(const std::string& test_name, const Grid2D<Complex>& input, Real abs_tol, Real rel_tol);