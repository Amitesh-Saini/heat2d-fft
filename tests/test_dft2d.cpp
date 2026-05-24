// test_dft2d.cpp
// Responsibility:
//   Test scaffold for the 2D DFT.
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

#include "dft2d.hpp"
#include "2D_test_utils.hpp"
#include "types.hpp"


int main() {

    std::vector<std::string> failed_tests;

    const Real abs_tol = 1e-10;
    const Real rel_tol = 1e-10;

    std::cout << "=== Running 2D DFT tests ===\n\n";


    // --------------------------------------------------------
    // Known-output tests
    // --------------------------------------------------------


    // Zero grid 2x2: DFT2D(0) = 0
    {
        Grid2D<Complex> input = make_zero_grid(2, 2);

        Grid2D<Complex> expected = make_zero_grid(2, 2);

        if(!run_known_output_2d_case("zero_grid_2x2", input, expected, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("zero_grid_2x2");
        }
    }

    // Zero grid 4x4: DFT2D(0) = 0
    {
        Grid2D<Complex> input = make_zero_grid(4, 4);

        Grid2D<Complex> expected = make_zero_grid(4, 4);

        if(!run_known_output_2d_case("zero_grid_4x4", input, expected, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("zero_grid_4x4");
        }
    }

    // Constant grid 2x2, c=1: DFT2D([1]) at (0,0) = nx*ny = 4, all other bins = 0
    {
        Grid2D<Complex> input = make_constant_grid(2, 2, Complex{1.0, 0.0});

        Grid2D<Complex> expected = make_zero_grid(2, 2);
        expected(0, 0) = Complex{4.0, 0.0};

        if(!run_known_output_2d_case("constant_grid_2x2_c1", input, expected, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("constant_grid_2x2_c1");
        }
    }

    // Constant grid 4x4, c=1: DFT2D at (0,0) = 16, all other bins = 0
    {
        Grid2D<Complex> input = make_constant_grid(4, 4, Complex{1.0, 0.0});

        Grid2D<Complex> expected = make_zero_grid(4, 4);
        expected(0, 0) = Complex{16.0, 0.0};

        if(!run_known_output_2d_case("constant_grid_4x4_c1", input, expected, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("constant_grid_4x4_c1");
        }
    }

    // Constant grid 2x4, c=2: DFT2D at (0,0) = nx*ny*c = 16, all other bins = 0
    {
        Grid2D<Complex> input = make_constant_grid(2, 4, Complex{2.0, 0.0});

        Grid2D<Complex> expected = make_zero_grid(2, 4);
        expected(0, 0) = Complex{16.0, 0.0};

        if(!run_known_output_2d_case("constant_grid_2x4_c2", input, expected, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("constant_grid_2x4_c2");
        }
    }

    // Impulse at (0,0) in 2x2: DFT2D = all-ones grid
    // delta(0,0) transforms to 1 everywhere
    {
        Grid2D<Complex> input = make_impulse_grid(2, 2, 0, 0);

        Grid2D<Complex> expected = make_constant_grid(2, 2, Complex{1.0, 0.0});

        if(!run_known_output_2d_case("impulse_00_2x2", input, expected, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("impulse_00_2x2");
        }
    }

    // Impulse at (0,0) in 4x4: DFT2D = all-ones grid
    {
        Grid2D<Complex> input = make_impulse_grid(4, 4, 0, 0);

        Grid2D<Complex> expected = make_constant_grid(4, 4, Complex{1.0, 0.0});

        if(!run_known_output_2d_case("impulse_00_4x4", input, expected, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("impulse_00_4x4");
        }
    }

    // Shifted impulse at (1,2) in 4x4:
    // DFT2D(delta_{1,2})[kx,ky] = exp(-2*pi*i*(kx*1/4 + ky*2/4))
    // Every entry is nonzero with magnitude 1, giving a nontrivial phase pattern.
    // This catches sign and axis-ordering bugs that the origin impulse (all-ones) cannot.
    {
        Grid2D<Complex> input = make_impulse_grid(4, 4, 1, 2);

        Grid2D<Complex> expected(4, 4);
        for(std::size_t kx = 0; kx < 4; ++kx) {
            for(std::size_t ky = 0; ky < 4; ++ky) {
                Real theta = -2.0 * PI * (static_cast<Real>(kx) * 1.0 / 4.0
                                        + static_cast<Real>(ky) * 2.0 / 4.0);
                expected(kx, ky) = Complex{std::cos(theta), std::sin(theta)};
            }
        }

        if(!run_known_output_2d_case("shifted_impulse_12_4x4", input, expected, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shifted_impulse_12_4x4");
        }
    }

    // Single Fourier mode (kx=1, ky=0) in 4x4:
    // DFT2D should produce nx*ny = 16 at bin (1,0), zero elsewhere
    {
        Grid2D<Complex> input = make_single_mode_grid(4, 4, 1, 0);

        Grid2D<Complex> expected = make_zero_grid(4, 4);
        expected(1, 0) = Complex{16.0, 0.0};

        if(!run_known_output_2d_case("single_mode_kx1_ky0_4x4", input, expected, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("single_mode_kx1_ky0_4x4");
        }
    }

    // Single Fourier mode (kx=0, ky=1) in 4x4:
    // DFT2D should produce 16 at bin (0,1), zero elsewhere
    {
        Grid2D<Complex> input = make_single_mode_grid(4, 4, 0, 1);

        Grid2D<Complex> expected = make_zero_grid(4, 4);
        expected(0, 1) = Complex{16.0, 0.0};

        if(!run_known_output_2d_case("single_mode_kx0_ky1_4x4", input, expected, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("single_mode_kx0_ky1_4x4");
        }
    }

    // Single Fourier mode (kx=1, ky=2) in 4x4:
    // DFT2D should produce 16 at bin (1,2), zero elsewhere
    // Tests that both axes are handled independently
    {
        Grid2D<Complex> input = make_single_mode_grid(4, 4, 1, 2);

        Grid2D<Complex> expected = make_zero_grid(4, 4);
        expected(1, 2) = Complex{16.0, 0.0};

        if(!run_known_output_2d_case("single_mode_kx1_ky2_4x4", input, expected, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("single_mode_kx1_ky2_4x4");
        }
    }

    // Single Fourier mode (kx=2, ky=2) in 4x4
    {
        Grid2D<Complex> input = make_single_mode_grid(4, 4, 2, 2);

        Grid2D<Complex> expected = make_zero_grid(4, 4);
        expected(2, 2) = Complex{16.0, 0.0};

        if(!run_known_output_2d_case("single_mode_kx2_ky2_4x4", input, expected, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("single_mode_kx2_ky2_4x4");
        }
    }

    // Single Fourier mode (kx=1, ky=1) in 2x4: 
    // DFT2D should produce nx*ny = 8 at bin (1,1), zero elsewhere
    // Non-square grid test
    {
        Grid2D<Complex> input = make_single_mode_grid(2, 4, 1, 1);

        Grid2D<Complex> expected = make_zero_grid(2, 4);
        expected(1, 1) = Complex{8.0, 0.0};

        if(!run_known_output_2d_case("single_mode_kx1_ky1_2x4", input, expected, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("single_mode_kx1_ky1_2x4");
        }
    }


    // --------------------------------------------------------
    // Inverse known-output tests
    // IDFT2D is the dual of the known-output DFT2D tests above.
    // These verify the inverse normalization and sign convention
    // directly, not just through round-trip.
    // --------------------------------------------------------


    // IDFT2D of all-ones 2x2 spectrum = impulse at (0,0)
    // Dual of impulse_00_2x2
    {
        Grid2D<Complex> spectrum = make_constant_grid(2, 2, Complex{1.0, 0.0});

        Grid2D<Complex> expected = make_impulse_grid(2, 2, 0, 0);

        if(!run_inverse_known_output_2d_case("idft2d_flat_spectrum_2x2", spectrum, expected, ITransform_2d::IDFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("idft2d_flat_spectrum_2x2");
        }
    }

    // IDFT2D of DC-only spectrum [16,0,...,0] in 4x4 = constant grid c=1
    // Dual of constant_grid_4x4_c1
    {
        Grid2D<Complex> spectrum = make_zero_grid(4, 4);
        spectrum(0, 0) = Complex{16.0, 0.0};

        Grid2D<Complex> expected = make_constant_grid(4, 4, Complex{1.0, 0.0});

        if(!run_inverse_known_output_2d_case("idft2d_dc_only_4x4", spectrum, expected, ITransform_2d::IDFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("idft2d_dc_only_4x4");
        }
    }

    // IDFT2D of single spectral spike at (1,0) in 4x4 = single Fourier mode (kx=1, ky=0)
    {
        Grid2D<Complex> spectrum = make_zero_grid(4, 4);
        spectrum(1, 0) = Complex{16.0, 0.0};

        Grid2D<Complex> expected = make_single_mode_grid(4, 4, 1, 0);

        if(!run_inverse_known_output_2d_case("idft2d_single_spike_kx1_ky0_4x4", spectrum, expected, ITransform_2d::IDFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("idft2d_single_spike_kx1_ky0_4x4");
        }
    }


    // IDFT2D of single spectral spike at (1,1) in 2x4 = single Fourier mode (kx=1, ky=1)
    // Non-square case: normalization factor is nx*ny = 8, not 16.
    // Directly tests the second axis and rectangular normalization.
    {
        Grid2D<Complex> spectrum = make_zero_grid(2, 4);
        spectrum(1, 1) = Complex{8.0, 0.0};

        Grid2D<Complex> expected = make_single_mode_grid(2, 4, 1, 1);

        if(!run_inverse_known_output_2d_case("idft2d_single_spike_kx1_ky1_2x4", spectrum, expected, ITransform_2d::IDFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("idft2d_single_spike_kx1_ky1_2x4");
        }
    }


    // --------------------------------------------------------
    // Round-trip tests
    // forward -> inverse -> compare against original input
    // --------------------------------------------------------


    // Real-valued round trip 2x2
    {
        Grid2D<Complex> input(2, 2);
        input(0,0) = Complex{1.0, 0.0};
        input(0,1) = Complex{2.0, 0.0};
        input(1,0) = Complex{3.0, 0.0};
        input(1,1) = Complex{4.0, 0.0};

        if(!run_round_trip_2d_case("round_trip_real_2x2", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_2x2");
        }
    }

    // Complex-valued round trip 2x2
    {
        Grid2D<Complex> input(2, 2);
        input(0,0) = Complex{ 1.0,  2.0};
        input(0,1) = Complex{-1.0,  0.5};
        input(1,0) = Complex{ 0.0, -3.0};
        input(1,1) = Complex{ 2.5,  1.0};

        if(!run_round_trip_2d_case("round_trip_complex_2x2", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_2x2");
        }
    }

    // Real-valued round trip 4x4
    {
        Grid2D<Complex> input(4, 4);
        for(std::size_t i = 0; i < 4; ++i) {
            for(std::size_t j = 0; j < 4; ++j) {
                input(i, j) = Complex{static_cast<Real>(i * 4 + j + 1), 0.0};
            }
        }

        if(!run_round_trip_2d_case("round_trip_real_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_4x4");
        }
    }

    // Complex-valued round trip 4x4
    {
        Grid2D<Complex> input(4, 4);
        for(std::size_t i = 0; i < 4; ++i) {
            for(std::size_t j = 0; j < 4; ++j) {
                input(i, j) = Complex{std::cos(static_cast<Real>(i + j)), std::sin(static_cast<Real>(i + j))};
            }
        }

        if(!run_round_trip_2d_case("round_trip_complex_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_4x4");
        }
    }

    // Non-square real round trip 2x4
    {
        Grid2D<Complex> input(2, 4);
        for(std::size_t i = 0; i < 2; ++i) {
            for(std::size_t j = 0; j < 4; ++j) {
                input(i, j) = Complex{static_cast<Real>(i * 4 + j + 1), 0.0};
            }
        }

        if(!run_round_trip_2d_case("round_trip_real_2x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_2x4");
        }
    }

    // Real mixed mode round trip 4x4
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 4);

        if(!run_round_trip_2d_case("round_trip_real_mixed_mode_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_mixed_mode_4x4");
        }
    }

    // Complex mixed mode round trip 4x4
    {
        Grid2D<Complex> input = make_complex_test_grid(4, 4);

        if(!run_round_trip_2d_case("round_trip_complex_mixed_mode_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_mixed_mode_4x4");
        }
    }

    // Larger N round trip 8x8 real
    {
        Grid2D<Complex> input(8, 8);
        for(std::size_t i = 0; i < 8; ++i) {
            for(std::size_t j = 0; j < 8; ++j) {
                input(i, j) = Complex{std::cos(static_cast<Real>(i)), std::sin(static_cast<Real>(j))};
            }
        }

        if(!run_round_trip_2d_case("round_trip_real_8x8", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_8x8");
        }
    }


    // --------------------------------------------------------
    // Linearity tests
    // DFT2D(alpha*x + beta*y) == alpha*DFT2D(x) + beta*DFT2D(y)
    // --------------------------------------------------------


    // Real scalar linearity 2x2
    {
        Grid2D<Complex> x(2, 2);
        x(0,0) = Complex{1.0, 0.0};  x(0,1) = Complex{2.0, 0.0};
        x(1,0) = Complex{3.0, 0.0};  x(1,1) = Complex{4.0, 0.0};

        Grid2D<Complex> y(2, 2);
        y(0,0) = Complex{4.0, 0.0};  y(0,1) = Complex{3.0, 0.0};
        y(1,0) = Complex{2.0, 0.0};  y(1,1) = Complex{1.0, 0.0};

        Complex alpha = {2.0, 0.0};
        Complex beta  = {-1.0, 0.0};

        if(!run_linearity_2d_case("linearity_real_scalars_2x2", x, y, alpha, beta, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_real_scalars_2x2");
        }
    }

    // Complex scalar linearity 4x4
    {
        Grid2D<Complex> x(4, 4);
        Grid2D<Complex> y(4, 4);
        for(std::size_t i = 0; i < 4; ++i) {
            for(std::size_t j = 0; j < 4; ++j) {
                x(i, j) = Complex{static_cast<Real>(i + 1), static_cast<Real>(j)};
                y(i, j) = Complex{static_cast<Real>(4 - i), -0.5 * static_cast<Real>(j + 1)};
            }
        }

        Complex alpha = {1.0,  2.0};
        Complex beta  = {0.5, -1.0};

        if(!run_linearity_2d_case("linearity_complex_scalars_4x4", x, y, alpha, beta, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_complex_scalars_4x4");
        }
    }

    // Linearity with zero scalar: alpha=0 should zero out the first term 4x4
    {
        Grid2D<Complex> x = make_real_mixed_mode_grid(4, 4);
        Grid2D<Complex> y = make_complex_test_grid(4, 4);

        Complex alpha = {0.0, 0.0};
        Complex beta  = {1.0, 0.0};

        if(!run_linearity_2d_case("linearity_zero_alpha_4x4", x, y, alpha, beta, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_zero_alpha_4x4");
        }
    }


    // --------------------------------------------------------
    // Parseval's identity tests
    // sum |u(i,j)|^2 == (1/(nx*ny)) * sum |U(k,l)|^2
    // --------------------------------------------------------


    // Parseval: constant real grid 2x2
    // Physical energy = 4 * 1^2 = 4
    {
        Grid2D<Complex> input = make_constant_grid(2, 2, Complex{1.0, 0.0});

        if(!run_parseval_2d_case("parseval_constant_2x2", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_constant_2x2");
        }
    }

    // Parseval: real-valued entries 4x4
    {
        Grid2D<Complex> input(4, 4);
        for(std::size_t i = 0; i < 4; ++i) {
            for(std::size_t j = 0; j < 4; ++j) {
                input(i, j) = Complex{static_cast<Real>(i * 4 + j + 1), 0.0};
            }
        }

        if(!run_parseval_2d_case("parseval_real_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_real_4x4");
        }
    }

    // Parseval: complex-valued entries 4x4
    {
        Grid2D<Complex> input(4, 4);
        for(std::size_t i = 0; i < 4; ++i) {
            for(std::size_t j = 0; j < 4; ++j) {
                input(i, j) = Complex{std::cos(static_cast<Real>(i + j)), std::sin(static_cast<Real>(i - j))};
            }
        }

        if(!run_parseval_2d_case("parseval_complex_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_complex_4x4");
        }
    }

    // Parseval: single Fourier mode 4x4
    // Physical energy = nx*ny = 16; spectral energy should match
    {
        Grid2D<Complex> input = make_single_mode_grid(4, 4, 1, 2);

        if(!run_parseval_2d_case("parseval_single_mode_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_single_mode_4x4");
        }
    }

    // Parseval: real mixed mode grid 4x4
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 4);

        if(!run_parseval_2d_case("parseval_real_mixed_mode_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_real_mixed_mode_4x4");
        }
    }

    // Parseval: non-square 2x4
    {
        Grid2D<Complex> input(2, 4);
        for(std::size_t i = 0; i < 2; ++i) {
            for(std::size_t j = 0; j < 4; ++j) {
                input(i, j) = Complex{static_cast<Real>(i + 1), static_cast<Real>(j + 1)};
            }
        }

        if(!run_parseval_2d_case("parseval_complex_2x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_complex_2x4");
        }
    }


    // --------------------------------------------------------
    // Circular shift theorem tests
    // DFT2D(x shifted by (sx, sy))[kx,ky] = exp(-2*pi*i*(kx*sx/nx + ky*sy/ny)) * X[kx,ky]
    // --------------------------------------------------------


    // Shift (1,0) on impulse at origin 4x4
    // Simple single-axis shift on a maximally sparse input
    {
        Grid2D<Complex> input = make_impulse_grid(4, 4, 0, 0);

        if(!run_shift_2d_case("shift_10_impulse_4x4", input, 1, 0, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_10_impulse_4x4");
        }
    }

    // Shift (0,1) on impulse at origin 4x4
    // Single-axis shift along the other axis
    {
        Grid2D<Complex> input = make_impulse_grid(4, 4, 0, 0);

        if(!run_shift_2d_case("shift_01_impulse_4x4", input, 0, 1, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_01_impulse_4x4");
        }
    }

    // Shift (1,1) on real mixed mode 4x4
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 4);

        if(!run_shift_2d_case("shift_11_real_mixed_4x4", input, 1, 1, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_11_real_mixed_4x4");
        }
    }

    // Shift (2,1) on complex test grid 4x4
    {
        Grid2D<Complex> input = make_complex_test_grid(4, 4);

        if(!run_shift_2d_case("shift_21_complex_4x4", input, 2, 1, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_21_complex_4x4");
        }
    }

    // Shift by full period (nx,ny) must return original spectrum 4x4
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 4);

        if(!run_shift_2d_case("shift_full_period_4x4", input, 4, 4, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_full_period_4x4");
        }
    }

    // Negative shift (-1, -1) on real mixed mode 4x4
    // Tests that negative shift wrapping is handled correctly
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 4);

        if(!run_shift_2d_case("shift_neg11_real_mixed_4x4", input, -1, -1, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_neg11_real_mixed_4x4");
        }
    }


    // Shift (1,2) on real mixed mode 2x4
    // Non-square grid: shift_x wraps mod 2, shift_y wraps mod 4.
    // nx/ny confusion in the phase factor would give wrong results here.
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(2, 4);

        if(!run_shift_2d_case("shift_12_real_mixed_2x4", input, 1, 2, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_12_real_mixed_2x4");
        }
    }


    // --------------------------------------------------------
    // Conjugate symmetry tests
    // For real-valued input: U(kx,ky) == conj(U((-kx mod nx), (-ky mod ny)))
    // --------------------------------------------------------


    // Conjugate symmetry: constant real 2x2
    {
        Grid2D<Complex> input = make_constant_grid(2, 2, Complex{1.0, 0.0});

        if(!run_conjugate_symmetry_2d_case("conj_sym_constant_2x2", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_constant_2x2");
        }
    }

    // Conjugate symmetry: arbitrary real 4x4
    {
        Grid2D<Complex> input(4, 4);
        for(std::size_t i = 0; i < 4; ++i) {
            for(std::size_t j = 0; j < 4; ++j) {
                input(i, j) = Complex{static_cast<Real>((i + 1) * (j + 2)), 0.0};
            }
        }

        if(!run_conjugate_symmetry_2d_case("conj_sym_arbitrary_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_arbitrary_4x4");
        }
    }

    // Conjugate symmetry: real mixed mode 4x4
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 4);

        if(!run_conjugate_symmetry_2d_case("conj_sym_real_mixed_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_real_mixed_4x4");
        }
    }

    // Conjugate symmetry: single real mode 4x4
    // cos-mode input is real, so spectrum must be conjugate symmetric
    {
        Grid2D<Complex> input(4, 4);
        for(std::size_t i = 0; i < 4; ++i) {
            for(std::size_t j = 0; j < 4; ++j) {
                Real theta = 2.0 * PI * static_cast<Real>(i) / 4.0
                           + 2.0 * PI * static_cast<Real>(j) / 4.0;
                input(i, j) = Complex{std::cos(theta), 0.0};
            }
        }

        if(!run_conjugate_symmetry_2d_case("conj_sym_cosine_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_cosine_4x4");
        }
    }

    // Conjugate symmetry: non-square 2x4 real grid
    {
        Grid2D<Complex> input = make_random_real_grid(2, 4, -1.0, 1.0);

        if(!run_conjugate_symmetry_2d_case("conj_sym_real_2x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_real_2x4");
        }
    }


    // --------------------------------------------------------
    // Modulation theorem tests
    // DFT2D(u * exp(2*pi*i*(kx0*i/nx + ky0*j/ny)))[kx,ky] = U[(kx-kx0) mod nx, (ky-ky0) mod ny]
    // --------------------------------------------------------


    // Modulation: impulse input, shift (1,0), 4x4
    // All-ones spectrum shifted by (1,0) -> all-ones again (impulse invariant)
    {
        Grid2D<Complex> input = make_impulse_grid(4, 4, 0, 0);

        if(!run_modulation_2d_case("modulation_impulse_k10_4x4", input, 1, 0, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_impulse_k10_4x4");
        }
    }

    // Modulation: real mixed mode, shift (1,1), 4x4
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 4);

        if(!run_modulation_2d_case("modulation_real_mixed_k11_4x4", input, 1, 1, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_real_mixed_k11_4x4");
        }
    }

    // Modulation: complex test grid, shift (2,1), 4x4
    // Larger shift exercises full circular wrap in both axes
    {
        Grid2D<Complex> input = make_complex_test_grid(4, 4);

        if(!run_modulation_2d_case("modulation_complex_k21_4x4", input, 2, 1, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_complex_k21_4x4");
        }
    }

    // Modulation: shift by full period (nx,ny) must return original spectrum 4x4
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 4);

        if(!run_modulation_2d_case("modulation_full_period_4x4", input, 4, 4, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_full_period_4x4");
        }
    }

    // Modulation: negative frequency shift (-1,-1), 4x4
    // Tests that negative shift wrapping in frequency space is correct
    {
        Grid2D<Complex> input = make_complex_test_grid(4, 4);

        if(!run_modulation_2d_case("modulation_neg_k11_4x4", input, -1, -1, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_neg_k11_4x4");
        }
    }


    // Modulation: real mixed mode 2x4, shift (1,2)
    // Non-square grid: kx_shift wraps mod 2, ky_shift wraps mod 4.
    // Any nx/ny swap in the modulation exponent produces wrong spectrum bins.
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(2, 4);

        if(!run_modulation_2d_case("modulation_real_mixed_k12_2x4", input, 1, 2, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_real_mixed_k12_2x4");
        }
    }


    // --------------------------------------------------------
    // Separability tests
    // 2D DFT == row-wise 1D DFT followed by column-wise 1D DFT
    // --------------------------------------------------------


    // Separability: impulse at origin 4x4
    {
        Grid2D<Complex> input = make_impulse_grid(4, 4, 0, 0);

        if(!run_separability_2d_case("separability_impulse_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("separability_impulse_4x4");
        }
    }

    // Separability: real mixed mode 4x4
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 4);

        if(!run_separability_2d_case("separability_real_mixed_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("separability_real_mixed_4x4");
        }
    }

    // Separability: complex test grid 4x4
    {
        Grid2D<Complex> input = make_complex_test_grid(4, 4);

        if(!run_separability_2d_case("separability_complex_4x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("separability_complex_4x4");
        }
    }

    // Separability: non-square 2x4
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(2, 4);

        if(!run_separability_2d_case("separability_real_2x4", input, Transform_2d::DFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("separability_real_2x4");
        }
    }


    // --------------------------------------------------------
    // Spectral decay tests
    // Verifies that the heat kernel exp(-alpha*(kx^2+ky^2)*t)
    // applied in spectral space produces the correct energy ratio
    // in physical space after the inverse transform.
    // --------------------------------------------------------


    // Spectral decay: DC-only input, any alpha/t -> no decay (kx=ky=0 mode)
    // exp(-alpha*(0+0)*t) = 1, so energy ratio should be 1
    {
        Grid2D<Complex> input = make_constant_grid(4, 4, Complex{1.0, 0.0});

        if(!run_spectral_decay_2d_case("spectral_decay_dc_only_4x4", input,
            Transform_2d::DFT2D, 1.0, 0.1, 1.0, 1.0, true, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_dc_only_4x4");
        }
    }

    // Spectral decay: real mixed mode 4x4, small t
    // Real input after decay+inverse should remain real-valued
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 4);

        if(!run_spectral_decay_2d_case("spectral_decay_real_mixed_small_t_4x4", input,
            Transform_2d::DFT2D, 0.1, 0.05, 1.0, 1.0, true, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_real_mixed_small_t_4x4");
        }
    }

    // Spectral decay: complex test grid 4x4, small t
    // Complex input is allowed to produce complex output
    {
        Grid2D<Complex> input = make_complex_test_grid(4, 4);

        if(!run_spectral_decay_2d_case("spectral_decay_complex_small_t_4x4", input,
            Transform_2d::DFT2D, 0.1, 0.05, 1.0, 1.0, false, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_complex_small_t_4x4");
        }
    }

    // Spectral decay: t=0 means no decay at all, energy ratio = 1
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 4);

        if(!run_spectral_decay_2d_case("spectral_decay_t0_4x4", input,
            Transform_2d::DFT2D, 1.0, 0.0, 1.0, 1.0, true, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_t0_4x4");
        }
    }

    // Spectral decay: larger domain Lx=Ly=2*pi, real mixed mode 4x4
    // Physical wavenumbers scale with 2*pi/L, so changing L changes decay rate
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 4);

        if(!run_spectral_decay_2d_case("spectral_decay_2pi_domain_4x4", input,
            Transform_2d::DFT2D, 0.1, 0.05, 2.0 * PI, 2.0 * PI, true, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_2pi_domain_4x4");
        }
    }


    // --------------------------------------------------------
    // Summary
    // --------------------------------------------------------


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