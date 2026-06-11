// test_fft2d.cpp
// Responsibility:
//   Test suite for the radix-2 2D FFT/IFFT implementation.
// What to do here:
//   - Verify FFT2D produces correct known outputs.
//   - Verify FFT2D/IFFT2D round-trip consistency.
//   - Verify fundamental Fourier properties hold at all tested sizes.
//   - Verify FFT2D agrees with DFT2D reference on small grids.
//   - Verify FFT2D agrees with FFTW on medium and large grids.
//   - Test power-of-two enforcement.
//   - Smoke-test the spectral heat decay operation.

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>

#include "fft2d.hpp"
#include "dft2d.hpp"
#include "2D_test_utils.hpp"
#include "types.hpp"


int main() {

    std::vector<std::string> failed_tests;

    const Real abs_tol = 1e-10;
    const Real rel_tol = 1e-10;

    std::cout << "=== Running 2D FFT tests ===\n\n";


    // --------------------------------------------------------
    // Known-output tests
    // Same analytic cases as the DFT2D tests but run through the FFT2D.
    // Establish that the FFT computes the correct transform before
    // any comparison against the DFT or FFTW reference.
    // --------------------------------------------------------


    // Zero grid 4x4: FFT2D(0) = 0
    {
        Grid2D<Complex> input = make_zero_grid(4, 4);
        Grid2D<Complex> expected = make_zero_grid(4, 4);

        if(!run_known_output_2d_case("zero_grid_4x4", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("zero_grid_4x4");
        }
    }

    // Zero grid 8x8
    {
        Grid2D<Complex> input = make_zero_grid(8, 8);
        Grid2D<Complex> expected = make_zero_grid(8, 8);

        if(!run_known_output_2d_case("zero_grid_8x8", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("zero_grid_8x8");
        }
    }

    // Constant grid 4x4, c=1: FFT2D at (0,0) = nx*ny = 16, all other bins = 0
    {
        Grid2D<Complex> input = make_constant_grid(4, 4, Complex{1.0, 0.0});
        Grid2D<Complex> expected = make_zero_grid(4, 4);
        expected(0, 0) = Complex{16.0, 0.0};

        if(!run_known_output_2d_case("constant_grid_4x4_c1", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("constant_grid_4x4_c1");
        }
    }

    // Constant grid 8x8, c=1: DC bin = 64
    {
        Grid2D<Complex> input = make_constant_grid(8, 8, Complex{1.0, 0.0});
        Grid2D<Complex> expected = make_zero_grid(8, 8);
        expected(0, 0) = Complex{64.0, 0.0};

        if(!run_known_output_2d_case("constant_grid_8x8_c1", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("constant_grid_8x8_c1");
        }
    }

    // Constant grid 4x8, c=3: DC bin = nx*ny*c = 96
    {
        Grid2D<Complex> input = make_constant_grid(4, 8, Complex{3.0, 0.0});
        Grid2D<Complex> expected = make_zero_grid(4, 8);
        expected(0, 0) = Complex{96.0, 0.0};

        if(!run_known_output_2d_case("constant_grid_4x8_c3", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("constant_grid_4x8_c3");
        }
    }

    // Impulse at (0,0) in 4x4: FFT2D = all-ones grid
    {
        Grid2D<Complex> input = make_impulse_grid(4, 4, 0, 0);
        Grid2D<Complex> expected = make_constant_grid(4, 4, Complex{1.0, 0.0});

        if(!run_known_output_2d_case("impulse_00_4x4", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("impulse_00_4x4");
        }
    }

    // Impulse at (0,0) in 8x8: FFT2D = all-ones grid
    {
        Grid2D<Complex> input = make_impulse_grid(8, 8, 0, 0);
        Grid2D<Complex> expected = make_constant_grid(8, 8, Complex{1.0, 0.0});

        if(!run_known_output_2d_case("impulse_00_8x8", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("impulse_00_8x8");
        }
    }

    // Shifted impulse at (1,2) in 4x4:
    // FFT2D(delta_{1,2})[kx,ky] = exp(-2*pi*i*(kx*1/4 + ky*2/4))
    // Every entry nonzero with magnitude 1, nontrivial phase pattern.
    // Catches sign and axis-ordering bugs that origin impulse cannot.
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

        if(!run_known_output_2d_case("shifted_impulse_12_4x4", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shifted_impulse_12_4x4");
        }
    }

    // Shifted impulse at (3,5) in 8x8:
    // FFT2D(delta_{3,5})[kx,ky] = exp(-2*pi*i*(kx*3/8 + ky*5/8))
    // Larger grid, non-trivial indices on both axes.
    {
        Grid2D<Complex> input = make_impulse_grid(8, 8, 3, 5);
        Grid2D<Complex> expected(8, 8);
        for(std::size_t kx = 0; kx < 8; ++kx) {
            for(std::size_t ky = 0; ky < 8; ++ky) {
                Real theta = -2.0 * PI * (static_cast<Real>(kx) * 3.0 / 8.0
                                        + static_cast<Real>(ky) * 5.0 / 8.0);
                expected(kx, ky) = Complex{std::cos(theta), std::sin(theta)};
            }
        }

        if(!run_known_output_2d_case("shifted_impulse_35_8x8", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shifted_impulse_35_8x8");
        }
    }

    // Single Fourier mode (kx=1, ky=0) in 4x4: spike at (1,0) = 16
    {
        Grid2D<Complex> input = make_single_mode_grid(4, 4, 1, 0);
        Grid2D<Complex> expected = make_zero_grid(4, 4);
        expected(1, 0) = Complex{16.0, 0.0};

        if(!run_known_output_2d_case("single_mode_kx1_ky0_4x4", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("single_mode_kx1_ky0_4x4");
        }
    }

    // Single Fourier mode (kx=1, ky=2) in 4x4: spike at (1,2) = 16
    {
        Grid2D<Complex> input = make_single_mode_grid(4, 4, 1, 2);
        Grid2D<Complex> expected = make_zero_grid(4, 4);
        expected(1, 2) = Complex{16.0, 0.0};

        if(!run_known_output_2d_case("single_mode_kx1_ky2_4x4", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("single_mode_kx1_ky2_4x4");
        }
    }

    // Single Fourier mode (kx=3, ky=5) in 8x8: spike at (3,5) = 64
    {
        Grid2D<Complex> input = make_single_mode_grid(8, 8, 3, 5);
        Grid2D<Complex> expected = make_zero_grid(8, 8);
        expected(3, 5) = Complex{64.0, 0.0};

        if(!run_known_output_2d_case("single_mode_kx3_ky5_8x8", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("single_mode_kx3_ky5_8x8");
        }
    }

    // Single Fourier mode (kx=1, ky=1) in 4x8: spike at (1,1) = 32
    // Non-square: normalization factor = 4*8 = 32
    {
        Grid2D<Complex> input = make_single_mode_grid(4, 8, 1, 1);
        Grid2D<Complex> expected = make_zero_grid(4, 8);
        expected(1, 1) = Complex{32.0, 0.0};

        if(!run_known_output_2d_case("single_mode_kx1_ky1_4x8", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("single_mode_kx1_ky1_4x8");
        }
    }

    // Single Fourier mode (kx=2, ky=5) in 8x16: spike at (2,5) = 128
    // Exercises deeper non-square recursion
    {
        Grid2D<Complex> input = make_single_mode_grid(8, 16, 2, 5);
        Grid2D<Complex> expected = make_zero_grid(8, 16);
        expected(2, 5) = Complex{128.0, 0.0};

        if(!run_known_output_2d_case("single_mode_kx2_ky5_8x16", input, expected, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("single_mode_kx2_ky5_8x16");
        }
    }


    // --------------------------------------------------------
    // Inverse known-output tests
    // Direct correctness check for the inverse transform.
    // Symmetric coverage with the forward known-output tests.
    // --------------------------------------------------------


    // IFFT2D of all-ones 4x4 spectrum = impulse at (0,0)
    {
        Grid2D<Complex> spectrum = make_constant_grid(4, 4, Complex{1.0, 0.0});
        Grid2D<Complex> expected = make_impulse_grid(4, 4, 0, 0);

        if(!run_inverse_known_output_2d_case("ifft2d_flat_spectrum_4x4", spectrum, expected, ITransform_2d::IFFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("ifft2d_flat_spectrum_4x4");
        }
    }

    // IFFT2D of DC-only spectrum [64,...,0] in 8x8 = constant grid c=1
    {
        Grid2D<Complex> spectrum = make_zero_grid(8, 8);
        spectrum(0, 0) = Complex{64.0, 0.0};
        Grid2D<Complex> expected = make_constant_grid(8, 8, Complex{1.0, 0.0});

        if(!run_inverse_known_output_2d_case("ifft2d_dc_only_8x8", spectrum, expected, ITransform_2d::IFFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("ifft2d_dc_only_8x8");
        }
    }

    // IFFT2D of spike at (1,0) in 4x4 = single Fourier mode (kx=1, ky=0)
    {
        Grid2D<Complex> spectrum = make_zero_grid(4, 4);
        spectrum(1, 0) = Complex{16.0, 0.0};
        Grid2D<Complex> expected = make_single_mode_grid(4, 4, 1, 0);

        if(!run_inverse_known_output_2d_case("ifft2d_spike_kx1_ky0_4x4", spectrum, expected, ITransform_2d::IFFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("ifft2d_spike_kx1_ky0_4x4");
        }
    }

    // IFFT2D of spike at (3,5) in 8x8 = single Fourier mode (kx=3, ky=5)
    {
        Grid2D<Complex> spectrum = make_zero_grid(8, 8);
        spectrum(3, 5) = Complex{64.0, 0.0};
        Grid2D<Complex> expected = make_single_mode_grid(8, 8, 3, 5);

        if(!run_inverse_known_output_2d_case("ifft2d_spike_kx3_ky5_8x8", spectrum, expected, ITransform_2d::IFFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("ifft2d_spike_kx3_ky5_8x8");
        }
    }

    // IFFT2D of spike at (1,1) in 4x8: normalization = 32, non-square
    {
        Grid2D<Complex> spectrum = make_zero_grid(4, 8);
        spectrum(1, 1) = Complex{32.0, 0.0};
        Grid2D<Complex> expected = make_single_mode_grid(4, 8, 1, 1);

        if(!run_inverse_known_output_2d_case("ifft2d_spike_kx1_ky1_4x8", spectrum, expected, ITransform_2d::IFFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("ifft2d_spike_kx1_ky1_4x8");
        }
    }


    // --------------------------------------------------------
    // Round-trip tests
    // FFT2D followed by IFFT2D must recover the original input
    // exactly within tolerance across a range of sizes and input types.
    // --------------------------------------------------------


    // Real round trip 4x4
    {
        Grid2D<Complex> input(4, 4);
        for(std::size_t i = 0; i < 4; ++i)
            for(std::size_t j = 0; j < 4; ++j)
                input(i, j) = Complex{static_cast<Real>(i * 4 + j + 1), 0.0};

        if(!run_round_trip_2d_case("round_trip_real_4x4", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_4x4");
        }
    }

    // Complex round trip 4x4
    {
        Grid2D<Complex> input(4, 4);
        for(std::size_t i = 0; i < 4; ++i)
            for(std::size_t j = 0; j < 4; ++j)
                input(i, j) = Complex{std::cos(static_cast<Real>(i + j)),
                                      std::sin(static_cast<Real>(i + j))};

        if(!run_round_trip_2d_case("round_trip_complex_4x4", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_4x4");
        }
    }

    // Real round trip 8x8
    {
        Grid2D<Complex> input(8, 8);
        for(std::size_t i = 0; i < 8; ++i)
            for(std::size_t j = 0; j < 8; ++j)
                input(i, j) = Complex{std::cos(static_cast<Real>(i)),
                                      std::sin(static_cast<Real>(j))};

        if(!run_round_trip_2d_case("round_trip_real_8x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_8x8");
        }
    }

    // Complex round trip 16x16
    {
        Grid2D<Complex> input(16, 16);
        for(std::size_t i = 0; i < 16; ++i)
            for(std::size_t j = 0; j < 16; ++j) {
                Real theta = 2.0 * PI * static_cast<Real>(i * 16 + j) / 256.0;
                input(i, j) = Complex{std::cos(3.0 * theta), std::sin(5.0 * theta)};
            }

        if(!run_round_trip_2d_case("round_trip_complex_16x16", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_16x16");
        }
    }

    // Non-square real round trip 4x8
    {
        Grid2D<Complex> input(4, 8);
        for(std::size_t i = 0; i < 4; ++i)
            for(std::size_t j = 0; j < 8; ++j)
                input(i, j) = Complex{static_cast<Real>(i * 8 + j + 1), 0.0};

        if(!run_round_trip_2d_case("round_trip_real_4x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_4x8");
        }
    }

    // Non-square complex round trip 8x4
    {
        Grid2D<Complex> input(8, 4);
        for(std::size_t i = 0; i < 8; ++i)
            for(std::size_t j = 0; j < 4; ++j)
                input(i, j) = Complex{std::cos(static_cast<Real>(i + j)),
                                      std::sin(static_cast<Real>(i * j + 1))};

        if(!run_round_trip_2d_case("round_trip_complex_8x4", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_8x4");
        }
    }

    // Non-square round trip 8x16
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 16);

        if(!run_round_trip_2d_case("round_trip_complex_8x16", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_8x16");
        }
    }

    // Non-square round trip 16x8
    {
        Grid2D<Complex> input = make_complex_test_grid(16, 8);

        if(!run_round_trip_2d_case("round_trip_complex_16x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_16x8");
        }
    }

    // Medium real round trip 32x32
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(32, 32);

        if(!run_round_trip_2d_case("round_trip_real_32x32", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_32x32");
        }
    }

    // Medium complex round trip 64x64
    {
        Grid2D<Complex> input = make_complex_test_grid(64, 64);

        if(!run_round_trip_2d_case("round_trip_complex_64x64", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_64x64");
        }
    }

    // Large real round trip 128x128
    // Exercises 7 levels of recursion on both axes
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(128, 128);

        if(!run_round_trip_2d_case("round_trip_real_128x128", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_real_128x128");
        }
    }

    // Large complex round trip 256x256
    {
        Grid2D<Complex> input = make_random_complex_grid(256, 256, -1.0, 1.0);

        if(!run_round_trip_2d_case("round_trip_complex_256x256", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("round_trip_complex_256x256");
        }
    }


    // --------------------------------------------------------
    // Linearity tests
    // FFT2D(alpha*x + beta*y) == alpha*FFT2D(x) + beta*FFT2D(y)
    // --------------------------------------------------------


    // Real scalars 4x4
    {
        Grid2D<Complex> x(4, 4), y(4, 4);
        for(std::size_t i = 0; i < 4; ++i)
            for(std::size_t j = 0; j < 4; ++j) {
                x(i, j) = Complex{static_cast<Real>(i + 1), 0.0};
                y(i, j) = Complex{static_cast<Real>(j + 1), 0.0};
            }
        Complex alpha = {2.0, 0.0};
        Complex beta  = {-1.0, 0.0};

        if(!run_linearity_2d_case("linearity_real_scalars_4x4", x, y, alpha, beta, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_real_scalars_4x4");
        }
    }

    // Complex scalars 8x8
    {
        Grid2D<Complex> x(8, 8), y(8, 8);
        for(std::size_t i = 0; i < 8; ++i)
            for(std::size_t j = 0; j < 8; ++j) {
                x(i, j) = Complex{static_cast<Real>(i + 1), static_cast<Real>(j)};
                y(i, j) = Complex{static_cast<Real>(8 - i), -0.5 * static_cast<Real>(j + 1)};
            }
        Complex alpha = {1.0,  2.0};
        Complex beta  = {0.5, -1.0};

        if(!run_linearity_2d_case("linearity_complex_scalars_8x8", x, y, alpha, beta, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_complex_scalars_8x8");
        }
    }

    // Real scalars non-square 4x8
    {
        Grid2D<Complex> x = make_real_mixed_mode_grid(4, 8);
        Grid2D<Complex> y = make_real_mixed_mode_grid(4, 8);
        Complex alpha = {3.0, 0.0};
        Complex beta  = {-2.0, 0.0};

        if(!run_linearity_2d_case("linearity_real_scalars_4x8", x, y, alpha, beta, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_real_scalars_4x8");
        }
    }

    // Complex scalars non-square 8x16
    {
        Grid2D<Complex> x = make_complex_test_grid(8, 16);
        Grid2D<Complex> y = make_complex_test_grid(8, 16);
        Complex alpha = {1.5, -0.5};
        Complex beta  = {-1.0,  2.0};

        if(!run_linearity_2d_case("linearity_complex_scalars_8x16", x, y, alpha, beta, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_complex_scalars_8x16");
        }
    }

    // Zero alpha 32x32: alpha=0 zeros out the first term
    {
        Grid2D<Complex> x = make_real_mixed_mode_grid(32, 32);
        Grid2D<Complex> y = make_complex_test_grid(32, 32);
        Complex alpha = {0.0, 0.0};
        Complex beta  = {1.0, 0.0};

        if(!run_linearity_2d_case("linearity_zero_alpha_32x32", x, y, alpha, beta, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_zero_alpha_32x32");
        }
    }

    // Complex scalars 64x64
    {
        Grid2D<Complex> x = make_complex_test_grid(64, 64);
        Grid2D<Complex> y = make_random_complex_grid(64, 64, -1.0, 1.0);
        Complex alpha = {0.7, -1.3};
        Complex beta  = {2.0,  0.5};

        if(!run_linearity_2d_case("linearity_complex_scalars_64x64", x, y, alpha, beta, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("linearity_complex_scalars_64x64");
        }
    }


    // --------------------------------------------------------
    // Parseval's identity tests
    // sum |u(i,j)|^2 == (1/(nx*ny)) * sum |U(k,l)|^2
    // Energy conservation must hold at all tested sizes and for
    // both square and rectangular grids.
    // --------------------------------------------------------


    // Constant real 4x4
    {
        Grid2D<Complex> input = make_constant_grid(4, 4, Complex{1.0, 0.0});

        if(!run_parseval_2d_case("parseval_constant_4x4", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_constant_4x4");
        }
    }

    // Real entries 8x8
    {
        Grid2D<Complex> input(8, 8);
        for(std::size_t i = 0; i < 8; ++i)
            for(std::size_t j = 0; j < 8; ++j)
                input(i, j) = Complex{static_cast<Real>(i * 8 + j + 1), 0.0};

        if(!run_parseval_2d_case("parseval_real_8x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_real_8x8");
        }
    }

    // Single Fourier mode 8x8
    // Physical energy = nx*ny = 64; spectral energy must match
    {
        Grid2D<Complex> input = make_single_mode_grid(8, 8, 3, 5);

        if(!run_parseval_2d_case("parseval_single_mode_8x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_single_mode_8x8");
        }
    }

    // Complex entries 8x8
    {
        Grid2D<Complex> input(8, 8);
        for(std::size_t i = 0; i < 8; ++i)
            for(std::size_t j = 0; j < 8; ++j) {
                Real theta = 2.0 * PI * static_cast<Real>(i + j) / 16.0;
                input(i, j) = Complex{std::cos(theta), std::sin(2.0 * theta)};
            }

        if(!run_parseval_2d_case("parseval_complex_8x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_complex_8x8");
        }
    }

    // Non-square real 4x8
    {
        Grid2D<Complex> input(4, 8);
        for(std::size_t i = 0; i < 4; ++i)
            for(std::size_t j = 0; j < 8; ++j)
                input(i, j) = Complex{static_cast<Real>(i + 1), static_cast<Real>(j + 1)};

        if(!run_parseval_2d_case("parseval_real_4x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_real_4x8");
        }
    }

    // Non-square complex 8x16
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 16);

        if(!run_parseval_2d_case("parseval_complex_8x16", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_complex_8x16");
        }
    }

    // Non-square complex 16x8
    {
        Grid2D<Complex> input = make_complex_test_grid(16, 8);

        if(!run_parseval_2d_case("parseval_complex_16x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_complex_16x8");
        }
    }

    // Real mixed mode 16x16
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(16, 16);

        if(!run_parseval_2d_case("parseval_real_mixed_16x16", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_real_mixed_16x16");
        }
    }

    // Medium real 32x32
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(32, 32);

        if(!run_parseval_2d_case("parseval_real_32x32", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_real_32x32");
        }
    }

    // Medium complex 64x64
    {
        Grid2D<Complex> input = make_random_complex_grid(64, 64, -1.0, 1.0);

        if(!run_parseval_2d_case("parseval_complex_64x64", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("parseval_complex_64x64");
        }
    }


    // --------------------------------------------------------
    // Circular shift theorem tests
    // FFT2D(x shifted by (sx,sy))[kx,ky] = exp(-2*pi*i*(kx*sx/nx + ky*sy/ny)) * X[kx,ky]
    // Tested with positive, negative, full-period, and
    // larger-than-period shifts on both square and rectangular grids.
    // --------------------------------------------------------


    // Shift (1,0) on impulse at origin 4x4
    {
        Grid2D<Complex> input = make_impulse_grid(4, 4, 0, 0);

        if(!run_shift_2d_case("shift_10_impulse_4x4", input, 1, 0, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_10_impulse_4x4");
        }
    }

    // Shift (0,1) on impulse at origin 8x8
    {
        Grid2D<Complex> input = make_impulse_grid(8, 8, 0, 0);

        if(!run_shift_2d_case("shift_01_impulse_8x8", input, 0, 1, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_01_impulse_8x8");
        }
    }

    // Shift (3,5) on real mixed mode 8x8
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(8, 8);

        if(!run_shift_2d_case("shift_35_real_mixed_8x8", input, 3, 5, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_35_real_mixed_8x8");
        }
    }

    // Shift (5,3) on complex test grid 8x8: axes swapped vs above
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 8);

        if(!run_shift_2d_case("shift_53_complex_8x8", input, 5, 3, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_53_complex_8x8");
        }
    }

    // Full period shift (nx,ny) must return original spectrum 8x8
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(8, 8);

        if(!run_shift_2d_case("shift_full_period_8x8", input, 8, 8, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_full_period_8x8");
        }
    }

    // Larger-than-period shift (nx+2, ny+3) must wrap correctly 8x8
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 8);

        if(!run_shift_2d_case("shift_over_period_8x8", input, 10, 11, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_over_period_8x8");
        }
    }

    // Negative shift (-1,-1) on real mixed mode 8x8
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(8, 8);

        if(!run_shift_2d_case("shift_neg11_real_mixed_8x8", input, -1, -1, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_neg11_real_mixed_8x8");
        }
    }

    // Negative shift (-3,-5) on complex test grid 8x8
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 8);

        if(!run_shift_2d_case("shift_neg35_complex_8x8", input, -3, -5, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_neg35_complex_8x8");
        }
    }

    // Non-square shift (1,2) on real mixed mode 4x8
    // shift_x wraps mod 4, shift_y wraps mod 8
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 8);

        if(!run_shift_2d_case("shift_12_real_mixed_4x8", input, 1, 2, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_12_real_mixed_4x8");
        }
    }

    // Non-square shift (2,1) on complex test grid 8x4
    // Swapped axes vs above
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 4);

        if(!run_shift_2d_case("shift_21_complex_8x4", input, 2, 1, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_21_complex_8x4");
        }
    }

    // Non-square negative shift (-3,-5) on real mixed mode 8x16
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(8, 16);

        if(!run_shift_2d_case("shift_neg35_real_8x16", input, -3, -5, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_neg35_real_8x16");
        }
    }

    // Medium shift on 32x32
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(32, 32);

        if(!run_shift_2d_case("shift_713_real_32x32", input, 7, 13, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("shift_713_real_32x32");
        }
    }


    // --------------------------------------------------------
    // Modulation theorem tests
    // FFT2D(u * exp(2*pi*i*(kx0*i/nx + ky0*j/ny)))[kx,ky] = U[(kx-kx0) mod nx, (ky-ky0) mod ny]
    // Includes positive, negative, full-period, and
    // larger-than-period shifts on square and rectangular grids.
    // --------------------------------------------------------


    // Impulse input, shift (1,0) 4x4
    {
        Grid2D<Complex> input = make_impulse_grid(4, 4, 0, 0);

        if(!run_modulation_2d_case("modulation_impulse_k10_4x4", input, 1, 0, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_impulse_k10_4x4");
        }
    }

    // Real mixed mode, shift (3,5) on 8x8
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(8, 8);

        if(!run_modulation_2d_case("modulation_real_mixed_k35_8x8", input, 3, 5, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_real_mixed_k35_8x8");
        }
    }

    // Complex test grid, shift (5,3) on 8x8: axes swapped vs above
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 8);

        if(!run_modulation_2d_case("modulation_complex_k53_8x8", input, 5, 3, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_complex_k53_8x8");
        }
    }

    // Full period shift (nx,ny) must return original spectrum 8x8
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(8, 8);

        if(!run_modulation_2d_case("modulation_full_period_8x8", input, 8, 8, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_full_period_8x8");
        }
    }

    // Larger-than-period shift (nx+3, ny+5) on 8x8
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 8);

        if(!run_modulation_2d_case("modulation_over_period_8x8", input, 11, 13, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_over_period_8x8");
        }
    }

    // Negative shift (-1,-1) on real mixed mode 8x8
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(8, 8);

        if(!run_modulation_2d_case("modulation_neg_k11_8x8", input, -1, -1, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_neg_k11_8x8");
        }
    }

    // Negative shift (-3,-5) on complex test grid 8x8
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 8);

        if(!run_modulation_2d_case("modulation_neg_k35_8x8", input, -3, -5, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_neg_k35_8x8");
        }
    }

    // Non-square shift (1,2) on real mixed mode 4x8
    // kx_shift wraps mod 4, ky_shift wraps mod 8
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 8);

        if(!run_modulation_2d_case("modulation_real_mixed_k12_4x8", input, 1, 2, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_real_mixed_k12_4x8");
        }
    }

    // Non-square shift (2,1) on complex test grid 8x4: swapped axes
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 4);

        if(!run_modulation_2d_case("modulation_complex_k21_8x4", input, 2, 1, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_complex_k21_8x4");
        }
    }

    // Non-square negative shift (-3,-5) on real mixed mode 8x16
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(8, 16);

        if(!run_modulation_2d_case("modulation_neg_k35_8x16", input, -3, -5, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_neg_k35_8x16");
        }
    }

    // Medium shift on 32x32
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(32, 32);

        if(!run_modulation_2d_case("modulation_k713_real_32x32", input, 7, 13, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("modulation_k713_real_32x32");
        }
    }


    // --------------------------------------------------------
    // Conjugate symmetry tests
    // For real-valued input: U(kx,ky) == conj(U((-kx mod nx), (-ky mod ny)))
    // Real-valued inputs only. Critical precondition for the solver
    // to produce real-valued output after the inverse transform.
    // --------------------------------------------------------


    // Constant real 4x4
    {
        Grid2D<Complex> input = make_constant_grid(4, 4, Complex{1.0, 0.0});

        if(!run_conjugate_symmetry_2d_case("conj_sym_constant_4x4", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_constant_4x4");
        }
    }

    // Arbitrary real 8x8
    {
        Grid2D<Complex> input(8, 8);
        for(std::size_t i = 0; i < 8; ++i)
            for(std::size_t j = 0; j < 8; ++j)
                input(i, j) = Complex{static_cast<Real>((i + 1) * (j + 2)), 0.0};

        if(!run_conjugate_symmetry_2d_case("conj_sym_arbitrary_8x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_arbitrary_8x8");
        }
    }

    // Real mixed mode 8x8
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(8, 8);

        if(!run_conjugate_symmetry_2d_case("conj_sym_real_mixed_8x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_real_mixed_8x8");
        }
    }

    // Cosine-mode real 8x8
    {
        Grid2D<Complex> input(8, 8);
        for(std::size_t i = 0; i < 8; ++i)
            for(std::size_t j = 0; j < 8; ++j) {
                Real theta = 2.0 * PI * static_cast<Real>(i) / 8.0
                           + 2.0 * PI * static_cast<Real>(j) / 8.0;
                input(i, j) = Complex{std::cos(theta), 0.0};
            }

        if(!run_conjugate_symmetry_2d_case("conj_sym_cosine_8x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_cosine_8x8");
        }
    }

    // Non-square real 4x8
    {
        Grid2D<Complex> input = make_random_real_grid(4, 8, -1.0, 1.0);

        if(!run_conjugate_symmetry_2d_case("conj_sym_real_4x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_real_4x8");
        }
    }

    // Non-square real 8x4
    {
        Grid2D<Complex> input = make_random_real_grid(8, 4, -1.0, 1.0);

        if(!run_conjugate_symmetry_2d_case("conj_sym_real_8x4", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_real_8x4");
        }
    }

    // Non-square real 8x16
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(8, 16);

        if(!run_conjugate_symmetry_2d_case("conj_sym_real_8x16", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_real_8x16");
        }
    }

    // Medium random real 32x32
    {
        Grid2D<Complex> input = make_random_real_grid(32, 32, -2.0, 2.0);

        if(!run_conjugate_symmetry_2d_case("conj_sym_random_32x32", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_random_32x32");
        }
    }

    // Medium real mixed mode 64x64
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(64, 64);

        if(!run_conjugate_symmetry_2d_case("conj_sym_real_mixed_64x64", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("conj_sym_real_mixed_64x64");
        }
    }


    // --------------------------------------------------------
    // Separability tests
    // FFT2D == row-wise FFT1D followed by column-wise FFT1D
    // Directly checks the row-column decomposition and is one of
    // the strongest tests for axis-ordering and row/column traversal
    // bugs, complementing the shifted-impulse, shift/modulation,
    // and FFTW rectangular comparison tests.
    // --------------------------------------------------------


    // Impulse at origin 4x4
    {
        Grid2D<Complex> input = make_impulse_grid(4, 4, 0, 0);

        if(!run_separability_2d_case("separability_impulse_4x4", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("separability_impulse_4x4");
        }
    }

    // Real mixed mode 8x8
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(8, 8);

        if(!run_separability_2d_case("separability_real_mixed_8x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("separability_real_mixed_8x8");
        }
    }

    // Complex test grid 8x8
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 8);

        if(!run_separability_2d_case("separability_complex_8x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("separability_complex_8x8");
        }
    }

    // Non-square 4x8
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 8);

        if(!run_separability_2d_case("separability_real_4x8", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("separability_real_4x8");
        }
    }

    // Non-square 8x4
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 4);

        if(!run_separability_2d_case("separability_complex_8x4", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("separability_complex_8x4");
        }
    }

    // Non-square 8x16
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 16);

        if(!run_separability_2d_case("separability_complex_8x16", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("separability_complex_8x16");
        }
    }

    // Medium 32x32
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(32, 32);

        if(!run_separability_2d_case("separability_real_32x32", input, Transform_2d::FFT2D, abs_tol, rel_tol)) {
            failed_tests.push_back("separability_real_32x32");
        }
    }


    // --------------------------------------------------------
    // FFT2D vs DFT2D agreement tests
    // The DFT is the mathematical definition; the FFT must match
    // it exactly. Only run on small grids where the O(N^4) DFT
    // is still tractable.
    // --------------------------------------------------------


    // Zero grid 4x4
    {
        Grid2D<Complex> input = make_zero_grid(4, 4);

        if(!run_fft2d_vs_dft2d_case("fft_vs_dft_zero_4x4", input, abs_tol, rel_tol)) {
            failed_tests.push_back("fft_vs_dft_zero_4x4");
        }
    }

    // Impulse at origin 4x4
    {
        Grid2D<Complex> input = make_impulse_grid(4, 4, 0, 0);

        if(!run_fft2d_vs_dft2d_case("fft_vs_dft_impulse_00_4x4", input, abs_tol, rel_tol)) {
            failed_tests.push_back("fft_vs_dft_impulse_00_4x4");
        }
    }

    // Shifted impulse 4x4
    {
        Grid2D<Complex> input = make_impulse_grid(4, 4, 1, 2);

        if(!run_fft2d_vs_dft2d_case("fft_vs_dft_shifted_impulse_4x4", input, abs_tol, rel_tol)) {
            failed_tests.push_back("fft_vs_dft_shifted_impulse_4x4");
        }
    }

    // Real mixed mode 4x4
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 4);

        if(!run_fft2d_vs_dft2d_case("fft_vs_dft_real_mixed_4x4", input, abs_tol, rel_tol)) {
            failed_tests.push_back("fft_vs_dft_real_mixed_4x4");
        }
    }

    // Complex test grid 4x4
    {
        Grid2D<Complex> input = make_complex_test_grid(4, 4);

        if(!run_fft2d_vs_dft2d_case("fft_vs_dft_complex_4x4", input, abs_tol, rel_tol)) {
            failed_tests.push_back("fft_vs_dft_complex_4x4");
        }
    }

    // Non-square real 4x8
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(4, 8);

        if(!run_fft2d_vs_dft2d_case("fft_vs_dft_real_4x8", input, abs_tol, rel_tol)) {
            failed_tests.push_back("fft_vs_dft_real_4x8");
        }
    }

    // Non-square complex 8x4
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 4);

        if(!run_fft2d_vs_dft2d_case("fft_vs_dft_complex_8x4", input, abs_tol, rel_tol)) {
            failed_tests.push_back("fft_vs_dft_complex_8x4");
        }
    }

    // Real mixed mode 8x8
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(8, 8);

        if(!run_fft2d_vs_dft2d_case("fft_vs_dft_real_mixed_8x8", input, abs_tol, rel_tol)) {
            failed_tests.push_back("fft_vs_dft_real_mixed_8x8");
        }
    }

    // Complex test grid 8x8
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 8);

        if(!run_fft2d_vs_dft2d_case("fft_vs_dft_complex_8x8", input, abs_tol, rel_tol)) {
            failed_tests.push_back("fft_vs_dft_complex_8x8");
        }
    }

    // Non-square 8x16
    {
        Grid2D<Complex> input = make_complex_test_grid(8, 16);

        if(!run_fft2d_vs_dft2d_case("fft_vs_dft_complex_8x16", input, abs_tol, rel_tol)) {
            failed_tests.push_back("fft_vs_dft_complex_8x16");
        }
    }


    // --------------------------------------------------------
    // FFT2D vs FFTW tests
    // FFTW is the industry-standard HPC reference. Agreement here
    // is the strongest absolute correctness guarantee available.
    // Run on medium and large grids where the DFT is intractable.
    //
    // Tolerances are scaled by grid size because floating-point
    // roundoff accumulates as O(log2(N) * eps) per axis. At N=512
    // (9 recursion levels each axis) the absolute error can reach
    // ~1e-9 while the relative L2 error stays near machine epsilon.
    // Using a single 1e-10 absolute tolerance would produce spurious
    // failures at large N that are not implementation bugs.
    // --------------------------------------------------------

    const Real fftw_abs_tol_small  = 1e-10;  // 32x32 .. 128x128
    const Real fftw_abs_tol_medium = 1e-9;   // 256x256 .. 512x512
    const Real fftw_abs_tol_large  = 1e-8;   // 1024x1024
    const Real fftw_rel_tol        = 1e-10;


    // Real mixed mode 32x32
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(32, 32);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_real_32x32", input, fftw_abs_tol_small, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_real_32x32");
        }
    }

    // Complex test grid 32x32
    {
        Grid2D<Complex> input = make_complex_test_grid(32, 32);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_complex_32x32", input, fftw_abs_tol_small, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_complex_32x32");
        }
    }

    // Non-square real 32x64
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(32, 64);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_real_32x64", input, fftw_abs_tol_small, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_real_32x64");
        }
    }

    // Non-square complex 64x32
    {
        Grid2D<Complex> input = make_complex_test_grid(64, 32);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_complex_64x32", input, fftw_abs_tol_small, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_complex_64x32");
        }
    }

    // Real mixed mode 64x64
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(64, 64);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_real_64x64", input, fftw_abs_tol_small, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_real_64x64");
        }
    }

    // Random complex 64x64
    {
        Grid2D<Complex> input = make_random_complex_grid(64, 64, -1.0, 1.0);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_random_complex_64x64", input, fftw_abs_tol_small, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_random_complex_64x64");
        }
    }

    // Real mixed mode 128x128
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(128, 128);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_real_128x128", input, fftw_abs_tol_small, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_real_128x128");
        }
    }

    // Random complex 128x128
    {
        Grid2D<Complex> input = make_random_complex_grid(128, 128, -1.0, 1.0);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_random_complex_128x128", input, fftw_abs_tol_small, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_random_complex_128x128");
        }
    }

    // Non-square 128x256
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(128, 256);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_real_128x256", input, fftw_abs_tol_medium, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_real_128x256");
        }
    }

    // Real mixed mode 256x256
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(256, 256);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_real_256x256", input, fftw_abs_tol_medium, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_real_256x256");
        }
    }

    // Random complex 256x256
    {
        Grid2D<Complex> input = make_random_complex_grid(256, 256, -1.0, 1.0);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_random_complex_256x256", input, fftw_abs_tol_medium, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_random_complex_256x256");
        }
    }

    // Real mixed mode 512x512
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(512, 512);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_real_512x512", input, fftw_abs_tol_medium, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_real_512x512");
        }
    }

    // Random complex 512x512
    {
        Grid2D<Complex> input = make_random_complex_grid(512, 512, -1.0, 1.0);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_random_complex_512x512", input, fftw_abs_tol_medium, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_random_complex_512x512");
        }
    }

    // Non-square 256x512
    {
        Grid2D<Complex> input = make_random_complex_grid(256, 512, -1.0, 1.0);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_complex_256x512", input, fftw_abs_tol_medium, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_complex_256x512");
        }
    }

    // 1024x1024: largest confidence case
    {
        Grid2D<Complex> input = make_random_complex_grid(1024, 1024, -1.0, 1.0);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_complex_1024x1024", input, fftw_abs_tol_large, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_complex_1024x1024");
        }
    }


    // 2048x2048: largest manual-confidence FFTW comparison case
    // This is a stress-level correctness check against FFTW.
    // It is intentionally only one case because memory and runtime grow quickly. 
    // Looser tols
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(2048, 2048);

        if(!run_fft2d_vs_fftw_case("fft_vs_fftw_real_2048x2048", input, 1e-7, fftw_rel_tol)) {
            failed_tests.push_back("fft_vs_fftw_real_2048x2048");
        }
    }


    // --------------------------------------------------------
    // Power-of-two enforcement
    // fft_2d_inplace and ifft_2d_inplace must throw
    // std::invalid_argument for dimensions that are not powers of two.
    // This is a contract test, not a numerical test.
    // --------------------------------------------------------


    // nx=3 (not power of two)
    {
        if(!run_power_of_two_enforcement_2d_case("pow2_enforce_nx3_ny4", 3, 4)) {
            failed_tests.push_back("pow2_enforce_nx3_ny4");
        }
    }

    // ny=3 (not power of two)
    {
        if(!run_power_of_two_enforcement_2d_case("pow2_enforce_nx4_ny3", 4, 3)) {
            failed_tests.push_back("pow2_enforce_nx4_ny3");
        }
    }

    // Both non-power-of-two
    {
        if(!run_power_of_two_enforcement_2d_case("pow2_enforce_nx6_ny6", 6, 6)) {
            failed_tests.push_back("pow2_enforce_nx6_ny6");
        }
    }

    // nx=5, ny=8: one valid, one not
    {
        if(!run_power_of_two_enforcement_2d_case("pow2_enforce_nx5_ny8", 5, 8)) {
            failed_tests.push_back("pow2_enforce_nx5_ny8");
        }
    }

    // nx=8, ny=12: one valid, one not
    {
        if(!run_power_of_two_enforcement_2d_case("pow2_enforce_nx8_ny12", 8, 12)) {
            failed_tests.push_back("pow2_enforce_nx8_ny12");
        }
    }


    // --------------------------------------------------------
    // Invalid-size enforcement
    // Zero dimensions are invalid grid sizes, distinct from
    // non-power-of-two dimensions. Both should throw
    // std::invalid_argument, but the reason is different:
    // a zero dimension is an empty grid, not just an unsupported FFT size.
    // --------------------------------------------------------


    // nx=0: empty in the first dimension
    {
        if(!run_power_of_two_enforcement_2d_case("invalid_size_nx0_ny4", 0, 4)) {
            failed_tests.push_back("invalid_size_nx0_ny4");
        }
    }

    // ny=0: empty in the second dimension
    {
        if(!run_power_of_two_enforcement_2d_case("invalid_size_nx4_ny0", 4, 0)) {
            failed_tests.push_back("invalid_size_nx4_ny0");
        }
    }


    // --------------------------------------------------------
    // Spectral heat decay smoke tests
    // Verifies that the heat kernel exp(-alpha*(kx^2+ky^2)*t)
    // applied in spectral space produces the correct energy ratio
    // in physical space after the inverse transform.
    // For real-valued input, also checks that imaginary leakage
    // after IFFT2D stays within tolerance.
    // --------------------------------------------------------


    // DC-only input 16x16: all energy in (0,0) mode, no decay regardless of alpha/t
    {
        Grid2D<Complex> input = make_constant_grid(16, 16, Complex{1.0, 0.0});

        if(!run_spectral_decay_2d_case("spectral_decay_dc_only_16x16", input,
            Transform_2d::FFT2D, 1.0, 0.1, 1.0, 1.0, true, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_dc_only_16x16");
        }
    }

    // t=0 means no decay, energy ratio = 1, 16x16
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(16, 16);

        if(!run_spectral_decay_2d_case("spectral_decay_t0_16x16", input,
            Transform_2d::FFT2D, 1.0, 0.0, 1.0, 1.0, true, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_t0_16x16");
        }
    }

    // alpha=0: no diffusivity means no decay for any t or input
    // exp(-0*(kx^2+ky^2)*t) = 1 for all modes, so energy ratio = 1
    // Distinct from t=0: here t is nonzero but diffusivity is zero
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(16, 16);

        if(!run_spectral_decay_2d_case("spectral_decay_alpha0_16x16", input,
            Transform_2d::FFT2D, 0.0, 1.0, 1.0, 1.0, true, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_alpha0_16x16");
        }
    }

    // Real mixed mode 16x16, small alpha and t
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(16, 16);

        if(!run_spectral_decay_2d_case("spectral_decay_real_mixed_16x16", input,
            Transform_2d::FFT2D, 0.1, 0.05, 1.0, 1.0, true, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_real_mixed_16x16");
        }
    }

    // Complex test grid 16x16, complex output allowed
    {
        Grid2D<Complex> input = make_complex_test_grid(16, 16);

        if(!run_spectral_decay_2d_case("spectral_decay_complex_16x16", input,
            Transform_2d::FFT2D, 0.1, 0.05, 1.0, 1.0, false, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_complex_16x16");
        }
    }

    // Non-square real 16x32
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(16, 32);

        if(!run_spectral_decay_2d_case("spectral_decay_real_16x32", input,
            Transform_2d::FFT2D, 0.1, 0.05, 1.0, 1.0, true, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_real_16x32");
        }
    }

    // Real mixed mode 32x32
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(32, 32);

        if(!run_spectral_decay_2d_case("spectral_decay_real_32x32", input,
            Transform_2d::FFT2D, 0.1, 0.05, 1.0, 1.0, true, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_real_32x32");
        }
    }

    // Larger domain Lx=Ly=2*pi, 32x32
    // Physical wavenumbers scale with 2*pi/L so changing L changes decay rate
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(32, 32);

        if(!run_spectral_decay_2d_case("spectral_decay_2pi_domain_32x32", input,
            Transform_2d::FFT2D, 0.1, 0.05, 2.0 * PI, 2.0 * PI, true, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_2pi_domain_32x32");
        }
    }

    // Medium real 128x128
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(128, 128);

        if(!run_spectral_decay_2d_case("spectral_decay_real_128x128", input,
            Transform_2d::FFT2D, 0.01, 0.1, 1.0, 1.0, true, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_real_128x128");
        }
    }

    // Large real 512x512 with expect_real_output=true
    // Checks both energy decay ratio and imaginary leakage at scale
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(512, 512);

        if(!run_spectral_decay_2d_case("spectral_decay_real_512x512", input,
            Transform_2d::FFT2D, 0.01, 0.1, 1.0, 1.0, true, abs_tol, rel_tol)) {
            failed_tests.push_back("spectral_decay_real_512x512");
        }
    }


    // Very large real 1024x1024 spectral decay case
    // Stress-level heat-decay smoke test: checks FFT -> heat kernel -> IFFT
    // at large scale, including energy-ratio consistency and imaginary leakage.
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(1024, 1024);

        if(!run_spectral_decay_2d_case("spectral_decay_real_1024x1024", input,
            Transform_2d::FFT2D, 0.01, 0.1, 1.0, 1.0, true, 1e-8, rel_tol)) {
            failed_tests.push_back("spectral_decay_real_1024x1024");
        }
    }


    // Very large real 2048x2048 spectral decay case
    // Stress-level heat-decay smoke test: checks FFT -> heat kernel -> IFFT
    // at large scale, including energy-ratio consistency and imaginary leakage.
    {
        Grid2D<Complex> input = make_real_mixed_mode_grid(2048, 2048);

        if(!run_spectral_decay_2d_case("spectral_decay_real_2048x2048", input,
            Transform_2d::FFT2D, 0.01, 0.1, 1.0, 1.0, true, 1e-8, rel_tol)) {
            failed_tests.push_back("spectral_decay_real_2048x2048");
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