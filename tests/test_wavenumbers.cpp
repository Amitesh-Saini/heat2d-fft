// test_wavenumbers.cpp
// Responsibility:
//   Deterministic tests for Fourier wavenumber construction and squared
//   wavenumber-grid construction.
// What to do here:
//   - Check FFT-frequency ordering for even and odd sizes.
//   - Check physical angular scaling by 2*pi/L.
//   - Check invalid input handling.
//   - Check construction of k^2 = kx^2 + ky^2 on the tensor-product grid.

#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <stdexcept>

#include "wavenumbers.hpp"
#include "types.hpp"
#include "grid2d.hpp"
#include "1D_test_utils.hpp"


bool approx_equal_real_vector(
    const RealVec& expected,
    const RealVec& actual,
    Real abs_tol,
    Real rel_tol
) {
    if(expected.size() != actual.size()) {
        return false;
    }

    for(std::size_t i = 0; i < expected.size(); ++i) {
        if(!approx_equal_real(expected[i], actual[i], abs_tol, rel_tol)) {
            return false;
        }
    }

    return true;
}


void print_real_vector_failure_report(
    const std::string& test_name,
    const RealVec& expected,
    const RealVec& actual,
    Real abs_tol,
    Real rel_tol
) {
    std::cout << "[FAIL] " << test_name << "\n";

    if(expected.size() != actual.size()) {
        std::cout << "  size mismatch: expected " << expected.size()
                  << ", actual " << actual.size() << "\n";
        return;
    }

    for(std::size_t i = 0; i < expected.size(); ++i) {
        if(!approx_equal_real(expected[i], actual[i], abs_tol, rel_tol)) {
            std::cout << "  first mismatch at index " << i << "\n";
            print_scalar_failure_report(test_name, expected[i], actual[i]);
            return;
        }
    }
}


bool approx_equal_real_grid(
    const Grid2D<Real>& expected,
    const Grid2D<Real>& actual,
    Real abs_tol,
    Real rel_tol
) {
    if(expected.nx() != actual.nx() || expected.ny() != actual.ny()) {
        return false;
    }

    for(std::size_t i = 0; i < expected.nx(); ++i) {
        for(std::size_t j = 0; j < expected.ny(); ++j) {
            if(!approx_equal_real(expected(i, j), actual(i, j), abs_tol, rel_tol)) {
                return false;
            }
        }
    }

    return true;
}


void print_real_grid_failure_report(
    const std::string& test_name,
    const Grid2D<Real>& expected,
    const Grid2D<Real>& actual,
    Real abs_tol,
    Real rel_tol
) {
    std::cout << "[FAIL] " << test_name << "\n";

    if(expected.nx() != actual.nx() || expected.ny() != actual.ny()) {
        std::cout << "  shape mismatch: expected "
                  << expected.nx() << " x " << expected.ny()
                  << ", actual "
                  << actual.nx() << " x " << actual.ny() << "\n";
        return;
    }

    for(std::size_t i = 0; i < expected.nx(); ++i) {
        for(std::size_t j = 0; j < expected.ny(); ++j) {
            if(!approx_equal_real(expected(i, j), actual(i, j), abs_tol, rel_tol)) {
                std::cout << "  first mismatch at index (" << i << ", " << j << ")\n";
                print_scalar_failure_report(test_name, expected(i, j), actual(i, j));
                return;
            }
        }
    }
}


template <typename Func>
bool expect_throw(Func f) {
    try {
        f();
    }
    catch(const std::exception&) {
        return true;
    }

    return false;
}


int main() {

    std::vector<std::string> failed_tests;

    const Real abs_tol = 1e-12;
    const Real rel_tol = 1e-12;

    std::cout << "=== Running wavenumber tests ===\n\n";


    // ------------------------------------------------------------
    // 1D Fourier wavenumber tests
    // build_fourier_wavenumbers(n, L) should return angular
    // wavenumbers in FFT output ordering.
    // ------------------------------------------------------------

    // Even n ordering:
    // n = 8, L = 1
    // integer ordering = [0, 1, 2, 3, -4, -3, -2, -1]
    // angular scaling = 2*pi/L = 2*pi
    {
        const std::string test_name = "wavenumbers_even_n8_L1";

        RealVec actual = build_fourier_wavenumbers(8, Real{1});

        RealVec expected = {
            Real{0}  * Real{2} * PI,
            Real{1}  * Real{2} * PI,
            Real{2}  * Real{2} * PI,
            Real{3}  * Real{2} * PI,
            Real{-4} * Real{2} * PI,
            Real{-3} * Real{2} * PI,
            Real{-2} * Real{2} * PI,
            Real{-1} * Real{2} * PI
        };

        if(!approx_equal_real_vector(expected, actual, abs_tol, rel_tol)) {
            print_real_vector_failure_report(test_name, expected, actual, abs_tol, rel_tol);
            failed_tests.push_back(test_name);
        }
    }


    // Odd n ordering:
    // n = 7, L = 1
    // integer ordering = [0, 1, 2, 3, -3, -2, -1]
    // angular scaling = 2*pi/L = 2*pi
    {
        const std::string test_name = "wavenumbers_odd_n7_L1";

        RealVec actual = build_fourier_wavenumbers(7, Real{1});

        RealVec expected = {
            Real{0}  * Real{2} * PI,
            Real{1}  * Real{2} * PI,
            Real{2}  * Real{2} * PI,
            Real{3}  * Real{2} * PI,
            Real{-3} * Real{2} * PI,
            Real{-2} * Real{2} * PI,
            Real{-1} * Real{2} * PI
        };

        if(!approx_equal_real_vector(expected, actual, abs_tol, rel_tol)) {
            print_real_vector_failure_report(test_name, expected, actual, abs_tol, rel_tol);
            failed_tests.push_back(test_name);
        }
    }


    // Domain scaling:
    // n = 4, L = 2
    // multiplier = 2*pi/L = pi
    // integer ordering = [0, 1, -2, -1]
    {
        const std::string test_name = "wavenumbers_scaling_n4_L2";

        RealVec actual = build_fourier_wavenumbers(4, Real{2});

        RealVec expected = {
            Real{0},
            PI,
            Real{-2} * PI,
            Real{-1} * PI
        };

        if(!approx_equal_real_vector(expected, actual, abs_tol, rel_tol)) {
            print_real_vector_failure_report(test_name, expected, actual, abs_tol, rel_tol);
            failed_tests.push_back(test_name);
        }
    }


    // Minimal valid size:
    // n = 2, L = 1
    // integer ordering = [0, -1]
    {
        const std::string test_name = "wavenumbers_minimal_n2_L1";

        RealVec actual = build_fourier_wavenumbers(2, Real{1});

        RealVec expected = {
            Real{0},
            Real{-1} * Real{2} * PI
        };

        if(!approx_equal_real_vector(expected, actual, abs_tol, rel_tol)) {
            print_real_vector_failure_report(test_name, expected, actual, abs_tol, rel_tol);
            failed_tests.push_back(test_name);
        }
    }


    // Invalid n = 0
    {
        const std::string test_name = "wavenumbers_invalid_n0";

        if(!expect_throw([] {
            (void)build_fourier_wavenumbers(0, Real{1});
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    // Invalid n = 1
    {
        const std::string test_name = "wavenumbers_invalid_n1";

        if(!expect_throw([] {
            (void)build_fourier_wavenumbers(1, Real{1});
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    // Invalid L = 0
    {
        const std::string test_name = "wavenumbers_invalid_L0";

        if(!expect_throw([] {
            (void)build_fourier_wavenumbers(8, Real{0});
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    // Invalid L < 0
    {
        const std::string test_name = "wavenumbers_invalid_negative_L";

        if(!expect_throw([] {
            (void)build_fourier_wavenumbers(8, Real{-1});
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    // Invalid L = infinity
    {
        const std::string test_name = "wavenumbers_invalid_infinite_L";

        if(!expect_throw([] {
            (void)build_fourier_wavenumbers(8, std::numeric_limits<Real>::infinity());
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    // Invalid L = NaN
    {
        const std::string test_name = "wavenumbers_invalid_nan_L";

        if(!expect_throw([] {
            (void)build_fourier_wavenumbers(8, std::numeric_limits<Real>::quiet_NaN());
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Squared wavenumber grid tests
    // build_squared_wavenumber_grid(kx, ky) should return:
    //
    //     k2(i,j) = kx[i]^2 + ky[j]^2
    //
    // This is the exact quantity used in the heat-equation decay:
    //
    //     exp(-alpha * k2(i,j) * t)
    // ------------------------------------------------------------

    // Simple hand-checkable k^2 grid
    {
        const std::string test_name = "squared_wavenumber_grid_small";

        RealVec kx = {Real{0}, Real{2}, Real{-2}};
        RealVec ky = {Real{0}, Real{3}};

        Grid2D<Real> actual = build_squared_wavenumber_grid(kx, ky);

        Grid2D<Real> expected(3, 2);
        expected(0, 0) = Real{0};
        expected(0, 1) = Real{9};

        expected(1, 0) = Real{4};
        expected(1, 1) = Real{13};

        expected(2, 0) = Real{4};
        expected(2, 1) = Real{13};

        if(!approx_equal_real_grid(expected, actual, abs_tol, rel_tol)) {
            print_real_grid_failure_report(test_name, expected, actual, abs_tol, rel_tol);
            failed_tests.push_back(test_name);
        }
    }


    // k^2 grid from actual Fourier wavenumbers:
    // n = 4, L = 2
    // k = [0, pi, -2pi, -pi]
    {
        const std::string test_name = "squared_wavenumber_grid_from_fourier_n4_L2";

        RealVec kx = build_fourier_wavenumbers(4, Real{2});
        RealVec ky = build_fourier_wavenumbers(4, Real{2});

        Grid2D<Real> actual = build_squared_wavenumber_grid(kx, ky);

        Grid2D<Real> expected(4, 4);

        expected(0, 0) = Real{0};
        expected(0, 1) = PI * PI;
        expected(0, 2) = Real{4} * PI * PI;
        expected(0, 3) = PI * PI;

        expected(1, 0) = PI * PI;
        expected(1, 1) = Real{2} * PI * PI;
        expected(1, 2) = Real{5} * PI * PI;
        expected(1, 3) = Real{2} * PI * PI;

        expected(2, 0) = Real{4} * PI * PI;
        expected(2, 1) = Real{5} * PI * PI;
        expected(2, 2) = Real{8} * PI * PI;
        expected(2, 3) = Real{5} * PI * PI;

        expected(3, 0) = PI * PI;
        expected(3, 1) = Real{2} * PI * PI;
        expected(3, 2) = Real{5} * PI * PI;
        expected(3, 3) = Real{2} * PI * PI;

        if(!approx_equal_real_grid(expected, actual, abs_tol, rel_tol)) {
            print_real_grid_failure_report(test_name, expected, actual, abs_tol, rel_tol);
            failed_tests.push_back(test_name);
        }
    }


        // Combined solver-style test:
    // Build kx from (nx, Lx), build ky from (ny, Ly), then build k^2 grid.
    //
    // nx = 4, Lx = 2 -> kx = [0, pi, -2pi, -pi]
    // ny = 2, Ly = 1 -> ky = [0, -2pi]
    //
    // k2(i,j) = kx[i]^2 + ky[j]^2
    {
        const std::string test_name = "combined_wavenumber_grid_rectangular_n4_n2";

        RealVec kx = build_fourier_wavenumbers(4, Real{2});
        RealVec ky = build_fourier_wavenumbers(2, Real{1});

        Grid2D<Real> actual = build_squared_wavenumber_grid(kx, ky);

        Grid2D<Real> expected(4, 2);

        expected(0, 0) = Real{0};
        expected(0, 1) = Real{4} * PI * PI;

        expected(1, 0) = PI * PI;
        expected(1, 1) = Real{5} * PI * PI;

        expected(2, 0) = Real{4} * PI * PI;
        expected(2, 1) = Real{8} * PI * PI;

        expected(3, 0) = PI * PI;
        expected(3, 1) = Real{5} * PI * PI;

        if(!approx_equal_real_grid(expected, actual, abs_tol, rel_tol)) {
            print_real_grid_failure_report(test_name, expected, actual, abs_tol, rel_tol);
            failed_tests.push_back(test_name);
        }
    }


    // Combined solver-style test with odd/even mixed sizes:
    // This verifies that build_fourier_wavenumbers and build_squared_wavenumber_grid
    // work together for non-square grids and odd/even dimensions.
    //
    // nx = 3, Lx = 1 -> kx = [0, 2pi, -2pi]
    // ny = 4, Ly = 2 -> ky = [0, pi, -2pi, -pi]
    //
    // k2(i,j) = kx[i]^2 + ky[j]^2
    {
        const std::string test_name = "combined_wavenumber_grid_mixed_n3_n4";

        RealVec kx = build_fourier_wavenumbers(3, Real{1});
        RealVec ky = build_fourier_wavenumbers(4, Real{2});

        Grid2D<Real> actual = build_squared_wavenumber_grid(kx, ky);

        Grid2D<Real> expected(3, 4);

        expected(0, 0) = Real{0};
        expected(0, 1) = PI * PI;
        expected(0, 2) = Real{4} * PI * PI;
        expected(0, 3) = PI * PI;

        expected(1, 0) = Real{4} * PI * PI;
        expected(1, 1) = Real{5} * PI * PI;
        expected(1, 2) = Real{8} * PI * PI;
        expected(1, 3) = Real{5} * PI * PI;

        expected(2, 0) = Real{4} * PI * PI;
        expected(2, 1) = Real{5} * PI * PI;
        expected(2, 2) = Real{8} * PI * PI;
        expected(2, 3) = Real{5} * PI * PI;

        if(!approx_equal_real_grid(expected, actual, abs_tol, rel_tol)) {
            print_real_grid_failure_report(test_name, expected, actual, abs_tol, rel_tol);
            failed_tests.push_back(test_name);
        }
    }


    // Empty kx should throw
    {
        const std::string test_name = "squared_wavenumber_grid_empty_kx";

        if(!expect_throw([] {
            RealVec kx = {};
            RealVec ky = {Real{1}};
            (void)build_squared_wavenumber_grid(kx, ky);
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    // Empty ky should throw
    {
        const std::string test_name = "squared_wavenumber_grid_empty_ky";

        if(!expect_throw([] {
            RealVec kx = {Real{1}};
            RealVec ky = {};
            (void)build_squared_wavenumber_grid(kx, ky);
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    // Both empty should throw
    {
        const std::string test_name = "squared_wavenumber_grid_both_empty";

        if(!expect_throw([] {
            RealVec kx = {};
            RealVec ky = {};
            (void)build_squared_wavenumber_grid(kx, ky);
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    // DC bin must be zero for any valid n and L
    {
        const std::string test_name = "wavenumbers_dc_zero";

        bool passed = true;
        for (std::size_t n : {2, 3, 4, 5, 8, 16}) {
            RealVec kx = build_fourier_wavenumbers(n, Real{1});
            if (!approx_equal_real(kx[0], Real{0}, abs_tol, rel_tol)) {
                passed = false;
                break;
            }
        }

        if (!passed) {
            std::cout << "[FAIL] " << test_name << ": DC bin was not zero\n";
            failed_tests.push_back(test_name);
        }
    }


    // Antisymmetry for even n: kx[n-i] == -kx[i] for i in [1, n/2)
    {
        const std::string test_name = "wavenumbers_antisymmetry_even";

        RealVec kx = build_fourier_wavenumbers(8, Real{1});
        bool passed = true;
        for (std::size_t i = 1; i < 4; ++i) {
            if (!approx_equal_real(kx[8 - i], -kx[i], abs_tol, rel_tol)) {
                passed = false;
                break;
            }
        }

        if (!passed) {
            std::cout << "[FAIL] " << test_name << ": antisymmetry violated\n";
            failed_tests.push_back(test_name);
        }
    }


    // Antisymmetry for odd n: kx[n-i] == -kx[i] for i in [1, (n-1)/2]
    {
        const std::string test_name = "wavenumbers_antisymmetry_odd";

        RealVec kx = build_fourier_wavenumbers(7, Real{1});
        bool passed = true;
        for (std::size_t i = 1; i <= 3; ++i) {
            if (!approx_equal_real(kx[7 - i], -kx[i], abs_tol, rel_tol)) {
                passed = false;
                break;
            }
        }

        if (!passed) {
            std::cout << "[FAIL] " << test_name << ": antisymmetry violated\n";
            failed_tests.push_back(test_name);
        }
    }


    // Uniform spacing between consecutive positive frequencies
    {
        const std::string test_name = "wavenumbers_uniform_spacing";

        Real L = Real{3};
        RealVec kx = build_fourier_wavenumbers(8, L);
        Real expected_spacing = Real{2} * PI / L;
        bool passed = true;
        for (std::size_t i = 1; i < 4; ++i) {
            if (!approx_equal_real(kx[i] - kx[i - 1], expected_spacing, abs_tol, rel_tol)) {
                passed = false;
                break;
            }
        }

        if (!passed) {
            std::cout << "[FAIL] " << test_name << ": spacing between positive frequencies not uniform\n";
            failed_tests.push_back(test_name);
        }
    }


    // Full separability: ksq(i,j) == kx[i]^2 + ky[j]^2 for all i,j
    {
        const std::string test_name = "squared_wavenumber_grid_separability";

        RealVec kx = build_fourier_wavenumbers(8, Real{1});
        RealVec ky = build_fourier_wavenumbers(6, Real{2});
        Grid2D<Real> ksq = build_squared_wavenumber_grid(kx, ky);
        bool passed = true;
        for (std::size_t i = 0; i < kx.size() && passed; ++i) {
            for (std::size_t j = 0; j < ky.size() && passed; ++j) {
                Real expected_val = kx[i] * kx[i] + ky[j] * ky[j];
                if (!approx_equal_real(ksq(i, j), expected_val, abs_tol, rel_tol)) {
                    std::cout << "[FAIL] " << test_name
                              << ": mismatch at (" << i << ", " << j << ")"
                              << " expected " << expected_val
                              << " actual " << ksq(i, j) << "\n";
                    passed = false;
                }
            }
        }

        if (!passed) failed_tests.push_back(test_name);
    }


    // k^2 grid is non-negative everywhere
    {
        const std::string test_name = "squared_wavenumber_grid_nonnegative";

        RealVec kx = build_fourier_wavenumbers(8, Real{1});
        RealVec ky = build_fourier_wavenumbers(8, Real{1});
        Grid2D<Real> ksq = build_squared_wavenumber_grid(kx, ky);
        bool passed = true;
        for (std::size_t i = 0; i < kx.size() && passed; ++i) {
            for (std::size_t j = 0; j < ky.size() && passed; ++j) {
                if (ksq(i, j) < Real{0}) {
                    std::cout << "[FAIL] " << test_name
                              << ": negative value at (" << i << ", " << j << ")\n";
                    passed = false;
                }
            }
        }

        if (!passed) failed_tests.push_back(test_name);
    }


    // Solver-facing spot check: exact decay inputs for low-frequency modes
    // n = 4, L = 2 -> kx = [0, pi, -2pi, -pi]
    // ksq(1,0) = pi^2         (mode (1,0))
    // ksq(0,1) = pi^2         (mode (0,1))
    // ksq(1,1) = 2*pi^2       (mode (1,1))
    {
        const std::string test_name = "squared_wavenumber_grid_solver_decay_inputs";

        RealVec kx = build_fourier_wavenumbers(4, Real{2});
        RealVec ky = build_fourier_wavenumbers(4, Real{2});
        Grid2D<Real> ksq = build_squared_wavenumber_grid(kx, ky);

        Grid2D<Real> expected(4, 4);
        expected(1, 0) = PI * PI;
        expected(0, 1) = PI * PI;
        expected(1, 1) = Real{2} * PI * PI;

        bool passed =
            approx_equal_real(ksq(1, 0), expected(1, 0), abs_tol, rel_tol) &&
            approx_equal_real(ksq(0, 1), expected(0, 1), abs_tol, rel_tol) &&
            approx_equal_real(ksq(1, 1), expected(1, 1), abs_tol, rel_tol);

        if (!passed) {
            std::cout << "[FAIL] " << test_name << "\n";
            print_scalar_failure_report(test_name + " ksq(1,0)", expected(1, 0), ksq(1, 0));
            print_scalar_failure_report(test_name + " ksq(0,1)", expected(0, 1), ksq(0, 1));
            print_scalar_failure_report(test_name + " ksq(1,1)", expected(1, 1), ksq(1, 1));
            failed_tests.push_back(test_name);
        }
    }


    // Nyquist bin magnitude: for even n=8, L=1
    // k_nyq = (n/2) * 2*pi/L = 4 * 2*pi = 8*pi
    // ksq(n/2, 0) and ksq(0, n/2) should both equal (8*pi)^2
    {
        const std::string test_name = "squared_wavenumber_grid_nyquist_magnitude";

        std::size_t n = 8;
        RealVec kx = build_fourier_wavenumbers(n, Real{1});
        RealVec ky = build_fourier_wavenumbers(n, Real{1});
        Grid2D<Real> ksq = build_squared_wavenumber_grid(kx, ky);

        Real k_nyq = Real{4} * Real{2} * PI;
        Real expected_val = k_nyq * k_nyq;

        bool passed =
            approx_equal_real(ksq(n / 2, 0), expected_val, abs_tol, rel_tol) &&
            approx_equal_real(ksq(0, n / 2), expected_val, abs_tol, rel_tol);

        if (!passed) {
            std::cout << "[FAIL] " << test_name << "\n";
            print_scalar_failure_report(test_name + " ksq(n/2,0)", expected_val, ksq(n / 2, 0));
            print_scalar_failure_report(test_name + " ksq(0,n/2)", expected_val, ksq(0, n / 2));
            failed_tests.push_back(test_name);
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

