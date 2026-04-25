#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include <stdexcept>
#include <functional>

#include "types.hpp"
#include "dft2d.hpp"
#include "fft2d.hpp"
#include "2D_test_utils.hpp"


namespace {

bool same_shape(const Grid2D<Complex>& a, const Grid2D<Complex>& b) {
    return a.nx() == b.nx() && a.ny() == b.ny();
}


void print_grid(const Grid2D<Complex>& grid) {
    const std::size_t nx = grid.nx();
    const std::size_t ny = grid.ny();

    for (std::size_t i = 0; i < nx; ++i) {
        std::cout << "  [ ";
        for (std::size_t j = 0; j < ny; ++j) {
            std::cout << grid(i, j);
            if (j + 1 < ny) {
                std::cout << ", ";
            }
        }
        std::cout << " ]\n";
    }
}

} 



void print_grid_failure_report(const std::string& test_name, const Grid2D<Complex>& expected, const Grid2D<Complex>& actual) {

    std::cout << std::setprecision(16);

    std::cout << "FAIL: " << test_name << "\n";

    std::cout << "Expected shape: " << expected.nx() << " x " << expected.ny() << "\n";

    std::cout << "Actual shape:   " << actual.nx() << " x " << actual.ny() << "\n";

    if (!same_shape(expected, actual)) {
        std::cout << "Shape mismatch: cannot compute entrywise error metrics.\n\n";
        return;
    }

    const std::size_t nx = expected.nx();
    const std::size_t ny = expected.ny();

    Real max_error = 0.0;
    std::size_t max_i = 0;
    std::size_t max_j = 0;

    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            Real error = std::abs(expected(i, j) - actual(i, j));

            if (error > max_error) {
                max_error = error;
                max_i = i;
                max_j = j;
            }
        }
    }

    std::cout << "Max absolute error:       " << max_error << "\n";
    std::cout << "Relative L2 error:        " << relative_l2_error_grid(expected, actual) << "\n";
    std::cout << "Relative infinity error:  " << relative_inf_error_grid(expected, actual) << "\n";

    std::cout << "Worst entry index:        (" << max_i << ", " << max_j << ")\n";

    std::cout << "Expected at worst entry:  " << expected(max_i, max_j) << "\n";

    std::cout << "Actual at worst entry:    " << actual(max_i, max_j) << "\n";

    std::cout << "Entrywise error there:    " << std::abs(expected(max_i, max_j) - actual(max_i, max_j)) << "\n";

    const std::size_t total_entries = nx * ny;

    if (total_entries <= 64) {
        std::cout << "\nExpected grid:\n";
        print_grid(expected);

        std::cout << "\nActual grid:\n";
        print_grid(actual);
    } else {
        std::cout << "\nGrid too large to print fully; showing summary only.\n";
    }

    std::cout << "\n";
}

void print_scalar_failure_report(const std::string& test_name, Real expected, Real actual, Real abs_tol, Real rel_tol) 

{
    std::cout << std::setprecision(16);

    const Real abs_error = std::abs(expected - actual);

    Real rel_error = abs_error;
    if (std::abs(expected) > 0.0) {
        rel_error = abs_error / std::abs(expected);
    }

    std::cout << "FAIL: " << test_name << "\n";
    std::cout << "Expected:       " << expected << "\n";
    std::cout << "Actual:         " << actual << "\n";
    std::cout << "Absolute error: " << abs_error << "\n";
    std::cout << "Relative error: " << rel_error << "\n";
    std::cout << "Abs tolerance:  " << abs_tol << "\n";
    std::cout << "Rel tolerance:  " << rel_tol << "\n\n";
}



void print_conjugate_symmetry_2d_failure_report(const std::string& test_name, std::size_t kx, std::size_t ky, std::size_t mirror_kx, std::size_t mirror_ky,
 Complex lhs, Complex rhs) {

    std::cout << std::setprecision(16);

    std::cout << "FAIL: " << test_name << "\n";

    std::cout << "Frequency index:          ("
              << kx << ", " << ky << ")\n";

    std::cout << "Mirror frequency index:   ("
              << mirror_kx << ", " << mirror_ky << ")\n";

    std::cout << "Spectrum value:           " << lhs << "\n";
    std::cout << "Conjugate mirror value:   " << rhs << "\n";
    std::cout << "Absolute error:           " << std::abs(lhs - rhs) << "\n\n";
}



