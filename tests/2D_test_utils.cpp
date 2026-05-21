#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include <stdexcept>
#include <functional>
#include <array>

#include "types.hpp"
#include "dft2d.hpp"
#include "fft2d.hpp"
#include "2D_test_utils.hpp"
#include "1D_test_utils.hpp"


const Real abs_tol = 1e-10;
const Real rel_tol = 1e-10;


// Fixed seed keeps randomized tests reproducible across runs.
static std::mt19937& test_rng()
{
    static std::mt19937 gen(123456789);
    return gen;
}



// Common use Helper


namespace {

enum class Trigfunc {
    Sin,
    Cos
};

struct RealMode {
    std::size_t kx;
    std::size_t ky;
    Real coefficient;
    Trigfunc trig;
};

struct ComplexMode {
    std::size_t kx;
    std::size_t ky;
    Complex amplitude;
};

constexpr std::array<RealMode, 5> REAL_MIXED_MODES = {{
    {1, 0,  0.60, Trigfunc::Sin},
    {0, 1, -0.40, Trigfunc::Cos},
    {1, 2,  0.25, Trigfunc::Sin},
    {2, 1,  0.35, Trigfunc::Cos},
    {3, 1, -0.20, Trigfunc::Sin}
}};

const std::array<ComplexMode, 5> COMPLEX_MIXED_MODES = {{
    {1, 0, Complex{ 0.60,  0.20}},
    {0, 1, Complex{-0.35,  0.40}},
    {1, 2, Complex{ 0.25, -0.30}},
    {2, 1, Complex{ 0.15,  0.50}},
    {3, 1, Complex{-0.20, -0.25}}
}};

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



// Failure Reports

void print_grid_failure_report(
    const std::string& test_name, const Grid2D<Complex>& expected, const Grid2D<Complex>& actual, Real abs_tol) {

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
    std::cout << "Relative L2 error:        " << relative_l2_error_grid(expected, actual, abs_tol) << "\n";
    std::cout << "Relative infinity error:  " << relative_inf_error_grid(expected, actual, abs_tol) << "\n";

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

void print_scalar_failure_report(
    const std::string& test_name, Real expected, Real actual, Real abs_tol, Real rel_tol) 

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



void print_conjugate_symmetry_2d_failure_report(
    const std::string& test_name, std::size_t kx, std::size_t ky, std::size_t mirror_kx, 
    std::size_t mirror_ky, Complex lhs, Complex rhs) {

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


// ------------------------------------------------------------
// Grid comparison and error helpers
// ------------------------------------------------------------

bool approx_equal_grid(const Grid2D<Complex>& a, const Grid2D<Complex>& b, Real abs_tol, Real rel_tol){

    if(!same_shape(a, b)){
        return false;
    }

    const auto& a_data = a.raw();
    const auto& b_data = b.raw();

    for(std::size_t k = 0; k < a_data.size(); ++k){
        
        if(!approx_equal_complex(a_data[k], b_data[k], abs_tol, rel_tol)){
            return false;
        }
    }

    return true;
}


Real max_abs_error_grid(const Grid2D<Complex>& expected, const Grid2D<Complex>& actual){

    if(!same_shape(expected, actual)){
        throw std::invalid_argument("Max_abs_error_grid: Grids Do not match shape");
    }

    Real max_error = 0.0;
    const auto& expected_data = expected.raw();
    const auto& actual_data = actual.raw();

    for(std::size_t k = 0; k < expected_data.size(); ++k){

        Real current_error = std::abs(expected_data[k] - actual_data[k]);

        if(current_error > max_error){

            max_error = current_error;
        }
    }

    return max_error;
}


Real relative_l2_error_grid(const Grid2D<Complex>& expected, const Grid2D<Complex>& actual, Real abs_tol){

    if(!same_shape(expected, actual)){
        throw std::invalid_argument("relative_l2_error_grid: Grids Do not match shape");
    }

    Real numerator = 0.0;
    Real denominator = 0.0;
    const auto& expected_data = expected.raw();
    const auto& actual_data = actual.raw();

    for(std::size_t k = 0; k < expected_data.size(); ++k){

        numerator += std::norm(expected_data[k] - actual_data[k]);
        denominator += std::norm(expected_data[k]);
    }

    if(denominator <= abs_tol * abs_tol) return std::sqrt(numerator); // return absolute l2 error

    return std::sqrt(numerator/denominator);
}


Real relative_inf_error_grid(const Grid2D<Complex>& expected,const Grid2D<Complex>& actual, Real abs_tol){

    if(!same_shape(expected, actual)){
        throw std::invalid_argument("relative_inf_error_grid: Grids Do not match shape");
    }

    Real max_error = 0.0;
    Real max_ref = 0.0;
    const auto& expected_data = expected.raw();
    const auto& actual_data = actual.raw();

    for(std::size_t k = 0; k < expected_data.size(); ++k){
        
        Real error_k = std::abs(expected_data[k] - actual_data[k]);
        Real ref_k = std::abs(expected_data[k]); 

        if(error_k > max_error) max_error = error_k;
        if(ref_k > max_ref) max_ref = ref_k;
    }

    if(max_ref <= abs_tol) return max_error; // Absolute inf norm

    return max_error/max_ref;
}


// ------------------------------------------------------------
// Grid construction helpers
// ------------------------------------------------------------


Grid2D<Complex> make_zero_grid(std::size_t nx, std::size_t ny){

    Grid2D<Complex> zero_grid(nx, ny);

    return zero_grid;
}

Grid2D<Complex> make_constant_grid(std::size_t nx, std::size_t ny, Complex c){

    Grid2D<Complex> constant_grid(nx, ny, c);

    return constant_grid;
}

Grid2D<Complex> make_impulse_grid(std::size_t nx, std::size_t ny, std::size_t i0, std::size_t j0){

    if (nx == 0 || ny == 0) throw std::invalid_argument("make_impulse_grid: grid dimensions must be positive");

    if(i0 >= nx || j0 >= ny) throw std::out_of_range("make_impulse_grid: Invalid indices");

     Grid2D<Complex> impulse_grid(nx, ny);

     impulse_grid(i0, j0) = Complex{1,0};

     return impulse_grid;
}

Grid2D<Complex> make_single_mode_grid(std::size_t nx, std::size_t ny, std::size_t kx, std::size_t ky){

    if (nx == 0 || ny == 0) throw std::invalid_argument("make_single_mode_grid: grid dimensions must be positive");
    
    if(kx >= nx || ky >= ny) throw std::out_of_range("make_single_mode_grid: Invalid indices");

    Grid2D<Complex> single_mode_grid(nx, ny);

    Real omega = 2.0 * PI * static_cast<Real>(kx) / static_cast<Real>(nx);
    Complex c_omega = {std::cos(omega), std::sin(omega)};
    Complex c_omega_phase = {1.0, 0.0};

    Real alpha = 2.0 * PI * static_cast<Real>(ky) / static_cast<Real>(ny);
    Complex c_alpha = {std::cos(alpha), std::sin(alpha)};

    for(std::size_t i = 0; i < nx; ++i){

        Complex c_alpha_phase = {1.0, 0.0};

        for(std::size_t j = 0; j < ny; ++j){

            single_mode_grid(i,j) = c_omega_phase * c_alpha_phase;

            c_alpha_phase *= c_alpha;
        }

        c_omega_phase *= c_omega;
    }

    return single_mode_grid;
}


Grid2D<Complex> make_real_mixed_mode_grid(std::size_t nx, std::size_t ny){

    if (nx == 0 || ny == 0) throw std::invalid_argument("make_real_mixed_mode_grid: grid dimensions must be positive");

    Grid2D<Complex> real_mixed_mode_grid(nx, ny);

    Real phase_x = 2.0 * PI / static_cast<Real>(nx);
    Real phase_y = 2.0 * PI / static_cast<Real>(ny);

    for(std::size_t i = 0; i < nx; ++i){

        for(std::size_t j = 0; j < ny; ++j){

            Real mode = 0.0;

            for(std::size_t k = 0; k < REAL_MIXED_MODES.size(); ++k){

                Real omega = (phase_x * static_cast<Real>(i) * static_cast<Real>(REAL_MIXED_MODES[k].kx)) 
                + (phase_y * static_cast<Real>(j) * static_cast<Real>(REAL_MIXED_MODES[k].ky));

                if(REAL_MIXED_MODES[k].trig == Trigfunc::Cos) mode += REAL_MIXED_MODES[k].coefficient * std::cos(omega);

                else if(REAL_MIXED_MODES[k].trig == Trigfunc::Sin) mode += REAL_MIXED_MODES[k].coefficient * std::sin(omega);
            }

            real_mixed_mode_grid(i,j) = mode;
        }
    }

    return real_mixed_mode_grid;
}


Grid2D<Complex> make_random_real_mixed_mode_grid(std::size_t nx, std::size_t ny, Real lower_bound, Real upper_bound){

    if (nx == 0 || ny == 0) throw std::invalid_argument("make_random_real_mixed_mode_grid: grid dimensions must be positive");

    if(lower_bound >= upper_bound){

        throw std::invalid_argument("make_random_real_mixed_mode_grid: Lower bound must be smaller than upper");
    }

    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
    auto& gen = test_rng();

    std::array<Real, REAL_MIXED_MODES.size()> coeffs;

    for(std::size_t k = 0; k < coeffs.size(); ++k) coeffs[k] = dist(gen);

    Grid2D<Complex> random_real_mixed_mode_grid(nx, ny);

    Real phase_x = 2.0 * PI / static_cast<Real>(nx);
    Real phase_y = 2.0 * PI / static_cast<Real>(ny);

    for(std::size_t i = 0; i < nx; ++i){

        for(std::size_t j = 0; j < ny; ++j){

            Real mode = 0.0;

            for(std::size_t k = 0; k < REAL_MIXED_MODES.size(); ++k){

                Real omega = (phase_x * static_cast<Real>(i) * static_cast<Real>(REAL_MIXED_MODES[k].kx)) 
                + (phase_y * static_cast<Real>(j) * static_cast<Real>(REAL_MIXED_MODES[k].ky));

                if(REAL_MIXED_MODES[k].trig == Trigfunc::Cos) mode += coeffs[k] * std::cos(omega);
                else                                          mode += coeffs[k] * std::sin(omega);

            }

            random_real_mixed_mode_grid(i,j) = mode;
        }
    }

    return random_real_mixed_mode_grid;
}



