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
#include <fftw3.h>
#include <limits>

#include "types.hpp"
#include "dft1d.hpp"
#include "fft1d.hpp"
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




void print_spectral_decay_failure_report(
    const std::string& test_name, Real theoretical_ratio, Real actual_ratio, Real initial_physical_energy, Real final_physical_energy, 
    Real initial_spectral_energy, Real final_spectral_energy, Real max_imag, bool expect_real_output, Real abs_tol, Real rel_tol){

    std::cout << std::setprecision(16);

    std::cout << "FAIL: " << test_name << "\n";

    const Real ratio_abs_diff = std::abs(theoretical_ratio - actual_ratio);
    const Real ratio_tol =
        abs_tol + rel_tol * std::max(std::abs(theoretical_ratio), std::abs(actual_ratio));

    std::cout << "Theoretical energy ratio: " << theoretical_ratio << "\n";
    std::cout << "Actual energy ratio:      " << actual_ratio << "\n";
    std::cout << "Ratio absolute diff:      " << ratio_abs_diff << "\n";
    std::cout << "Allowed ratio tolerance:  " << ratio_tol << "\n";

    std::cout << "\nPhysical energy:\n";
    std::cout << "Initial physical energy:  " << initial_physical_energy << "\n";
    std::cout << "Final physical energy:    " << final_physical_energy << "\n";

    std::cout << "\nSpectral energy:\n";
    std::cout << "Initial spectral energy:  " << initial_spectral_energy << "\n";
    std::cout << "Final spectral energy:    " << final_spectral_energy << "\n";

    const Real initial_energy_diff =
        std::abs(initial_physical_energy - initial_spectral_energy);

    const Real final_energy_diff =
        std::abs(final_physical_energy - final_spectral_energy);

    const Real initial_energy_tol =
        abs_tol + rel_tol * std::max(std::abs(initial_physical_energy),
                                     std::abs(initial_spectral_energy));

    const Real final_energy_tol =
        abs_tol + rel_tol * std::max(std::abs(final_physical_energy),
                                     std::abs(final_spectral_energy));

    std::cout << "\nPhysical/spectral consistency:\n";
    std::cout << "Initial energy diff:      " << initial_energy_diff << "\n";
    std::cout << "Initial energy tolerance: " << initial_energy_tol << "\n";
    std::cout << "Final energy diff:        " << final_energy_diff << "\n";
    std::cout << "Final energy tolerance:   " << final_energy_tol << "\n";

    std::cout << "\nImaginary leakage after inverse transform:\n";
    std::cout << "Expected real output:     "
              << (expect_real_output ? "yes" : "no") << "\n";
    std::cout << "Max imaginary part:       " << max_imag << "\n";

    if (expect_real_output) {
        std::cout << "Imaginary tolerance:      " << abs_tol << "\n";
        std::cout << "Imaginary check status:   "
                  << (max_imag > abs_tol ? "failed" : "passed") << "\n";
    } else {
        std::cout << "Imaginary check status:   skipped; complex output is allowed\n";
    }

    std::cout << "\n";
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


Grid2D<Complex> make_random_real_grid(std::size_t nx, std::size_t ny, Real lower_bound, Real upper_bound){

    if (nx == 0 || ny == 0) throw std::invalid_argument("make_random_real_grid: grid dimensions must be positive");

    if(lower_bound >= upper_bound){

        throw std::invalid_argument("make_random_real_grid: Lower bound must be smaller than upper");
    }

    Grid2D<Complex> random_real_grid(nx, ny);

    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
    auto& gen = test_rng();

    for(std::size_t i = 0; i < nx; ++i){
        for(std::size_t j = 0; j < ny; ++j){

            random_real_grid(i,j) = {dist(gen), 0.0};
        }
    }

    return random_real_grid;
}


Grid2D<Complex> make_complex_test_grid(std::size_t nx, std::size_t ny){

    if (nx == 0 || ny == 0) throw std::invalid_argument("make_complex_test_grid: grid dimensions must be positive");

    Grid2D<Complex> complex_mixed_mode_grid(nx, ny);

    Real omega = 2.0 * PI / static_cast<Real>(nx);
    Real alpha = 2.0 * PI / static_cast<Real>(ny);

    for(std::size_t k = 0; k < COMPLEX_MIXED_MODES.size(); ++k){

        Complex gamma = {std::cos(omega * static_cast<Real>(COMPLEX_MIXED_MODES[k].kx)), std::sin(omega * static_cast<Real>(COMPLEX_MIXED_MODES[k].kx))};
        Complex gamma_phase = {1,0};

        Complex beta = {std::cos(alpha * static_cast<Real>(COMPLEX_MIXED_MODES[k].ky)), std::sin(alpha * static_cast<Real>(COMPLEX_MIXED_MODES[k].ky))};

        for(std::size_t i = 0; i < nx; ++i){

            Complex beta_phase = {1,0};

            for(std::size_t j = 0; j < ny; ++j){

                complex_mixed_mode_grid(i,j) += COMPLEX_MIXED_MODES[k].amplitude * gamma_phase * beta_phase;
                beta_phase *= beta;
            }

            gamma_phase *= gamma;
        }
    }

    return complex_mixed_mode_grid;
}


Grid2D<Complex> make_random_complex_grid(std::size_t nx, std::size_t ny, Real lower_bound, Real upper_bound){

    if (nx == 0 || ny == 0) throw std::invalid_argument("make_random_complex_grid: grid dimensions must be positive");

    if(lower_bound >= upper_bound){

        throw std::invalid_argument("make_random_complex_grid: Lower bound must be smaller than upper");
    }

    Grid2D<Complex> random_complex_grid(nx, ny);

    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
    auto& gen = test_rng();

    for(std::size_t i = 0; i < nx; ++i){
        for(std::size_t j = 0; j < ny; ++j){

            random_complex_grid(i,j) = {dist(gen), dist(gen)};
        }
    }

    return random_complex_grid;

}



// Energy Calculators


Real physical_energy_2d(const Grid2D<Complex>& field){

    Real sum = 0.0;

    const auto& raw_field = field.raw();

    for(std::size_t i = 0; i < raw_field.size(); ++i){

        sum += std::norm(raw_field[i]);
    }

    return sum;
}


Real spectral_energy_2d(const Grid2D<Complex>& spectrum){

    if(spectrum.nx() == 0 || spectrum.ny() == 0) throw std::invalid_argument("spectral_energy_2d: grid dimensions must be positive");

    Real sum = 0.0;

    const auto& raw_spectrum = spectrum.raw();

    for(std::size_t i = 0; i < raw_spectrum.size(); ++i){

        sum += std::norm(raw_spectrum[i]);
    }

    return sum / static_cast<Real>(spectrum.nx() * spectrum.ny());
}



// Fourier Property Helpers


Grid2D<Complex> circular_shift_2d(const Grid2D<Complex>& field, std::ptrdiff_t shift_x, std::ptrdiff_t shift_y){

    if(field.nx() == 0 || field.ny() == 0) throw std::invalid_argument("circular_shift_2d: grid dimensions must be positive");

    std::ptrdiff_t nx = static_cast<std::ptrdiff_t>(field.nx());
    std::ptrdiff_t ny = static_cast<std::ptrdiff_t>(field.ny());

    Grid2D<Complex> circular_shifted_field(field.nx(), field.ny());


    for(std::size_t i = 0; i < field.nx(); ++i){

        std::ptrdiff_t wrapped_x = (((static_cast<std::ptrdiff_t>(i) - shift_x) % nx) + nx) % nx;

        for(std::size_t j = 0; j < field.ny(); ++j){

            std::ptrdiff_t wrapped_y = (((static_cast<std::ptrdiff_t>(j) - shift_y) % ny) + ny) % ny;

            circular_shifted_field(i,j) = field(static_cast<std::size_t>(wrapped_x), 
                                                static_cast<std::size_t>(wrapped_y));
        }
    }

    return circular_shifted_field;
}

Grid2D<Complex> modulate_2d(const Grid2D<Complex>& field, std::ptrdiff_t kx_shift, std::ptrdiff_t ky_shift){

    if (field.nx() == 0 || field.ny() == 0) throw std::invalid_argument("modulate_2d: grid dimensions must be positive");

    Real nx = static_cast<Real>(field.nx());
    Real ny = static_cast<Real>(field.ny());

    Grid2D<Complex> modulated_field(field.nx(), field.ny());

    std::ptrdiff_t reduced_shift_kx = kx_shift % static_cast<std::ptrdiff_t>(field.nx());
    std::ptrdiff_t reduced_shift_ky = ky_shift % static_cast<std::ptrdiff_t>(field.ny());

    Real omega = 2.0 * PI * static_cast<Real>(reduced_shift_kx) / nx;
    Complex c_omega = {std::cos(omega), std::sin(omega)};
    Complex c_omega_phase = {1.0, 0.0};

    Real alpha = 2.0 * PI * static_cast<Real>(reduced_shift_ky) / ny;
    Complex c_alpha = {std::cos(alpha), std::sin(alpha)};

    for(std::size_t i = 0; i < field.nx(); ++i){

        Complex c_alpha_phase = {1.0, 0.0};

        for(std::size_t j = 0; j < field.ny(); ++j){

            modulated_field(i, j) = field(i,j) * c_omega_phase * c_alpha_phase;

            c_alpha_phase *= c_alpha;
        }

        c_omega_phase *= c_omega;
    }

    return modulated_field;
}


Real max_imag_part(const Grid2D<Complex>& field){

    if (field.nx() == 0 || field.ny() == 0) throw std::invalid_argument("max_imag_part: grid dimensions must be positive");

    Real max_imag = 0.0;
    
    const auto& raw_field = field.raw();

    for(std::size_t k = 0; k < raw_field.size(); ++k){
        
        if(std::abs(raw_field[k].imag()) > max_imag) max_imag = std::abs(raw_field[k].imag());
    }

    return max_imag;
}


bool is_real_grid_within_tol(const Grid2D<Complex>& field, Real abs_tol){

    if (field.nx() == 0 || field.ny() == 0) throw std::invalid_argument("max_imag_part: grid dimensions must be positive");

    const auto& raw_field = field.raw();

    for(std::size_t k = 0; k < raw_field.size(); ++k){

        if(std::abs(raw_field[k].imag()) > abs_tol) return false;
    }

    return true;
}



// Run tests


bool run_known_output_2d_case(
    const std::string& test_name, const Grid2D<Complex>& input, const Grid2D<Complex>& expected, 
    Transform_2d transform, Real abs_tol, Real rel_tol){

    Grid2D<Complex> spectrum(input.nx(), input.ny());

    if(transform == Transform_2d::DFT2D){

        spectrum = dft_2d(input);
    }

    else if(transform == Transform_2d::FFT2D){

        spectrum = input;

        fft_2d_inplace(spectrum);
    }

    else throw std::invalid_argument("run_known_output_2d_case: unsupported transform");

    if(!approx_equal_grid(spectrum, expected, abs_tol, rel_tol)){
        print_grid_failure_report(test_name, expected, spectrum, abs_tol);
        return false;
    }

    return true;
}


bool run_inverse_known_output_2d_case(
    const std::string& test_name, const Grid2D<Complex>& spectrum, const Grid2D<Complex>& expected, 
    ITransform_2d inverse_transform, Real abs_tol, Real rel_tol){


    Grid2D<Complex> actual(spectrum.nx(), spectrum.ny());

    if(inverse_transform == ITransform_2d::IDFT2D){

        actual = idft_2d(spectrum);
    }

    else if(inverse_transform == ITransform_2d::IFFT2D){

        actual = spectrum;

        ifft_2d_inplace(actual);
    }

    else throw std::invalid_argument("run_inverse_known_output_2d_case: unsupported transform");

    if(!approx_equal_grid(actual, expected, abs_tol, rel_tol)){
        print_grid_failure_report(test_name, expected, actual, abs_tol);
        return false;
    }

    return true;
}


bool run_round_trip_2d_case(
    const std::string& test_name, const Grid2D<Complex>& input, Transform_2d transform, Real abs_tol, Real rel_tol){


    Grid2D<Complex> spectrum;
    Grid2D<Complex> reconstructed_input;

    if(transform == Transform_2d::DFT2D){

        spectrum = dft_2d(input);
        reconstructed_input = idft_2d(spectrum);
    }

    else if(transform == Transform_2d::FFT2D){

        spectrum = input;
        fft_2d_inplace(spectrum);

        reconstructed_input = spectrum;
        ifft_2d_inplace(reconstructed_input);
    }

    else throw std::invalid_argument("run_round_trip_2d_case: unsupported transform");

    if(!approx_equal_grid(input, reconstructed_input, abs_tol, rel_tol)){

        print_grid_failure_report(test_name, input, reconstructed_input, abs_tol);
        return false;
    }

    return true;
}


bool run_linearity_2d_case(
    const std::string& test_name, const Grid2D<Complex>& x, const Grid2D<Complex>& y, Complex alpha, 
    Complex beta, Transform_2d transform, Real abs_tol, Real rel_tol){

    if(!same_shape(x, y)) throw std::invalid_argument("run_linearity_2d_case: x and y not same shape");
    
    if(x.size() != y.size()) throw std::invalid_argument("run_linearity_2d_case: x.size != y.size");

    Grid2D<Complex> combined_xy(x.nx(), x.ny());

    for(std::size_t i = 0; i < x.nx(); ++i){
        for(std::size_t j = 0; j < x.ny(); ++j){

            combined_xy(i,j) = (alpha * x(i,j)) + (beta * y(i,j));
        }
    }

    Grid2D<Complex> lhs;
    Grid2D<Complex> rhs(x.nx(), x.ny());
    Grid2D<Complex> dx;
    Grid2D<Complex> dy;

    if(transform == Transform_2d::DFT2D){

        lhs = dft_2d(combined_xy);

        dx = dft_2d(x);
        dy = dft_2d(y); 
    }

    else if(transform == Transform_2d::FFT2D){

        lhs = combined_xy;
        dx = x;
        dy = y;

        fft_2d_inplace(lhs);

        fft_2d_inplace(dx);
        fft_2d_inplace(dy);
    }

    else throw std::invalid_argument("run_linearity_2d_case: unsupported transform");


    for(std::size_t i = 0; i < x.nx(); ++i){
        for(std::size_t j = 0; j < x.ny(); ++j){

            rhs(i,j) = (alpha * dx(i,j)) + (beta * dy(i,j));
        }
    }

    if(!approx_equal_grid(lhs, rhs, abs_tol, rel_tol)){
        print_grid_failure_report(test_name, lhs, rhs, abs_tol);
        return false;
    }

    return true;
}


bool run_parseval_2d_case(
    const std::string& test_name, const Grid2D<Complex>& input, Transform_2d transform, Real abs_tol, Real rel_tol){

    if(input.nx() == 0 || input.ny() == 0) throw std::invalid_argument("run_parseval_2d_case: grid dimensions must be positive");

    Real physical_energy = 0.0;
    Real spectral_energy = 0.0;
    Grid2D<Complex> output;

    if(transform == Transform_2d::DFT2D){

        output = dft_2d(input);
    }

    else if(transform == Transform_2d::FFT2D){

        output = input;
        fft_2d_inplace(output);
    }

    else throw std::invalid_argument("run_parseval_2d_case: unsupported transform");

    physical_energy = physical_energy_2d(input);
    spectral_energy = spectral_energy_2d(output);

    Real diff = std::abs(spectral_energy - physical_energy);
    Real tolerance_bound = abs_tol + rel_tol * std::max(spectral_energy, physical_energy);

    if(diff > tolerance_bound){

        print_scalar_failure_report(test_name, physical_energy, spectral_energy, abs_tol, rel_tol);
        return false;
    }

    return true;
}


bool run_shift_2d_case(
    const std::string& test_name, const Grid2D<Complex>& input, std::ptrdiff_t shift_x, std::ptrdiff_t shift_y, 
    Transform_2d transform, Real abs_tol, Real rel_tol){

    if(input.nx() == 0 || input.ny() == 0) throw std::invalid_argument("run_shift_2d_case: grid dimensions must be positive");

    Grid2D<Complex> expected_output;
    Grid2D<Complex> actual_output;

    if(transform == Transform_2d::DFT2D){

        expected_output = dft_2d(circular_shift_2d(input, shift_x, shift_y));
        actual_output = dft_2d(input);
    }

    else if(transform == Transform_2d::FFT2D){

        expected_output = circular_shift_2d(input, shift_x, shift_y);
        fft_2d_inplace(expected_output);

        actual_output = input;
        fft_2d_inplace(actual_output);
    }

    else throw std::invalid_argument("run_shift_2d_case: unsupported transform");

    Complex omega = {std::cos(2.0 * PI * static_cast<Real>(shift_x) / static_cast<Real>(input.nx())), 
                    -1.0 * std::sin(2.0 * PI * static_cast<Real>(shift_x) / static_cast<Real>(input.nx()))};
    
    Complex omega_phase = {1,0};
    
    Complex alpha = {std::cos(2.0 * PI * static_cast<Real>(shift_y) / static_cast<Real>(input.ny())), 
                    -1.0 * std::sin(2.0 * PI * static_cast<Real>(shift_y) / static_cast<Real>(input.ny()))};


    for(std::size_t i = 0; i < input.nx(); ++i){

        Complex alpha_phase = {1,0};

        for(std::size_t j = 0; j < input.ny(); ++j){

            actual_output(i,j) *= omega_phase * alpha_phase;

            alpha_phase *= alpha;
        }

        omega_phase *= omega;
    }

    if(!approx_equal_grid(expected_output, actual_output, abs_tol, rel_tol)){

        print_grid_failure_report(test_name, expected_output, actual_output, abs_tol);
        return false;
    }

    return true;
}


bool run_conjugate_symmetry_2d_case(
    const std::string& test_name, const Grid2D<Complex>& real_input, Transform_2d transform, 
    Real abs_tol, Real rel_tol){

    if(real_input.nx() == 0 || real_input.ny() == 0) throw std::invalid_argument("run_conjugate_symmetry_2d_case: grid dimensions must be positive");

    Grid2D<Complex> output;

    Complex lhs;
    Complex rhs;

    if(transform == Transform_2d::DFT2D){

        output = dft_2d(real_input);
    }

    else if(transform == Transform_2d::FFT2D){

        output = real_input;
        fft_2d_inplace(output);
    }

    else throw std::invalid_argument("run_conjugate_symmetry_2d_case: unsupported transform");

    for(std::size_t i = 0; i <= real_input.nx() / 2; ++i){

        std::size_t mirror_i = (real_input.nx() - i) % real_input.nx();

        for(std::size_t j = 0; j < real_input.ny(); ++j){

            std::size_t mirror_j = (real_input.ny() - j) % real_input.ny();

            lhs = output(i,j);
            rhs = std::conj(output(mirror_i, mirror_j));

            if(!approx_equal_complex(lhs, rhs, abs_tol, rel_tol)){

                print_conjugate_symmetry_2d_failure_report(test_name, i, j, mirror_i, mirror_j, lhs, rhs);
                return false;
            }
        }
    }

    return true;
}

bool run_modulation_2d_case(
    const std::string& test_name, const Grid2D<Complex>& input, std::ptrdiff_t kx_shift, std::ptrdiff_t ky_shift, 
    Transform_2d transform, Real abs_tol, Real rel_tol){
  
    if(input.nx() == 0 || input.ny() == 0) throw std::invalid_argument("run_modulation_2d_case: grid dimensions must be positive");

    Grid2D<Complex> modulated_input = modulate_2d(input, kx_shift, ky_shift);

    Grid2D<Complex> output;
    Grid2D<Complex> modulated_output;
    Grid2D<Complex> expected_output(input.nx(), input.ny());

    std::ptrdiff_t nx = static_cast<std::ptrdiff_t> (input.nx());
    std::ptrdiff_t ny = static_cast<std::ptrdiff_t> (input.ny());

    std::size_t sx = static_cast<std::size_t>((((kx_shift % nx) + nx) % nx));
    std::size_t sy = static_cast<std::size_t>((((ky_shift % ny) + ny) % ny));

    if(transform == Transform_2d::DFT2D){

        output = dft_2d(input);
        modulated_output = dft_2d(modulated_input);
    }

    else if(transform == Transform_2d::FFT2D){

        output = input;
        fft_2d_inplace(output);

        modulated_output = modulated_input;
        fft_2d_inplace(modulated_output);
    }

    else throw std::invalid_argument("run_modulation_2d_case: unsupported transform");

    for(std::size_t i = 0; i < input.nx(); ++i){

        std::size_t shift_i = (i + input.nx() - sx) % input.nx();

        for(std::size_t j = 0; j < input.ny(); ++j){

            expected_output(i,j) = output(shift_i, (j + input.ny() - sy) % input.ny());
        }
    }

    if(!approx_equal_grid(expected_output, modulated_output, abs_tol, rel_tol)){

        print_grid_failure_report(test_name, expected_output, modulated_output, abs_tol);
        return false;
    }


    return true;
}


bool run_fft2d_vs_dft2d_case(
    const std::string& test_name, const Grid2D<Complex>& input, Real abs_tol, Real rel_tol){

    if(input.nx() == 0 || input.ny() == 0) throw std::invalid_argument("run_fft2d_vs_dft2d_case: grid dimensions must be positive");

    Grid2D<Complex> discrete = dft_2d(input);

    Grid2D<Complex> fast = input;
    fft_2d_inplace(fast);

    if(!approx_equal_grid(discrete, fast, abs_tol, rel_tol)){

        print_grid_failure_report(test_name, discrete, fast, abs_tol);
        return false;
    }


    return true;
}


bool run_power_of_two_enforcement_2d_case(const std::string& test_name, std::size_t nx, std::size_t ny){

    Grid2D<Complex> input(nx, ny);

    try{
        fft_2d_inplace(input);
    } catch (const std::invalid_argument&){
        return true;
    }

    std::cout << "FAIL: " << test_name << "\n";
    std::cout << "Expected std::invalid_argument for dimensions ("
              << nx << ", " << ny << ") but no exception was thrown.\n\n";

    return false;
}


bool run_separability_2d_case(const std::string& test_name, const Grid2D<Complex>& input,
    Transform_2d transform, Real abs_tol, Real rel_tol){

    if(input.nx() == 0 || input.ny() == 0) throw std::invalid_argument("run_separability_2d_case: grid dimensions must be positive");

    Grid2D<Complex> call_fucnction;
    Grid2D<Complex> manual = input;

    if(transform == Transform_2d::DFT2D){

        call_fucnction = dft_2d(input);

        // row tranforms

        ComplexVec row(input.ny());

        for(std::size_t i = 0; i < input.nx(); ++i){

            for(std::size_t j = 0; j < input.ny(); ++j){

                row[j] = manual(i,j);
            }

            row = dft_1d(row);

            for(std::size_t k = 0; k < input.ny(); ++k){

                manual(i,k) = row[k];
            }
        }

        // column tranforms

        ComplexVec col(input.nx());

        for(std::size_t j = 0; j < input.ny(); ++j){

            for(std::size_t i = 0; i < input.nx(); ++i){

                col[i] = manual(i,j);
            }

            col = dft_1d(col);

            for(std::size_t k = 0; k < input.nx(); ++k){

                manual(k,j) = col[k];
            }
        }
    }

    else if(transform == Transform_2d::FFT2D){

        call_fucnction = input;
        fft_2d_inplace(call_fucnction); // already checks if nx and ny power of 2

        ComplexVec row(input.ny());

        for(std::size_t i = 0; i < input.nx(); ++i){

            for(std::size_t j = 0; j < input.ny(); ++j){

                row[j] = manual(i,j);
            }

            fft_1d_inplace(row);

            for(std::size_t k = 0; k < input.ny(); ++k){

                manual(i,k) = row[k];
            }
        }

        // column tranforms

        ComplexVec col(input.nx());

        for(std::size_t j = 0; j < input.ny(); ++j){

            for(std::size_t i = 0; i < input.nx(); ++i){

                col[i] = manual(i,j);
            }

            fft_1d_inplace(col);

            for(std::size_t k = 0; k < input.nx(); ++k){

                manual(k,j) = col[k];
            }
        }
    }

    else throw std::invalid_argument("run_separability_2d_case: unsupported transform");


    if(!approx_equal_grid(call_fucnction, manual, abs_tol, rel_tol)){

        print_grid_failure_report(test_name, manual, call_fucnction, abs_tol);
        return false;
    }


    return true;
}


bool run_spectral_decay_2d_case(const std::string& test_name, const Grid2D<Complex>& input,
    Transform_2d transform, Real alpha, Real time, Real Lx, Real Ly, bool expect_real_output, Real abs_tol, Real rel_tol){

    if(input.nx() == 0 || input.ny() == 0) throw std::invalid_argument("run_spectral_decay_2d_case: grid dimensions must be positive");

    if(time < 0.0) throw std::invalid_argument("run_spectral_decay_2d_case: time must be nonnegative");

    if(alpha < 0.0) throw std::invalid_argument("run_spectral_decay_2d_case: alpha must be nonnegative");

    if(Lx <= 0.0 || Ly <= 0.0) throw std::invalid_argument("run_spectral_decay_2d_case: domain lengths Lx and Ly must be positive");

    std::ptrdiff_t nx_signed = static_cast<std::ptrdiff_t>(input.nx());
    std::ptrdiff_t ny_signed = static_cast<std::ptrdiff_t>(input.ny());
    
    Real initial_physical_energy = physical_energy_2d(input);

    if(initial_physical_energy <= abs_tol) throw std::invalid_argument("run_spectral_decay_2d_case: initial physical energy is near zero, so the energy ratio is undefined or numerically unstable");

    Real decayed_physical_energy = 0.0;
    Real initial_spectral_energy = 0.0;
    Real decayed_spectral_energy = 0.0;
    Real expected_ratio = 0.0;
    Real actual_ratio = 0.0;
    Real kx_sq = 0.0;
    Real ky_sq = 0.0;
    Real pi_factor_x = 4.0 * PI * PI / (Lx * Lx);
    Real pi_factor_y = 4.0 * PI * PI / (Ly * Ly);
    Real max_imaginary = 0.0;

    Grid2D<Complex> initial_spectrum;
    Grid2D<Complex> decayed_spectrum;
    Grid2D<Complex> decayed_physical;

    if(transform == Transform_2d::DFT2D){

        initial_spectrum = dft_2d(input);
        decayed_spectrum = initial_spectrum;
    }

    else if(transform == Transform_2d::FFT2D){

        initial_spectrum = input;
        fft_2d_inplace(initial_spectrum);
        decayed_spectrum = initial_spectrum;
    }

    else throw std::invalid_argument("run_spectral_decay_2d_case: unsupported transform");

    for(std::size_t i = 0; i < input.nx(); ++i){

        std::ptrdiff_t i_signed = static_cast<std::ptrdiff_t>(i);

        std::ptrdiff_t mx;

        if(i <= input.nx()/2) mx = i_signed;
        else mx = i_signed - nx_signed;

        Real mx_real = static_cast<Real>(mx);
        kx_sq = pi_factor_x * mx_real * mx_real;

        for(std::size_t j = 0; j < input.ny(); ++j){

            std::ptrdiff_t j_signed = static_cast<std::ptrdiff_t>(j);

            std::ptrdiff_t my;

            if(j <= input.ny()/2) my = j_signed;
            else my = j_signed - ny_signed;

            Real my_real = static_cast<Real>(my);

            ky_sq = pi_factor_y * my_real * my_real;

            decayed_spectrum(i,j) *= std::exp(-1.0 * alpha * time * (kx_sq + ky_sq));
        }
    }

    if(transform == Transform_2d::DFT2D){

        decayed_physical = idft_2d(decayed_spectrum);
    }

    else if(transform == Transform_2d::FFT2D){

        decayed_physical = decayed_spectrum;
        ifft_2d_inplace(decayed_physical);
    }

    initial_spectral_energy = spectral_energy_2d(initial_spectrum);

    if(initial_spectral_energy <= abs_tol) throw std::invalid_argument("run_spectral_decay_2d_case: initial spectral energy is near zero, so the energy ratio is undefined or numerically unstable");

    decayed_spectral_energy = spectral_energy_2d(decayed_spectrum);
    decayed_physical_energy = physical_energy_2d(decayed_physical);

    expected_ratio = decayed_spectral_energy / initial_spectral_energy;
    actual_ratio = decayed_physical_energy / initial_physical_energy;
    max_imaginary = max_imag_part(decayed_physical);

    bool ratio_failed = !approx_equal_real(expected_ratio, actual_ratio, abs_tol, rel_tol);
    bool imag_failed = expect_real_output && (max_imaginary > abs_tol);

    if(ratio_failed || imag_failed){

    print_spectral_decay_failure_report(test_name, expected_ratio, actual_ratio, initial_physical_energy, decayed_physical_energy, initial_spectral_energy,
                                        decayed_spectral_energy, max_imaginary, expect_real_output, abs_tol, rel_tol);

    return false;
    }

    return true;
}


bool run_fft2d_vs_fftw_case(const std::string& test_name, const Grid2D<Complex>& input, Real abs_tol, Real rel_tol){

    if(input.nx() == 0 || input.ny() == 0) throw std::invalid_argument("run_fft2d_vs_fftw_case: grid dimensions must be positive");

    if(input.nx() > static_cast<std::size_t>(std::numeric_limits<int>::max()) || input.ny() > static_cast<std::size_t>(std::numeric_limits<int>::max())){
        throw std::invalid_argument("run_fft2d_vs_fftw_case: grid dimensions exceed FFTW int dimension limit");
    }

    // fftw uses int as input paranters for indexing as int is standard in C

    int nx = static_cast<int>(input.nx());
    int ny = static_cast<int>(input.ny());

    fftw_complex* in = fftw_alloc_complex(input.size());
    fftw_complex* out = fftw_alloc_complex(input.size());

    if (in == nullptr || out == nullptr) {
        if (in != nullptr) fftw_free(in);
        if (out != nullptr) fftw_free(out);

        throw std::runtime_error("run_fft2d_vs_fftw_case: FFTW allocation failed");
    }

    Grid2D<Complex> custom_spectrum = input;
    Grid2D<Complex> custom_output;
    Grid2D<Complex> fftw_in_custom_out;
    Grid2D<Complex> fftw_spectrum(input.nx(), input.ny());
    Grid2D<Complex> fftw_output(input.nx(), input.ny());

    fft_2d_inplace(custom_spectrum);

    for(std::size_t i = 0; i < input.nx(); ++i){

        std::size_t i_ny = i*input.ny();

        for(std::size_t j = 0; j < input.ny(); ++j){

            std::size_t i_ny_j = i_ny + j;

            in[i_ny_j][0] = input(i, j).real();
            in[i_ny_j][1] = input(i, j).imag();
        }
    }

    fftw_plan fwd = fftw_plan_dft_2d(nx, ny, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    if (fwd == nullptr) {
        fftw_free(in);
        fftw_free(out);

        throw std::runtime_error("run_fft2d_vs_fftw_case: FFTW forward plan creation failed");
    }

    fftw_execute(fwd);

    fftw_plan inv = fftw_plan_dft_2d(nx, ny, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);

    if (inv == nullptr) {
        fftw_destroy_plan(fwd);
        fftw_free(in);
        fftw_free(out);

        throw std::runtime_error("run_fft2d_vs_fftw_case: FFTW inverse plan creation failed");
    }

    fftw_execute(inv);

    for(std::size_t k = 0; k < input.size(); ++k){

        in[k][0] /= static_cast<Real>(input.size());
        in[k][1] /= static_cast<Real>(input.size());
    }

    for(std::size_t i = 0; i < input.nx(); ++i){

        std::size_t i_ny = i*input.ny();

        for(std::size_t j = 0; j < input.ny(); ++j){

            std::size_t i_ny_j = i_ny + j;

            fftw_spectrum(i,j) = {out[i_ny_j][0], out[i_ny_j][1]};
            fftw_output(i,j) = {in[i_ny_j][0], in[i_ny_j][1]};
        }
    }

    fftw_destroy_plan(fwd);
    fftw_destroy_plan(inv);
    fftw_free(in);
    fftw_free(out);

    custom_output = custom_spectrum;
    fftw_in_custom_out = fftw_spectrum;

    ifft_2d_inplace(fftw_in_custom_out);
    ifft_2d_inplace(custom_output);


    bool forward = approx_equal_grid(fftw_spectrum, custom_spectrum, abs_tol, rel_tol);
    bool inverse = approx_equal_grid(fftw_output, fftw_in_custom_out, abs_tol, rel_tol);
    bool round = approx_equal_grid(fftw_output, custom_output, abs_tol, rel_tol);

    if(!forward){

        std::cout << "Forward FFT failure \n";
        
        print_grid_failure_report(test_name, fftw_spectrum, custom_spectrum, abs_tol);
        return false;
    } 

    else if(!inverse){

        std::cout << "Inverse FFT failure \n";

        print_grid_failure_report(test_name, fftw_output, fftw_in_custom_out, abs_tol);
        return false;
    }

    else if(!round){

        std::cout << "Round Trip failure \n";
        print_grid_failure_report(test_name, fftw_output, custom_output, abs_tol);
        return false;
    }

    return true;
}