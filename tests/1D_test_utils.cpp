
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
#include "dft1d.hpp"
#include "fft1d.hpp"
#include "1D_test_utils.hpp"



bool approx_equal_complex(const Complex& a, const Complex& b, Real abs_tol, Real rel_tol){

    return std::abs(a-b) <= abs_tol + rel_tol * std::max(std::abs(a), std::abs(b));
}





bool approx_equal_vector(const ComplexVec& expected, const ComplexVec& actual, Real abs_tol, Real rel_tol){

    if(expected.size() != actual.size()) return false;

    for(std::size_t k = 0; k < expected.size(); k++){

        if(!approx_equal_complex(expected[k], actual[k], abs_tol, rel_tol)) return false;

    }
    return true;
}




double max_abs_error(const ComplexVec& expected, const ComplexVec& actual){
    
    if(expected.size() != actual.size()) throw std::invalid_argument("Expected and actual vector sizes do not match");

    Real max_error = 0.0;

    for(std::size_t k = 0; k < expected.size(); k++){

        Real current_error = std::abs(expected[k] - actual[k]);

        if(current_error > max_error) max_error = current_error;
    }

    return max_error;
}




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



void print_scalar_failure_report(const std::string& test_name,
                                 double expected,
                                 double actual)
{
    std::cout << "FAIL: " << test_name << "\n";

    std::cout << std::setprecision(16);

    double abs_error = std::abs(expected - actual);

    double rel_error = 0.0;
    double scale = std::max(std::abs(expected), std::abs(actual));

    if(scale > 0.0) {
        rel_error = abs_error / scale;
    }

    std::cout << "Expected value:\n";
    std::cout << "  " << expected << "\n";

    std::cout << "Actual value:\n";
    std::cout << "  " << actual << "\n";

    std::cout << "Error metrics:\n";
    std::cout << "  absolute_error = " << abs_error << "\n";
    std::cout << "  relative_error = " << rel_error << "\n";
}



void print_conjugate_symmetry_failure_report(const std::string& test_name, std::size_t k, std::size_t mirror,
    Complex lhs, Complex rhs){

    std::cout << "FAIL: " << test_name << "\n";
    std::cout << std::setprecision(16);

    std::cout << "Conjugate symmetry failed at frequency pair:\n";
    std::cout << "  k      = " << k << "\n";
    std::cout << "  mirror = " << mirror << "\n";

    std::cout << "Values:\n";
    std::cout << "  X[k]              = ("
              << lhs.real() << ", " << lhs.imag() << ")\n";

    std::cout << "  conj(X[mirror])   = ("
              << rhs.real() << ", " << rhs.imag() << ")\n";

    std::cout << "Error metrics:\n";
    std::cout << "  absolute_error = " << std::abs(lhs - rhs) << "\n";
}






bool run_known_output_case(const std::string& test_name, const ComplexVec& input, const ComplexVec& expected, 
 Transform_1d transform, Real abs_tol, Real rel_tol){

    ComplexVec spectrum;

    if(transform == Transform_1d::DFT){
        spectrum = dft_1d(input); 
    } 
    
    else if(transform == Transform_1d::FFT){

        spectrum = input;
        
        fft_1d_inplace(spectrum);
    }

   else{
    throw std::invalid_argument("run_known_output_case: unsupported Transform_1d value");
    }

    if(!approx_equal_vector(expected, spectrum, abs_tol, rel_tol)) {
        print_failure_report(test_name, expected, spectrum);
        return false;
    }
    
    return true;
}



bool run_round_trip_case(const std::string& test_name, const ComplexVec& input, Transform_1d transform, 
 Real abs_tol, Real rel_tol){

    ComplexVec spectrum;
    ComplexVec reconstructed_input;

    if(transform == Transform_1d::DFT){
        spectrum = dft_1d(input); 
        reconstructed_input = idft_1d(spectrum);
    } 
    
    else if(transform == Transform_1d::FFT){

        spectrum = input;
        
        fft_1d_inplace(spectrum);
        
        reconstructed_input = spectrum;

        ifft_1d_inplace(reconstructed_input);

    }

   else{
    throw std::invalid_argument("run_round_trip_case: unsupported Transform_1d value");
    }

    if(!approx_equal_vector(input, reconstructed_input, abs_tol, rel_tol)){
        print_failure_report(test_name, input, reconstructed_input);
        return false;
    }

    return true;

}




bool run_linearity_case(const std::string& test_name, const ComplexVec& x, const ComplexVec& y, 
 Complex alpha, Complex beta, Transform_1d transform, Real abs_tol, Real rel_tol) {


    if(x.size() != y.size()) throw std::invalid_argument("Linearity Test: x.size != y.size");

    std::size_t N = x.size();
 
    // Build alpha*x + beta*y in physical space
    ComplexVec combined(N);
    for(std::size_t j = 0; j < N; j++){
        combined[j] = alpha * x[j] + beta * y[j];
    }

    ComplexVec rhs(N);
    ComplexVec lhs(N);

    if(transform == Transform_1d::DFT){

        // DFT of the combined signal
        lhs = dft_1d(combined);

        // alpha*DFT(x) + beta*DFT(y) in frequency space
        ComplexVec dx = dft_1d(x);
        ComplexVec dy = dft_1d(y);
        for(std::size_t k = 0; k < N; k++){
            rhs[k] = alpha * dx[k] + beta * dy[k];
            }
    }
 
    else if(transform == Transform_1d::FFT){

        // FFT of the combined signal

        lhs = combined;

        fft_1d_inplace(lhs);

        // alphaFFT(x) + beta*FFT(y) in frequency space

        ComplexVec copy_x = x;
        ComplexVec copy_y = y;

        fft_1d_inplace(copy_x);
        fft_1d_inplace(copy_y);
        for(std::size_t k = 0; k < N; k++){
            rhs[k] = alpha * copy_x[k] + beta * copy_y[k];
            }
    }

    else{
    throw std::invalid_argument("Linearity test: unsupported Transform_1d value");
    }

 
    if(!approx_equal_vector(lhs, rhs, abs_tol, rel_tol)){
        print_failure_report(test_name, lhs, rhs);
        return false;
    }
 
    return true;
}




bool run_parseval_case(const std::string& test_name, const ComplexVec& input, Transform_1d transform,
 Real abs_tol, Real rel_tol){


    std::size_t N = input.size();

    if(N == 0) throw std::invalid_argument("Parseval test: input size is 0");

    Real physical_energy = 0.0;
    Real fourier_energy = 0.0;
    ComplexVec output;

    if(transform == Transform_1d::DFT){
        
        output = dft_1d(input);
    }

    else if(transform == Transform_1d::FFT){

        output = input;
        fft_1d_inplace(output);
    }

    else{
    throw std::invalid_argument("Parseval test: unsupported Transform_1d value");
    }

    for(std::size_t i = 0; i < N; ++i){
        physical_energy += std::norm(input[i]);
        fourier_energy += std::norm(output[i]);
        }
    fourier_energy /= N;

    Real diff = std::abs(fourier_energy - physical_energy);
    Real tolerance_bound = abs_tol + rel_tol * std::max(fourier_energy, physical_energy);

    if(diff > tolerance_bound){

        print_scalar_failure_report(test_name, physical_energy, fourier_energy);
        return false;
    }

    return true;
}


bool run_time_shift_case(const std::string& test_name, const ComplexVec& input, int shift, Transform_1d transform,
 Real abs_tol, Real rel_tol){

    std::size_t N = input.size();

    if(N == 0) throw std::invalid_argument("Time Shift test: input size is 0");

    ComplexVec expected_output;
    ComplexVec actual_output;
    ComplexVec shifted_input(N);

    std::size_t s = static_cast<std::size_t>(((shift % static_cast<int>(N)) + static_cast<int>(N)) % static_cast<int>(N));

    Real theta = 2 * PI * static_cast<Real>(s) / static_cast<Real>(N);

    Complex omega = {std::cos(theta), (-std::sin(theta))};
    Complex phase = {1.0 , 0.0};


    for(std::size_t i = 0; i < N; ++i){

        shifted_input[i] = input[(i + N - s) % N];
    }

    if(transform == Transform_1d::DFT){

        actual_output = dft_1d(shifted_input);
        expected_output = dft_1d(input);
    }

    else if(transform == Transform_1d::FFT){

        actual_output = shifted_input;
        fft_1d_inplace(actual_output);

        expected_output = input;
        fft_1d_inplace(expected_output);

    }

    else{
    throw std::invalid_argument("Time Shift test: unsupported Transform_1d value");
    }

    for(std::size_t k = 0; k < N; ++k){
            expected_output[k] *= phase;
            phase *= omega;
            }

    if(!approx_equal_vector(expected_output, actual_output, abs_tol, rel_tol)){
        print_failure_report(test_name, expected_output, actual_output);
        return false;
    }

    return true;
}



bool run_conjugate_symmetry_case(const std::string& test_name, const ComplexVec& input, Transform_1d transform,
 Real abs_tol, Real rel_tol){

    std::size_t N = input.size();

    Complex lhs;
    Complex rhs;

    if(N == 0) throw std::invalid_argument("Conjugate Symmetry test: input size is 0");

    for(std::size_t i = 0; i < N; ++i){

        if(std::abs(input[i].imag()) > abs_tol) throw std::invalid_argument("Conjugate Symmetry test: input has imaginary values");
    }

    ComplexVec output;

    if(transform == Transform_1d::DFT) output = dft_1d(input);

    else if(transform == Transform_1d::FFT){

        output = input;
        fft_1d_inplace(output);
    }

    else{
    throw std::invalid_argument("Conjugate Symmetry test: unsupported Transform_1d value");
    }

    for(std::size_t k = 0; k <= N/2; ++k){

        std::size_t mirror = (N-k) % N; 

        lhs = output[k];
        rhs = std::conj(output[mirror]);

        if(!approx_equal_complex(lhs, rhs, abs_tol, rel_tol)){

            print_conjugate_symmetry_failure_report(test_name, k, mirror, lhs, rhs);
            return false;
        }

    }

    return true;
}


bool run_inverse_known_output_case(const std::string& test_name, const ComplexVec& spectrum, 
 const ComplexVec& expected, ITransform_1d transform, Real abs_tol, Real rel_tol){


    std::size_t N = spectrum.size();

    if(expected.size() != N) throw std::invalid_argument("Inverse Known Output test: size mismatch");

    if(N == 0) throw std::invalid_argument("Inverse Known Output test: input size is 0");

    ComplexVec actual;

    if(transform == ITransform_1d::IDFT) actual = idft_1d(spectrum);

    else if(transform == ITransform_1d::IFFT){

        actual = spectrum;
        ifft_1d_inplace(actual);
    }


    else{
    throw std::invalid_argument("Inverse Known Output test: unsupported Transform_1d value");
    }

    if(!approx_equal_vector(expected, actual, abs_tol, rel_tol)){
        
        print_failure_report(test_name, expected, actual);
        return false;
    }

    return true;
}


bool run_modulation_case(const std::string& test_name, const ComplexVec& input, int frequency_shift,
 Transform_1d transform, Real abs_tol, Real rel_tol){

    std::size_t N = input.size();

    if(N == 0) throw std::invalid_argument("Modulation case test: input size is 0");

    ComplexVec expected_output;
    ComplexVec actual_output;
    ComplexVec shifted_input(N);

    std::size_t s = static_cast<std::size_t>(((frequency_shift % static_cast<int>(N)) + static_cast<int>(N)) % static_cast<int>(N));

    Real theta = 2 * PI * static_cast<Real>(s) / static_cast<Real>(N);

    Complex omega = {std::cos(theta), (std::sin(theta))};
    Complex phase = {1.0 , 0.0};

    for(std::size_t i = 0; i < N; ++i){
        shifted_input[i] = input[i] * phase;
        phase *= omega;
    }

    if(transform == Transform_1d::DFT){

        actual_output = dft_1d(shifted_input);
        expected_output = dft_1d(input);
    }

    else if(transform == Transform_1d::FFT){

        actual_output = shifted_input;
        fft_1d_inplace(actual_output);

        expected_output = input;
        fft_1d_inplace(expected_output);
    }

    else{
    throw std::invalid_argument("Modulation case test: unsupported Transform_1d value");
    }

    ComplexVec original_output = expected_output;

    for(std::size_t k = 0; k < N; ++k){
        
        expected_output[k] = original_output[(k + N - s) % N];
    }

    if(!approx_equal_vector(expected_output, actual_output, abs_tol, rel_tol)){

        print_failure_report(test_name, expected_output, actual_output);
        return false;
    }

    return true;
}




bool run_fft_vs_dft_case(const std::string& test_name, const ComplexVec& input, Real abs_tol, Real rel_tol){

    std::size_t N = input.size();

    if(N == 0) throw std::invalid_argument("Modulation case test: input size is 0");

    ComplexVec expected_output;
    ComplexVec actual_output;

    expected_output = dft_1d(input);

    actual_output = input;
    fft_1d_inplace(actual_output);

    if(!approx_equal_vector(expected_output, actual_output, abs_tol, rel_tol)){
        
        print_failure_report(test_name, expected_output, actual_output);
        return false;
    }

    return true;

}

