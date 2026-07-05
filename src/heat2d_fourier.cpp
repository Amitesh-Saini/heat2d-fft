#include "heat2d_fourier.hpp"
#include "fft2d.hpp"
#include "wavenumbers.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// heat2d_fourier.cpp
// Responsibility:
//   Implementation of the 2D Fourier spectral solver for the periodic 2D
//   heat equation.
//
// Method:
//   - Store the initial temperature field in physical space.
//   - Convert it to complex form and transform it with a 2D FFT.
//   - Evolve each Fourier mode by exp(-alpha * (Kx^2 + Ky^2) * t).
//   - Inverse transform to recover physical-space temperature snapshots.

namespace {

bool is_power_of_two(std::size_t n){
    return n > 0 && (n & (n - 1)) == 0;
}

} // namespace


const Heat2DConfig& Heat2DFourierSolver::config() const{
    return config_;
}


void Heat2DFourierSolver::validate_config() const{
    
    if(config_.nx < 2 || config_.ny < 2){
        throw std::invalid_argument("Heat2DConfig: nx and ny must be at least 2");
    }

    if(!is_power_of_two(config_.nx) || !is_power_of_two(config_.ny)){
        throw std::invalid_argument("Heat2DConfig: nx and ny must be powers of two for the radix-2 FFT backend");
    }

    if(!std::isfinite(config_.Lx) || !std::isfinite(config_.Ly)){
        throw std::invalid_argument("Heat2DConfig: Lx and Ly must be finite");
    }

    if(config_.Lx <= Real{0} || config_.Ly <= Real{0}){
        throw std::invalid_argument("Heat2DConfig: Lx and Ly must be positive");
    }

    if(!std::isfinite(config_.alpha)){
        throw std::invalid_argument("Heat2DConfig: alpha must be finite");
    }

    if(config_.alpha <= Real{0}){
        throw std::invalid_argument("Heat2DConfig: alpha must be positive");
    }

    if(config_.output_times.empty()){
        throw std::invalid_argument("Heat2DConfig: output_times must not be empty");
    }

    for(const Real t : config_.output_times){
        if(!std::isfinite(t)){
            throw std::invalid_argument("Heat2DConfig: output times must be finite");
        }

        if(t < Real{0}){
            throw std::invalid_argument("Heat2DConfig: output times must be nonnegative");
        }
    }

    if(!std::is_sorted(config_.output_times.begin(), config_.output_times.end())){
        throw std::invalid_argument("Heat2DConfig: output_times must be sorted in nondecreasing order");
    }
}


void Heat2DFourierSolver::validate_initial_condition_shape(const Grid2D<Real>& initial_temperature) const{

    if(initial_temperature.nx() != config_.nx || initial_temperature.ny() != config_.ny){
        throw std::invalid_argument("Heat2DFourierSolver: initial condition grid shape does not match solver config");
    }
}


Heat2DFourierSolver::Heat2DFourierSolver(const Heat2DConfig& config): config_(config){

    validate_config();
}


void Heat2DFourierSolver::set_initial_condition(const Grid2D<Real>& initial_temperature){

    validate_initial_condition_shape(initial_temperature);

    initial_temperature_field_ = initial_temperature;
    has_initial_condition_ = true;
}


Grid2D<Real> Heat2DFourierSolver::make_snapshot_at_time(
    const Grid2D<Complex>& initial_spectral_coefficients, const Grid2D<Real>& squared_wavenumbers, Real time) const{

    if(initial_spectral_coefficients.nx() != config_.nx || initial_spectral_coefficients.ny() != config_.ny){
        throw std::invalid_argument("Heat2DFourierSolver::make_snapshot_at_time: initial spectral coefficient grid shape does not match solver config");
    }

    if(squared_wavenumbers.nx() != config_.nx || squared_wavenumbers.ny() != config_.ny){
        
        throw std::invalid_argument("Heat2DFourierSolver::make_snapshot_at_time: squared wavenumber grid shape does not match solver config");
    }
    
    Grid2D<Complex> spectral_coefficients = initial_spectral_coefficients;
    Grid2D<Real> snapshot(config_.nx, config_.ny);

    for(std::size_t i = 0; i < spectral_coefficients.nx(); ++i){

        for(std::size_t j = 0; j < spectral_coefficients.ny(); ++j){

            Real decay = std::exp(- config_.alpha * squared_wavenumbers(i,j) * time);
            spectral_coefficients(i,j) *= decay;
        }
    }

    ifft_2d_inplace(spectral_coefficients);

    for(std::size_t i = 0; i < snapshot.nx(); ++i){

        for(std::size_t j = 0; j < snapshot.ny(); ++j){

            snapshot(i,j) = spectral_coefficients(i,j).real();
        }
    }

    return snapshot;
}


std::vector<Grid2D<Real>> Heat2DFourierSolver::solve(){

    if(!has_initial_condition_) {
        throw std::runtime_error("Heat2DFourierSolver::solve: initial condition has not been set");
    }

    std::vector<Grid2D<Real>> snapshots;
    snapshots.reserve(config_.output_times.size());

    Grid2D<Complex> initial_spectral_coefficients(config_.nx, config_.ny);

    for(std::size_t i = 0; i < initial_spectral_coefficients.nx(); ++i){

        for(std::size_t j = 0; j < initial_spectral_coefficients.ny(); ++j){

            initial_spectral_coefficients(i,j) = {initial_temperature_field_(i,j), Real{0}};
        }
    }

    fft_2d_inplace(initial_spectral_coefficients);

    RealVec kx = build_fourier_wavenumbers(config_.nx, config_.Lx);
    RealVec ky = build_fourier_wavenumbers(config_.ny, config_.Ly);

    Grid2D<Real> squared_wavenumbers = build_squared_wavenumber_grid(kx, ky);

    for(std::size_t k = 0; k < config_.output_times.size(); ++k){

        snapshots.push_back(make_snapshot_at_time(initial_spectral_coefficients, squared_wavenumbers, config_.output_times[k]));
    }

    return snapshots;
}


