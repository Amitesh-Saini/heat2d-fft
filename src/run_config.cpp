// run_config.cpp
// Responsibility:
//   Implement helper functionality for the high-level RunConfig structure.
//
//   This file should contain logic related to complete simulation-run settings,
//   not the numerical heat solver itself.
//
//   Examples of responsibilities that belong here:
//     - default values for run-level settings,
//     - validation of run-level options,
//     - helper functions for interpreting initial-condition choices,
//     - helper functions for output/run naming.
//
//   The goal is to keep main.cpp small and prevent Heat2DConfig from being
//   overloaded with non-solver responsibilities.

#include "run_config.hpp"          

#include "time_grid.hpp"          
#include "initial_conditions.hpp"  
#include "heat2d_fourier.hpp"      

#include <string>           
#include <variant>                
#include <vector>                 
#include <stdexcept>   

template <typename... T>
struct visitor : T... {
    using T::operator()...;
};

template <typename... T> visitor(T...) -> visitor<T...>;   


std::string fft_backend_to_string(FftBackend backend){

    switch (backend) {
        case FftBackend::custom: return "custom";
        case FftBackend::fftw:   return "fftw";
    }

    throw std::invalid_argument("fft_backend_to_string: unhandled FftBackend value");
}


FftBackend fft_backend_from_string(const std::string& name){

    if (name == "custom") return FftBackend::custom;
    if (name == "fftw")   return FftBackend::fftw;

    throw std::invalid_argument("fft_backend_from_string: unknown backend '" + name + "'");
}


std::string initial_condition_type_name(const InitialConditionParams& ic){

    return std::visit(visitor{
    [](const GaussianIcParams&)          { return std::string("gaussian"); },
    [](const HotSquareIcParams&)         { return std::string("hot_square"); },
    [](const ConstantIcParams&)          { return std::string("constant"); },
    [](const SingleFourierModeIcParams&) { return std::string("single_fourier_mode"); },
    [](const MultiFourierModeIcParams&)  { return std::string("multi_fourier_mode"); }
    }, ic);
}


Heat2DConfig make_heat2d_config(const RunConfig& run_config){

    Heat2DConfig heat2d;

    heat2d.Lx = run_config.solver.Lx;
    heat2d.Ly = run_config.solver.Ly;
    heat2d.alpha = run_config.solver.alpha;
    heat2d.nx = run_config.solver.nx;
    heat2d.ny = run_config.solver.ny;

    std::visit(visitor{

        [&](const UniformTimeSpec& spec) {heat2d.output_times = make_uniform_time_grid(spec.t_start, spec.t_end, spec.num_snapshots);},
        [&](const ExplicitTimeSpec& spec) {heat2d.output_times = spec.times;}, 

    }, run_config.time_spec);

    return heat2d;
}


Grid2D<Real> make_initial_condition(const RunConfig& run_config){

    Real Lx = run_config.solver.Lx;
    Real Ly = run_config.solver.Ly;
    std::size_t nx = run_config.solver.nx;
    std::size_t ny = run_config.solver.ny;

    return std::visit(visitor{

        [&](const GaussianIcParams& spec) -> Grid2D<Real> {return make_gaussian_ic(Lx, Ly, nx, ny, spec.amplitude, spec.sigma, spec.image_radius_x, spec.image_radius_y);},

        [&](const ConstantIcParams& spec) -> Grid2D<Real> {return make_constant_ic(Lx,Ly, nx, ny, spec.T0);},

        [&](const HotSquareIcParams& spec) -> Grid2D<Real> {
            return make_hot_square_ic(Lx, Ly, nx, ny, spec.amplitude, spec.width_x, spec.width_y, spec.smooth_width_x, spec.smooth_width_y);},

        [&](const SingleFourierModeIcParams& spec) -> Grid2D<Real> {return make_custom_single_fourier_mode_ic(Lx, Ly, nx, ny, spec.kx, spec.ky, spec.amplitude, spec.phase);},

        [&](const MultiFourierModeIcParams& spec) -> Grid2D<Real> {
            std::vector<FourierMode2D> modes = spec.modes.empty() ? make_default_fourier_modes() : spec.modes; 
            return make_custom_multi_fourier_mode_ic(Lx, Ly, nx, ny, modes);},
            
    }, run_config.initial_condition);

}