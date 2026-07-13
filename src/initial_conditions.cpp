// initial_conditions.cpp
// Responsibility:
//   Implements concrete initial-condition generators for the 2D periodic FFT
//   heat-equation solver.
//
//   Each function builds a physical-space Grid2D<Real> on
//       [-Lx/2, Lx/2) x [-Ly/2, Ly/2)
//   and returns it ready for forward FFT and spectral time evolution.
//
// Implemented initial conditions:
//   - constant field
//   - truncated periodic image-sum Gaussian
//   - tanh-smoothed hot square
//   - single plane-wave Fourier mode
//   - multi-mode plane-wave Fourier field


#include "initial_conditions.hpp"

std::vector<FourierMode2D> make_default_fourier_modes() {
    return {
        {std::ptrdiff_t{1}, std::ptrdiff_t{1}, Real{1.0},  Real{0}},
        {std::ptrdiff_t{-1}, std::ptrdiff_t{1}, Real{0.5},  -PI / Real{2}},
        {std::ptrdiff_t{1}, std::ptrdiff_t{0}, Real{0.25}, PI / Real{2}},
        {std::ptrdiff_t{0}, std::ptrdiff_t{-1}, Real{0.75}, PI}
    };
}

RealVec make_periodic_coordinates(Real L, std::size_t n) {

    if(n < 2) throw std::invalid_argument("make_periodic_coordinates: n must be at least 2");

    if(!std::isfinite(L) || L <= Real{0}) throw std::invalid_argument("make_periodic_coordinates: L must be finite and positive");

    const Real dx = L / static_cast<Real>(n);
    const Real origin = -L / Real{2};

    RealVec coordinates(n);

    for(std::size_t i = 0; i < n; ++i){
        
        coordinates[i] = origin + static_cast<Real>(i) * dx;
    }

    return coordinates;
}


void validate_grid_spec_2d(Real Lx, Real Ly, std::size_t nx, std::size_t ny, const ValidationConfig& validation){


    if(nx <= 1 || ny <= 1) throw std::invalid_argument("grid dimensions must be at least 2");

    if(!std::isfinite(Lx) || !std::isfinite(Ly)) throw std::invalid_argument("lengths must be finite");

    if(Lx < validation.min_domain_length || Ly < validation.min_domain_length) throw std::invalid_argument("length too small");

    Real dx = Lx / static_cast<Real>(nx);
    Real dy = Ly / static_cast<Real>(ny);

    if(dx < validation.min_grid_spacing || dy < validation.min_grid_spacing) throw std::invalid_argument("step size too small");
}


Grid2D<Real> make_gaussian_ic(
    Real Lx, Real Ly, std::size_t nx, std::size_t ny, Real amplitude, 
    std::optional<Real> sigma, std::size_t image_radius_x, std::size_t image_radius_y, const ValidationConfig& validation){

    validate_grid_spec_2d(Lx, Ly, nx, ny, validation);

    const Real Lmin = std::min(Lx, Ly);

    const Real sigma_value = sigma.value_or(Real{0.10} * Lmin);

    if(!std::isfinite(sigma_value)) throw std::invalid_argument("make_gaussian_ic: sigma must be finite");

    if(sigma_value <= Real{0.0}) throw std::invalid_argument("make_gaussian_ic: sigma must be positive");

    if(sigma_value > Real{0.25} * Lmin) throw std::invalid_argument("make_gaussian_ic: sigma is too large for the periodic domain");

    if(!std::isfinite(amplitude)) throw std::invalid_argument("make_gaussian_ic: amplitude must be finite");

    if(image_radius_x > 4 || image_radius_y > 4) throw std::invalid_argument("make_gaussian_ic: image radius too large");

    Real negLx = -Lx / Real{2};
    Real negLy = -Ly / Real{2};
    Real sigma_sq = sigma_value * sigma_value;
    std::ptrdiff_t irx = static_cast<std::ptrdiff_t>(image_radius_x);
    std::ptrdiff_t iry = static_cast<std::ptrdiff_t>(image_radius_y);

    Real dx = Lx / static_cast<Real>(nx);
    Real dy = Ly / static_cast<Real>(ny);

    Grid2D<Real> gaussian_grid(nx, ny);

    for(std::size_t i = 0; i < nx; ++i){

        Real x_i = negLx + static_cast<Real>(i) * dx;

        for(std::size_t j = 0; j < ny; ++j){

            Real y_j = negLy + static_cast<Real>(j) * dy;

            Real value = 0;

            for(std::ptrdiff_t m = -irx; m < irx + 1; ++m){
                
                Real x_part = (x_i + static_cast<Real>(m) * Lx) * (x_i + static_cast<Real>(m) * Lx) / sigma_sq;

                for(std::ptrdiff_t n = -iry; n < iry + 1; ++n){

                    Real y_part = (y_j + static_cast<Real>(n) * Ly) * (y_j + static_cast<Real>(n) * Ly) / sigma_sq;

                    value += std::exp(-(x_part + y_part));
                }
            }

            gaussian_grid(i,j) = amplitude * value;
        }
    }


    return gaussian_grid;
}


Grid2D<Real> make_hot_square_ic(
    Real Lx, Real Ly, std::size_t nx, std::size_t ny, Real amplitude, 
    std::optional<Real> width_x, std::optional<Real> width_y, 
    std::optional<Real> smooth_width_x, std::optional<Real> smooth_width_y, 
    const ValidationConfig& validation){

    validate_grid_spec_2d(Lx, Ly, nx, ny, validation);

    Real dx = Lx / static_cast<Real>(nx);
    Real dy = Ly / static_cast<Real>(ny);

    const Real wx = width_x.value_or(Real{0.20} * Lx);
    const Real wy = width_y.value_or(Real{0.20} * Ly);

    if(!std::isfinite(wx) || !std::isfinite(wy)) throw std::invalid_argument("make_hot_square_ic: widths must be finite");

    if(wx <= Real{0} || wy <= Real{0}) throw std::invalid_argument("make_hot_square_ic: widths must be positive");

    if(wx < dx || wy < dy) throw std::invalid_argument("make_hot_square_ic: widths must be at least one grid spacing");

    if(wx > Real{0.5} * Lx || wy > Real{0.5} * Ly) throw std::invalid_argument("make_hot_square_ic: widths are too large for a localized hot square");

    if(!std::isfinite(amplitude)) throw std::invalid_argument("make_hot_square_ic: amplitude must be finite");

    const Real default_smooth_width_x = std::min(Real{3} * dx, Real{0.10} * wx);
    const Real default_smooth_width_y = std::min(Real{3} * dy, Real{0.10} * wy);

    const Real smooth_width_x_value = smooth_width_x.value_or(default_smooth_width_x);
    const Real smooth_width_y_value = smooth_width_y.value_or(default_smooth_width_y);

    if(!std::isfinite(smooth_width_x_value) || !std::isfinite(smooth_width_y_value)) throw std::invalid_argument("make_hot_square_ic: smoothing widths must be finite");

    if(smooth_width_x_value <= 0 || smooth_width_y_value <= 0) throw std::invalid_argument("make_hot_square_ic: smoothing widths must be positive");

    if(smooth_width_x_value > Real{0.25} * wx || smooth_width_y_value > Real{0.25} * wy ) throw std::invalid_argument("make_hot_square_ic: smoothing widths are too large relative to square width");

    Real negLx = -Lx / Real{2};
    Real negLy = -Ly / Real{2};

    Real half_wx = wx / Real{2};
    Real half_wy = wy / Real{2};

    Grid2D<Real> hot_square(nx, ny, Real{0});

    for(std::size_t i = 0; i < nx; ++i){

        Real x_i = negLx + static_cast<Real>(i) * dx;

        Real x_i_plus = (x_i + half_wx) / smooth_width_x_value;
        Real x_i_min = (x_i - half_wx) / smooth_width_x_value;

        Real x_part = Real{0.5} * (std::tanh(x_i_plus) - std::tanh(x_i_min));

        for(std::size_t j = 0; j < ny; ++j){

            Real y_j = negLy + static_cast<Real>(j) * dy;

            Real y_j_plus = (y_j + half_wy) / smooth_width_y_value;
            Real y_j_min = (y_j - half_wy) / smooth_width_y_value;

            Real y_part = Real{0.5} * (std::tanh(y_j_plus) - std::tanh(y_j_min));

            hot_square(i, j) = amplitude * x_part * y_part;
        }
    }

    return hot_square;
}


Grid2D<Real> make_constant_ic(
    Real Lx, Real Ly, std::size_t nx, std::size_t ny, Real T0, 
    const ValidationConfig& validation){

    validate_grid_spec_2d(Lx, Ly, nx, ny, validation);

    if(!std::isfinite(T0)) throw std::invalid_argument("make_constant_ic: temperature must be finite");

    Grid2D<Real> constant_grid(nx, ny, T0);

    return constant_grid;
}


Grid2D<Real> make_custom_single_fourier_mode_ic(
    Real Lx, Real Ly, std::size_t nx, std::size_t ny, std::ptrdiff_t kx, std::ptrdiff_t ky, 
    Real amplitude, Real phase, const ValidationConfig& validation){

    validate_grid_spec_2d(Lx, Ly, nx, ny, validation);

    Real dx = Lx / static_cast<Real>(nx);
    Real dy = Ly / static_cast<Real>(ny);

    const std::ptrdiff_t kx_max = static_cast<std::ptrdiff_t>(nx / 2);
    const std::ptrdiff_t ky_max = static_cast<std::ptrdiff_t>(ny / 2);

    if(kx < -kx_max || kx > kx_max || ky < -ky_max || ky > ky_max){
        throw std::out_of_range("make_custom_single_fourier_mode_ic: mode indices exceed Nyquist limit");
    }

    if(!std::isfinite(amplitude)) throw std::invalid_argument("make_custom_single_fourier_mode_ic: amplitude must be finite");

    if(!std::isfinite(phase)) throw std::invalid_argument("make_custom_single_fourier_mode_ic: phase must be finite");

    Real x_coeff = Real{2} * PI * static_cast<Real>(kx) / Lx;
    Real y_coeff = Real{2} * PI * static_cast<Real>(ky) / Ly;
    Real negLx = -Lx / Real{2};
    Real negLy = -Ly / Real{2};
    
    Grid2D<Real> single_mode_grid(nx, ny);

    for(std::size_t i = 0; i < nx; ++i){

        Real wave_x = (negLx + static_cast<Real>(i) * dx) * x_coeff;

        for(std::size_t j = 0; j < ny; ++j){

            Real wave_y = (negLy + static_cast<Real>(j) * dy) * y_coeff;

            single_mode_grid(i,j) = amplitude * std::cos(wave_x + wave_y + phase);
        } 
    }

    return single_mode_grid;
}




Grid2D<Real> make_custom_multi_fourier_mode_ic(
    Real Lx, Real Ly, std::size_t nx, std::size_t ny, const std::vector<FourierMode2D>& modes, 
    const ValidationConfig& validation){

    validate_grid_spec_2d(Lx, Ly, nx, ny, validation);
    
    if(modes.empty()) throw std::invalid_argument("make_custom_multi_fourier_mode_ic: must contain at least one mode");

    Real dx = Lx / static_cast<Real>(nx);
    Real dy = Ly / static_cast<Real>(ny);

    const std::ptrdiff_t kx_max = static_cast<std::ptrdiff_t>(nx / 2);
    const std::ptrdiff_t ky_max = static_cast<std::ptrdiff_t>(ny / 2);

    for(std::size_t k = 0; k < modes.size(); ++k){

        if(!std::isfinite(modes[k].amplitude)) throw std::invalid_argument("make_custom_multi_fourier_mode_ic: amplitude must be finite");

        if(!std::isfinite(modes[k].phase)) throw std::invalid_argument("make_custom_multi_fourier_mode_ic: phase must be finite");

        if(modes[k].kx < -kx_max || modes[k].kx > kx_max || modes[k].ky < -ky_max || modes[k].ky > ky_max){
            throw std::out_of_range("make_custom_multi_fourier_mode_ic: mode indices exceed Nyquist limit");
        }
    }

    Real x_coeff = Real{2} * PI / Lx;
    Real y_coeff = Real{2} * PI / Ly;
    Real negLx = -Lx / Real{2};
    Real negLy = -Ly / Real{2};


    Grid2D<Real> multi_mode_grid(nx, ny);

    for(std::size_t k = 0; k < modes.size(); ++k){

        for(std::size_t i = 0; i < nx; ++i){

            Real wave_x = (negLx + static_cast<Real>(i) * dx) * x_coeff * static_cast<Real>(modes[k].kx);

            for(std::size_t j = 0; j < ny; ++j){

                Real wave_y = (negLy + static_cast<Real>(j) * dy) * y_coeff * static_cast<Real>(modes[k].ky);

                multi_mode_grid(i,j) += modes[k].amplitude * std::cos(wave_x + wave_y + modes[k].phase);
            }
        }
    }

    return multi_mode_grid;
}