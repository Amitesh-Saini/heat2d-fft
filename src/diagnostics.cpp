// diagnostics.cpp7
// Responsibility:
//   Implement diagnostic calculations for heat-equation snapshots.
//
//   This file should compute scalar summary quantities from Grid2D<Real>
//   snapshots, such as:
//     - mean temperature,
//     - L2 norm,
//     - minimum value,
//     - maximum value.
//
//   These diagnostics can be written to diagnostics.csv by the snapshot writer
//   and later plotted by Python scripts.
//
//   First versions can compute diagnostics directly from physical-space
//   snapshots. Later versions may compute some diagnostics spectrally using
//   Parseval's theorem or the zero Fourier mode.

#include "diagnostics.hpp"

const Real abs_tol = 1e-10;
const Real rel_tol = 1e-10;


Real compute_mean(const Grid2D<Real>& field){

    Real sum = Real{0};

    for(const auto& value : field.raw()) sum += value; 
    
    return sum / static_cast<Real>(field.size());

}


Real compute_l2_norm(const Grid2D<Real>& field, Real Lx, Real Ly){

    Real dx = Lx / static_cast<Real>(field.nx());
    Real dy = Ly / static_cast<Real>(field.ny());
    
    Real sum = Real{0};

    for(const auto& value : field.raw()) sum += value*value;

    return std::sqrt(dx * dy * sum);
}

Real compute_min(const Grid2D<Real>& field){

    const auto& raw_field = field.raw();

    if(raw_field.empty()) throw std::invalid_argument("compute_min: empty field");

    Real min = raw_field[0];

    for(const auto& value : raw_field){

        if(value < min) min = value;
    }

    return min;

}

Real compute_max(const Grid2D<Real>& field){

    const auto& raw_field = field.raw();

    if(raw_field.empty()) throw std::invalid_argument("compute_max: empty field");


    Real max = raw_field[0];

    for(const auto& value : raw_field){

        if(value > max) max = value;
    }

    return max;
}


SnapshotDiagnostics compute_snapshot_diagnostics(const Grid2D<Real>& field, Real time, Real Lx, Real Ly){

    SnapshotDiagnostics snapshot_diagnostics;

    snapshot_diagnostics.time = time;
    snapshot_diagnostics.max_value = compute_max(field);
    snapshot_diagnostics.l2_norm = compute_l2_norm(field, Lx, Ly);
    snapshot_diagnostics.mean = compute_mean(field);
    snapshot_diagnostics.min_value = compute_min(field);

    return snapshot_diagnostics;
}

Grid2D<Real> make_exact_fourier_mode_solution(
    Real Lx, Real Ly, std::size_t nx, std::size_t ny, const std::vector<FourierMode2D>& modes, Real alpha, Real time){

    Grid2D<Real> u(nx, ny);
    
    for(std::size_t i = 0; i < nx; ++i){

        const Real x = -Lx/Real{2} + static_cast<Real>(i) * Lx / static_cast<Real>(nx);

        for(std::size_t j = 0; j < ny; ++j){

            const Real y = -Ly/Real{2} + static_cast<Real>(j) * Ly / static_cast<Real>(ny);
            
            Real value = Real{0}; 

            for(const auto& m : modes){

                Real kxr = static_cast<Real>(m.kx);
                Real kyr = static_cast<Real>(m.ky);

                Real k2 = Real{4} * PI * PI * ((kxr * kxr / (Lx * Lx)) + (kyr * kyr / (Ly * Ly)));

                value += (m.amplitude * std::exp(-alpha * k2 * time) 
                * std::cos(Real{2} * PI * ((kxr * x / Lx) + (kyr * y / Ly)) + m.phase));
            }

            u(i,j) = value;
        }
    }

    return u;
}


Real compute_relative_l2_error(
    const Grid2D<Real>& numerical, const Grid2D<Real>& reference, Real Lx, Real Ly){

    if(numerical.nx() != reference.nx() || numerical.ny() != reference.ny()) throw std::invalid_argument("compute_relative_l2_error: Grids Do not match shape");

    Real numerator = 0.0;
    Real denominator = 0.0;

    const auto& reference_data = reference.raw();
    const auto& numerical_data = numerical.raw();

    for(std::size_t k = 0; k < reference_data.size(); ++k){

        numerator += (reference_data[k] - numerical_data[k]) * (reference_data[k] - numerical_data[k]);
        denominator += (reference_data[k])* (reference_data[k]);
    }

    const Real l2_floor = Real{1e-30};

    if(denominator <= l2_floor) return std::sqrt(numerator); // return absolute l2 error

    return std::sqrt(numerator/denominator);
}
