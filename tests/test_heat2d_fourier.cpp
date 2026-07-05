// test_heat2d_fourier.cpp
// Responsibility:
//   Deterministic tests for the 2D Fourier spectral heat-equation solver.
// What to do here:
//   - Compare solver snapshots against exact analytical Fourier-mode solutions.
//   - Check physical invariants such as constant preservation, mean conservation,
//     and nonincreasing L2 energy.
//   - Check solver validation for invalid configs and mismatched initial grids.

#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <stdexcept>
#include <optional>
#include <iomanip>

#include "heat2d_fourier.hpp"
#include "initial_conditions.hpp"
#include "grid2d.hpp"
#include "types.hpp"
#include "1D_test_utils.hpp"
#include "diagnostics.hpp"


Grid2D<Real> make_exact_single_fourier_solution_grid(
    Real Lx, Real Ly, std::size_t nx, std::size_t ny, std::ptrdiff_t kx, std::ptrdiff_t ky, Real amplitude,
    Real phase, Real alpha, Real time){

    Grid2D<Real> exact(nx, ny);

    const Real dx = Lx / static_cast<Real>(nx);
    const Real dy = Ly / static_cast<Real>(ny);

    const Real x_min = -Lx / Real{2};
    const Real y_min = -Ly / Real{2};

    const Real Kx = Real{2} * PI * static_cast<Real>(kx) / Lx;
    const Real Ky = Real{2} * PI * static_cast<Real>(ky) / Ly;

    const Real lambda = Kx * Kx + Ky * Ky;
    const Real decay = std::exp(-alpha * lambda * time);

    for(std::size_t i = 0; i < nx; ++i) {
        const Real x = x_min + static_cast<Real>(i) * dx;

        for(std::size_t j = 0; j < ny; ++j) {
            const Real y = y_min + static_cast<Real>(j) * dy;

            exact(i, j) = amplitude * std::cos(Kx * x + Ky * y + phase) * decay;
        }
    }

    return exact;
}

bool approx_equal_real_grid(
    const Grid2D<Real>& expected, const Grid2D<Real>& actual, Real abs_tol, Real rel_tol){

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
    const std::string& test_name, const Grid2D<Real>& expected, const Grid2D<Real>& actual, Real abs_tol, Real rel_tol){
        
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
                std::cout << "  expected = " << expected(i, j) << "\n";
                std::cout << "  actual   = " << actual(i, j) << "\n";
                std::cout << "  abs err  = " << std::abs(expected(i, j) - actual(i, j)) << "\n";
                return;
            }
        }
    }
}


Real grid_sum(const Grid2D<Real>& grid) {
    Real sum = Real{0};

    for(std::size_t i = 0; i < grid.nx(); ++i) {
        for(std::size_t j = 0; j < grid.ny(); ++j) {
            sum += grid(i, j);
        }
    }

    return sum;
}


Real grid_mean(const Grid2D<Real>& grid) {
    return grid_sum(grid) / static_cast<Real>(grid.nx() * grid.ny());
}


Real grid_l2_energy(const Grid2D<Real>& grid) {
    Real energy = Real{0};

    for(std::size_t i = 0; i < grid.nx(); ++i) {
        for(std::size_t j = 0; j < grid.ny(); ++j) {
            energy += grid(i, j) * grid(i, j);
        }
    }

    return energy;
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

    const Real abs_tol = Real{1e-9};
    const Real rel_tol = Real{1e-9};

    std::cout << "=== Running 2D Fourier heat solver tests ===\n\n";


    // ------------------------------------------------------------
    // Constant solution preservation
    // For periodic heat equation, a constant field has zero Laplacian
    // and should remain exactly constant for all output times.
    // ------------------------------------------------------------

    {
        const std::string test_name = "constant_solution_preserved";

        Heat2DConfig cfg;
        cfg.nx = 8;
        cfg.ny = 8;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.75};
        cfg.output_times = {Real{0}, Real{0.1}, Real{0.5}, Real{1.0}};

        const Real T0 = Real{3.25};

        Grid2D<Real> u0 = make_constant_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, T0, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(const auto& snapshot : snapshots) {
            Grid2D<Real> expected(cfg.nx, cfg.ny, T0);

            if(!approx_equal_real_grid(expected, snapshot, abs_tol, rel_tol)) {
                ok = false;
                print_real_grid_failure_report(test_name, expected, snapshot, abs_tol, rel_tol);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Single Fourier mode exact decay: square domain
    // Checks full solver pipeline against exact analytical solution.
    // ------------------------------------------------------------

    {
        const std::string test_name = "single_fourier_exact_decay_square_domain";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.20};
        cfg.output_times = {Real{0}, Real{0.03}, Real{0.10}};

        const std::ptrdiff_t kx = 1;
        const std::ptrdiff_t ky = 2;
        const Real amplitude = Real{0.75};
        const Real phase = Real{0.30};

        Grid2D<Real> u0 = make_custom_single_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, kx, ky, amplitude, phase, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(std::size_t t_index = 0; t_index < cfg.output_times.size(); ++t_index) {
            Grid2D<Real> expected = make_exact_single_fourier_solution_grid(
                cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
                kx, ky, amplitude, phase,
                cfg.alpha, cfg.output_times[t_index]
            );

            if(!approx_equal_real_grid(expected, snapshots[t_index], abs_tol, rel_tol)) {
                ok = false;
                print_real_grid_failure_report(test_name, expected, snapshots[t_index], abs_tol, rel_tol);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Single Fourier mode exact decay: rectangular domain/grid
    // This catches bugs where Lx/Ly, nx/ny, or axis scaling are mixed up.
    // ------------------------------------------------------------

    {
        const std::string test_name = "single_fourier_exact_decay_rectangular_domain";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 8;
        cfg.Lx = Real{3};
        cfg.Ly = Real{1.5};
        cfg.alpha = Real{0.12};
        cfg.output_times = {Real{0}, Real{0.02}, Real{0.15}};

        const std::ptrdiff_t kx = -3;
        const std::ptrdiff_t ky = 2;
        const Real amplitude = Real{1.20};
        const Real phase = Real{-0.40};

        Grid2D<Real> u0 = make_custom_single_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, kx, ky, amplitude, phase, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(std::size_t t_index = 0; t_index < cfg.output_times.size(); ++t_index) {
            Grid2D<Real> expected = make_exact_single_fourier_solution_grid(
                cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
                kx, ky, amplitude, phase,
                cfg.alpha, cfg.output_times[t_index]
            );

            if(!approx_equal_real_grid(expected, snapshots[t_index], abs_tol, rel_tol)) {
                ok = false;
                print_real_grid_failure_report(test_name, expected, snapshots[t_index], abs_tol, rel_tol);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Multi-mode exact decay
    // Each Fourier mode should decay independently with its own Kx^2+Ky^2.
    // ------------------------------------------------------------

    {
        const std::string test_name = "multi_fourier_exact_decay";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2.5};
        cfg.Ly = Real{3.0};
        cfg.alpha = Real{0.07};
        cfg.output_times = {Real{0}, Real{0.04}, Real{0.12}};

        std::vector<FourierMode2D> modes = {
            {std::ptrdiff_t{1},  std::ptrdiff_t{1},  Real{1.0},  Real{0.0}},
            {std::ptrdiff_t{-2}, std::ptrdiff_t{1},  Real{0.35}, -PI / Real{2}},
            {std::ptrdiff_t{0},  std::ptrdiff_t{-3}, Real{-0.2}, PI / Real{3}}
        };

        Grid2D<Real> u0 = make_custom_multi_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, modes, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(std::size_t t_index = 0; t_index < cfg.output_times.size(); ++t_index) {
            Grid2D<Real> expected = make_exact_fourier_mode_solution(
                cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
                modes,
                cfg.alpha, cfg.output_times[t_index]
            );

            if(!approx_equal_real_grid(expected, snapshots[t_index], abs_tol, rel_tol)) {
                ok = false;
                print_real_grid_failure_report(test_name, expected, snapshots[t_index], abs_tol, rel_tol);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // t = 0 snapshot returns initial condition
    // This uses a Gaussian IC to test a non-Fourier-manufactured field.
    // ------------------------------------------------------------

    {
        const std::string test_name = "zero_time_returns_initial_condition_gaussian";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.4};
        cfg.output_times = {Real{0}};

        Grid2D<Real> u0 = make_gaussian_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
            Real{1.0},
            std::nullopt,
            1,
            1,
            ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == 1 &&
                  approx_equal_real_grid(u0, snapshots[0], abs_tol, rel_tol);

        if(!ok) {
            if(!snapshots.empty()) {
                print_real_grid_failure_report(test_name, u0, snapshots[0], abs_tol, rel_tol);
            }
            else {
                std::cout << "[FAIL] " << test_name << ": no snapshot returned\n";
            }

            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Mean conservation
    // Periodic heat equation conserves total heat / spatial average.
    // ------------------------------------------------------------

    {
        const std::string test_name = "mean_conservation_gaussian";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.15};
        cfg.output_times = {Real{0}, Real{0.05}, Real{0.25}, Real{0.75}};

        Grid2D<Real> u0 = make_gaussian_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
            Real{1.0},
            std::nullopt,
            1,
            1,
            ValidationConfig{}
        );

        const Real initial_mean = grid_mean(u0);

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(const auto& snapshot : snapshots) {
            const Real snapshot_mean = grid_mean(snapshot);

            if(!approx_equal_real(initial_mean, snapshot_mean, Real{1e-9}, Real{1e-9})) {
                ok = false;
                std::cout << "[FAIL] " << test_name << "\n";
                print_scalar_failure_report(test_name, initial_mean, snapshot_mean);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // L2 energy should not increase under heat evolution.
    // ------------------------------------------------------------

    {
        const std::string test_name = "energy_nonincreasing_smooth_hot_square";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.10};
        cfg.output_times = {Real{0}, Real{0.05}, Real{0.20}, Real{0.50}};

        Grid2D<Real> u0 = make_hot_square_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
            Real{1.0},
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        Real previous_energy = std::numeric_limits<Real>::infinity();

        for(const auto& snapshot : snapshots) {
            const Real energy = grid_l2_energy(snapshot);

            if(energy > previous_energy + Real{1e-9}) {
                ok = false;
                std::cout << "[FAIL] " << test_name
                          << ": energy increased from "
                          << previous_energy << " to " << energy << "\n";
                break;
            }

            previous_energy = energy;
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Invalid config tests
    // ------------------------------------------------------------

    {
        const std::string test_name = "invalid_config_non_power_of_two_nx";

        if(!expect_throw([] {
            Heat2DConfig cfg;
            cfg.nx = 12;
            cfg.ny = 8;
            cfg.Lx = Real{2};
            cfg.Ly = Real{2};
            cfg.alpha = Real{1};
            cfg.output_times = {Real{0}, Real{0.1}};

            Heat2DFourierSolver solver(cfg);
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    {
        const std::string test_name = "invalid_config_negative_domain_length";

        if(!expect_throw([] {
            Heat2DConfig cfg;
            cfg.nx = 8;
            cfg.ny = 8;
            cfg.Lx = Real{-2};
            cfg.Ly = Real{2};
            cfg.alpha = Real{1};
            cfg.output_times = {Real{0}, Real{0.1}};

            Heat2DFourierSolver solver(cfg);
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    {
        const std::string test_name = "invalid_config_nonpositive_alpha";

        if(!expect_throw([] {
            Heat2DConfig cfg;
            cfg.nx = 8;
            cfg.ny = 8;
            cfg.Lx = Real{2};
            cfg.Ly = Real{2};
            cfg.alpha = Real{0};
            cfg.output_times = {Real{0}, Real{0.1}};

            Heat2DFourierSolver solver(cfg);
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    {
        const std::string test_name = "invalid_config_unsorted_output_times";

        if(!expect_throw([] {
            Heat2DConfig cfg;
            cfg.nx = 8;
            cfg.ny = 8;
            cfg.Lx = Real{2};
            cfg.Ly = Real{2};
            cfg.alpha = Real{1};
            cfg.output_times = {Real{0}, Real{0.2}, Real{0.1}};

            Heat2DFourierSolver solver(cfg);
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    {
        const std::string test_name = "initial_condition_shape_mismatch_throws";

        if(!expect_throw([] {
            Heat2DConfig cfg;
            cfg.nx = 8;
            cfg.ny = 8;
            cfg.Lx = Real{2};
            cfg.Ly = Real{2};
            cfg.alpha = Real{1};
            cfg.output_times = {Real{0}, Real{0.1}};

            Heat2DFourierSolver solver(cfg);

            Grid2D<Real> wrong_shape(4, 8, Real{1});
            solver.set_initial_condition(wrong_shape);
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }


    {
        const std::string test_name = "solve_without_initial_condition_throws";

        if(!expect_throw([] {
            Heat2DConfig cfg;
            cfg.nx = 8;
            cfg.ny = 8;
            cfg.Lx = Real{2};
            cfg.Ly = Real{2};
            cfg.alpha = Real{1};
            cfg.output_times = {Real{0}, Real{0.1}};

            Heat2DFourierSolver solver(cfg);
            (void)solver.solve();
        })) {
            std::cout << "[FAIL] " << test_name << ": expected exception but none was thrown\n";
            failed_tests.push_back(test_name);
        }
    }

    // ------------------------------------------------------------
    // Nyquist-adjacent exact decay
    // kx = nx/2, ky = ny/2 are the highest representable wavenumbers.
    // The decay factor is maximal here, and FFT frequency-ordering
    // bugs (off-by-one at the Nyquist bin, wrong negative-frequency
    // handling) are most likely to manifest at this mode.
    // ------------------------------------------------------------

    {
        const std::string test_name = "nyquist_mode_exact_decay";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.10};
        cfg.output_times = {Real{0}, Real{0.01}, Real{0.05}};

        const std::ptrdiff_t kx = static_cast<std::ptrdiff_t>(cfg.nx / 2);
        const std::ptrdiff_t ky = static_cast<std::ptrdiff_t>(cfg.ny / 2);
        const Real amplitude = Real{0.50};
        const Real phase = Real{0.0};

        Grid2D<Real> u0 = make_custom_single_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, kx, ky, amplitude, phase, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(std::size_t t_index = 0; t_index < cfg.output_times.size(); ++t_index) {
            Grid2D<Real> expected = make_exact_single_fourier_solution_grid(
                cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
                kx, ky, amplitude, phase,
                cfg.alpha, cfg.output_times[t_index]
            );

            if(!approx_equal_real_grid(expected, snapshots[t_index], abs_tol, rel_tol)) {
                ok = false;
                print_real_grid_failure_report(test_name, expected, snapshots[t_index], abs_tol, rel_tol);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Pure x-axis mode exact decay (ky = 0)
    // Isolates the x wavenumber path. With ky = 0 the physical
    // angular wavenumber is Kx only, so the decay factor is
    // exp(-alpha * Kx^2 * t). A bug in the y wavenumber build
    // cannot compensate for a bug in the x path here.
    // ------------------------------------------------------------

    {
        const std::string test_name = "pure_x_mode_exact_decay";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.30};
        cfg.output_times = {Real{0}, Real{0.02}, Real{0.10}};

        const std::ptrdiff_t kx = 3;
        const std::ptrdiff_t ky = 0;
        const Real amplitude = Real{1.25};
        const Real phase = Real{0.60};

        Grid2D<Real> u0 = make_custom_single_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, kx, ky, amplitude, phase, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(std::size_t t_index = 0; t_index < cfg.output_times.size(); ++t_index) {
            Grid2D<Real> expected = make_exact_single_fourier_solution_grid(
                cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
                kx, ky, amplitude, phase,
                cfg.alpha, cfg.output_times[t_index]
            );

            if(!approx_equal_real_grid(expected, snapshots[t_index], abs_tol, rel_tol)) {
                ok = false;
                print_real_grid_failure_report(test_name, expected, snapshots[t_index], abs_tol, rel_tol);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Pure y-axis mode exact decay (kx = 0)
    // Isolates the y wavenumber path. Decay factor is
    // exp(-alpha * Ky^2 * t). Together with the pure x test this
    // covers both axes independently before testing diagonal modes.
    // ------------------------------------------------------------

    {
        const std::string test_name = "pure_y_mode_exact_decay";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.30};
        cfg.output_times = {Real{0}, Real{0.02}, Real{0.10}};

        const std::ptrdiff_t kx = 0;
        const std::ptrdiff_t ky = 3;
        const Real amplitude = Real{0.85};
        const Real phase = Real{-0.50};

        Grid2D<Real> u0 = make_custom_single_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, kx, ky, amplitude, phase, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(std::size_t t_index = 0; t_index < cfg.output_times.size(); ++t_index) {
            Grid2D<Real> expected = make_exact_single_fourier_solution_grid(
                cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
                kx, ky, amplitude, phase,
                cfg.alpha, cfg.output_times[t_index]
            );

            if(!approx_equal_real_grid(expected, snapshots[t_index], abs_tol, rel_tol)) {
                ok = false;
                print_real_grid_failure_report(test_name, expected, snapshots[t_index], abs_tol, rel_tol);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Asymmetric domain exact decay: dx != dy, Lx/nx != Ly/ny
    // The existing rectangular test has Lx/nx == Ly/ny (both axes
    // share the same grid spacing), so an axis-swap bug in the
    // wavenumber builder could go undetected. This test uses
    // nx=32, ny=8, Lx=2, Ly=4 so dx=1/16 and dy=1/2 are clearly
    // different, and the physical wavenumber scales differ too.
    // ------------------------------------------------------------

    {
        const std::string test_name = "asymmetric_domain_unequal_spacing_exact_decay";

        Heat2DConfig cfg;
        cfg.nx = 32;
        cfg.ny = 8;
        cfg.Lx = Real{2};
        cfg.Ly = Real{4};
        cfg.alpha = Real{0.08};
        cfg.output_times = {Real{0}, Real{0.05}, Real{0.20}};

        const std::ptrdiff_t kx = 2;
        const std::ptrdiff_t ky = 1;
        const Real amplitude = Real{1.10};
        const Real phase = Real{0.80};

        Grid2D<Real> u0 = make_custom_single_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, kx, ky, amplitude, phase, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(std::size_t t_index = 0; t_index < cfg.output_times.size(); ++t_index) {
            Grid2D<Real> expected = make_exact_single_fourier_solution_grid(
                cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
                kx, ky, amplitude, phase,
                cfg.alpha, cfg.output_times[t_index]
            );

            if(!approx_equal_real_grid(expected, snapshots[t_index], abs_tol, rel_tol)) {
                ok = false;
                print_real_grid_failure_report(test_name, expected, snapshots[t_index], abs_tol, rel_tol);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Mean conservation: hot square IC
    // Verifies that the periodic heat equation conserves the spatial
    // mean for a non-Fourier-manufactured, smooth localized IC.
    // ------------------------------------------------------------

    {
        const std::string test_name = "mean_conservation_hot_square";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.20};
        cfg.output_times = {Real{0}, Real{0.05}, Real{0.25}, Real{0.75}};

        Grid2D<Real> u0 = make_hot_square_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
            Real{1.0},
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            ValidationConfig{}
        );

        const Real initial_mean = grid_mean(u0);

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(const auto& snapshot : snapshots) {
            const Real snapshot_mean = grid_mean(snapshot);

            if(!approx_equal_real(initial_mean, snapshot_mean, Real{1e-9}, Real{1e-9})) {
                ok = false;
                std::cout << "[FAIL] " << test_name << "\n";
                print_scalar_failure_report(test_name, initial_mean, snapshot_mean);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Mean conservation: multi-mode IC
    // A sum of cosine modes has zero mean (each cosine integrates to
    // zero over the periodic domain). That zero mean must be preserved
    // for all output times.
    // ------------------------------------------------------------

    {
        const std::string test_name = "mean_conservation_multi_mode";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.10};
        cfg.output_times = {Real{0}, Real{0.05}, Real{0.25}, Real{0.75}};

        std::vector<FourierMode2D> modes = {
            {std::ptrdiff_t{1},  std::ptrdiff_t{1},  Real{1.0},  Real{0.0}},
            {std::ptrdiff_t{-2}, std::ptrdiff_t{1},  Real{0.35}, -PI / Real{2}},
            {std::ptrdiff_t{0},  std::ptrdiff_t{-3}, Real{-0.2}, PI / Real{3}}
        };

        Grid2D<Real> u0 = make_custom_multi_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, modes, ValidationConfig{}
        );

        const Real initial_mean = grid_mean(u0);

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(const auto& snapshot : snapshots) {
            const Real snapshot_mean = grid_mean(snapshot);

            if(!approx_equal_real(initial_mean, snapshot_mean, Real{1e-9}, Real{1e-9})) {
                ok = false;
                std::cout << "[FAIL] " << test_name << "\n";
                print_scalar_failure_report(test_name, initial_mean, snapshot_mean);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // L2 energy nonincreasing: Gaussian IC
    // Energy must be nonincreasing at every output time. Uses a
    // different IC type than the existing hot-square energy test.
    // ------------------------------------------------------------

    {
        const std::string test_name = "energy_nonincreasing_gaussian";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.15};
        cfg.output_times = {Real{0}, Real{0.05}, Real{0.20}, Real{0.50}};

        Grid2D<Real> u0 = make_gaussian_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
            Real{1.0},
            std::nullopt,
            1,
            1,
            ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        Real previous_energy = std::numeric_limits<Real>::infinity();

        for(const auto& snapshot : snapshots) {
            const Real energy = grid_l2_energy(snapshot);

            if(energy > previous_energy + Real{1e-9}) {
                ok = false;
                std::cout << "[FAIL] " << test_name
                          << ": energy increased from "
                          << previous_energy << " to " << energy << "\n";
                break;
            }

            previous_energy = energy;
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // L2 energy nonincreasing: multi-mode IC
    // Same invariant check using a multi-mode IC.
    // ------------------------------------------------------------

    {
        const std::string test_name = "energy_nonincreasing_multi_mode";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.10};
        cfg.output_times = {Real{0}, Real{0.05}, Real{0.20}, Real{0.50}};

        std::vector<FourierMode2D> modes = {
            {std::ptrdiff_t{1},  std::ptrdiff_t{2},  Real{1.00}, Real{0.0}},
            {std::ptrdiff_t{-1}, std::ptrdiff_t{1},  Real{0.50}, PI / Real{4}},
            {std::ptrdiff_t{3},  std::ptrdiff_t{-2}, Real{0.25}, Real{0.0}}
        };

        Grid2D<Real> u0 = make_custom_multi_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, modes, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        Real previous_energy = std::numeric_limits<Real>::infinity();

        for(const auto& snapshot : snapshots) {
            const Real energy = grid_l2_energy(snapshot);

            if(energy > previous_energy + Real{1e-9}) {
                ok = false;
                std::cout << "[FAIL] " << test_name
                          << ": energy increased from "
                          << previous_energy << " to " << energy << "\n";
                break;
            }

            previous_energy = energy;
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Constant IC: energy is constant (not just nonincreasing)
    // A constant field has zero Laplacian so it does not diffuse.
    // Every Fourier coefficient except the DC bin is zero, so the
    // total L2 energy must be exactly preserved, not just bounded.
    // ------------------------------------------------------------

    {
        const std::string test_name = "energy_constant_for_constant_ic";

        Heat2DConfig cfg;
        cfg.nx = 8;
        cfg.ny = 8;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.50};
        cfg.output_times = {Real{0}, Real{0.10}, Real{0.50}, Real{1.00}};

        const Real T0 = Real{2.50};

        Grid2D<Real> u0 = make_constant_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, T0, ValidationConfig{}
        );

        const Real initial_energy = grid_l2_energy(u0);

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(const auto& snapshot : snapshots) {
            const Real energy = grid_l2_energy(snapshot);

            if(!approx_equal_real(initial_energy, energy, Real{1e-9}, Real{1e-9})) {
                ok = false;
                std::cout << "[FAIL] " << test_name << "\n";
                print_scalar_failure_report(test_name, initial_energy, energy);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Negative-mean constant IC preserved
    // The existing constant test uses T0 = 3.25 (positive). This
    // checks that a negative constant is also preserved exactly,
    // ruling out any accidental sign clipping or abs() in the solver.
    // ------------------------------------------------------------

    {
        const std::string test_name = "negative_constant_ic_preserved";

        Heat2DConfig cfg;
        cfg.nx = 8;
        cfg.ny = 8;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.75};
        cfg.output_times = {Real{0}, Real{0.1}, Real{0.5}, Real{1.0}};

        const Real T0 = Real{-2.75};

        Grid2D<Real> u0 = make_constant_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, T0, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(const auto& snapshot : snapshots) {
            Grid2D<Real> expected(cfg.nx, cfg.ny, T0);

            if(!approx_equal_real_grid(expected, snapshot, abs_tol, rel_tol)) {
                ok = false;
                print_real_grid_failure_report(test_name, expected, snapshot, abs_tol, rel_tol);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Long-time decay to near-zero
    // A single nonzero Fourier mode with large alpha and large t.
    // The exact solution is amplitude * cos(...) * exp(-alpha*K^2*t).
    // With alpha=1, kx=ky=2, Lx=Ly=2, t=2 the decay factor is
    // exp(-1 * (2*pi)^2 * 2) ~ 5e-35, so the solution is effectively
    // zero to double precision. This exercises numerical stability of
    // the exp() call at large negative exponent and confirms no
    // floating-point blow-up occurs at long times.
    // ------------------------------------------------------------

    {
        const std::string test_name = "long_time_decay_to_near_zero";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{1.0};
        cfg.output_times = {Real{0}, Real{0.5}, Real{2.0}};

        const std::ptrdiff_t kx = 2;
        const std::ptrdiff_t ky = 2;
        const Real amplitude = Real{1.0};
        const Real phase = Real{0.0};

        Grid2D<Real> u0 = make_custom_single_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, kx, ky, amplitude, phase, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == cfg.output_times.size();

        for(std::size_t t_index = 0; t_index < cfg.output_times.size(); ++t_index) {
            Grid2D<Real> expected = make_exact_single_fourier_solution_grid(
                cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
                kx, ky, amplitude, phase,
                cfg.alpha, cfg.output_times[t_index]
            );

            if(!approx_equal_real_grid(expected, snapshots[t_index], abs_tol, rel_tol)) {
                ok = false;
                print_real_grid_failure_report(test_name, expected, snapshots[t_index], abs_tol, rel_tol);
                break;
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Single output time (edge case)
    // output_times with exactly one entry. Exercises the reserve()
    // and loop in solve() at minimum size.
    // ------------------------------------------------------------

    {
        const std::string test_name = "single_output_time";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.20};
        cfg.output_times = {Real{0.10}};

        const std::ptrdiff_t kx = 1;
        const std::ptrdiff_t ky = 1;
        const Real amplitude = Real{0.90};
        const Real phase = Real{0.0};

        Grid2D<Real> u0 = make_custom_single_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, kx, ky, amplitude, phase, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots = solver.solve();

        bool ok = snapshots.size() == 1;

        if(ok) {
            Grid2D<Real> expected = make_exact_single_fourier_solution_grid(
                cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
                kx, ky, amplitude, phase,
                cfg.alpha, cfg.output_times[0]
            );

            if(!approx_equal_real_grid(expected, snapshots[0], abs_tol, rel_tol)) {
                ok = false;
                print_real_grid_failure_report(test_name, expected, snapshots[0], abs_tol, rel_tol);
            }
        } else {
            std::cout << "[FAIL] " << test_name
                      << ": expected 1 snapshot, got " << snapshots.size() << "\n";
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // Idempotent double solve
    // Calling solve() twice on the same solver with the same initial
    // condition must return identical results. This catches bugs where
    // solve() mutates internal state (e.g. modifies the stored IC in
    // place rather than working on a copy).
    // ------------------------------------------------------------

    {
        const std::string test_name = "double_solve_idempotent";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.15};
        cfg.output_times = {Real{0}, Real{0.05}, Real{0.20}};

        const std::ptrdiff_t kx = 2;
        const std::ptrdiff_t ky = -1;
        const Real amplitude = Real{0.70};
        const Real phase = Real{PI / Real{3}};

        Grid2D<Real> u0 = make_custom_single_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, kx, ky, amplitude, phase, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(u0);

        std::vector<Grid2D<Real>> snapshots_first  = solver.solve();
        std::vector<Grid2D<Real>> snapshots_second = solver.solve();

        bool ok = snapshots_first.size() == snapshots_second.size();

        for(std::size_t t_index = 0; t_index < snapshots_first.size() && ok; ++t_index) {
            if(!approx_equal_real_grid(snapshots_first[t_index], snapshots_second[t_index], abs_tol, rel_tol)) {
                ok = false;
                std::cout << "[FAIL] " << test_name
                          << ": snapshots differ at output_times index " << t_index << "\n";
                print_real_grid_failure_report(
                    test_name,
                    snapshots_first[t_index], snapshots_second[t_index],
                    abs_tol, rel_tol
                );
            }
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }


    // ------------------------------------------------------------
    // set_initial_condition replacement
    // Set IC A, solve, then replace with IC B, solve again. The
    // second solve must use IC B, not IC A. This verifies that
    // set_initial_condition correctly overwrites the stored field
    // and that no stale spectral state persists between solves.
    // ------------------------------------------------------------

    {
        const std::string test_name = "set_initial_condition_replacement";

        Heat2DConfig cfg;
        cfg.nx = 16;
        cfg.ny = 16;
        cfg.Lx = Real{2};
        cfg.Ly = Real{2};
        cfg.alpha = Real{0.20};
        cfg.output_times = {Real{0.10}};

        const std::ptrdiff_t kx_a = 1;
        const std::ptrdiff_t ky_a = 0;
        const Real amplitude_a = Real{1.0};
        const Real phase_a = Real{0.0};

        const std::ptrdiff_t kx_b = 0;
        const std::ptrdiff_t ky_b = 2;
        const Real amplitude_b = Real{0.5};
        const Real phase_b = PI / Real{4};

        Grid2D<Real> u0_a = make_custom_single_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, kx_a, ky_a, amplitude_a, phase_a, ValidationConfig{}
        );

        Grid2D<Real> u0_b = make_custom_single_fourier_mode_ic(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, kx_b, ky_b, amplitude_b, phase_b, ValidationConfig{}
        );

        Heat2DFourierSolver solver(cfg);

        solver.set_initial_condition(u0_a);
        (void)solver.solve();

        solver.set_initial_condition(u0_b);
        std::vector<Grid2D<Real>> snapshots_b = solver.solve();

        Grid2D<Real> expected_b = make_exact_single_fourier_solution_grid(
            cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
            kx_b, ky_b, amplitude_b, phase_b,
            cfg.alpha, cfg.output_times[0]
        );

        bool ok = snapshots_b.size() == 1;

        if(ok) {
            if(!approx_equal_real_grid(expected_b, snapshots_b[0], abs_tol, rel_tol)) {
                ok = false;
                print_real_grid_failure_report(test_name, expected_b, snapshots_b[0], abs_tol, rel_tol);
            }
        } else {
            std::cout << "[FAIL] " << test_name
                      << ": expected 1 snapshot, got " << snapshots_b.size() << "\n";
        }

        if(!ok) {
            failed_tests.push_back(test_name);
        }
    }

    // ------------------------------------------------------------
    // Spectral convergence
    //
    // Runs the solver at N x N grids (N = 8, 16, 32, 64) with a smooth
    // Gaussian IC and checks that the L-infinity error against the exact
    // analytical solution decreases by at least a factor of 10 at each
    // doubling of N.
    //
    // Why a Gaussian IC and not a pure Fourier mode:
    //   A pure Fourier mode is represented exactly on any grid that resolves
    //   it, so it reaches machine precision immediately at any N — that is
    //   exactness, not convergence. The Gaussian is smooth but not bandlimited,
    //   so increasing N genuinely reduces the aliasing error in the sampled IC,
    //   which is the convergence signal we want to measure.
    //
    // Why t = 0 is not used:
    //   At t = 0 the solver output is an exact round-trip (FFT then IFFT)
    //   to within floating-point precision at any N, so there is no
    //   convergence signal. A moderate time gives a meaningful error measure.
    //
    // Exact solution:
    //   The PDE solution for the periodized Gaussian IC is another Gaussian
    //   with a time-broadened exponent. If the IC is exp(-r^2 / sigma^2) then
    //   at time t the solution is exp(-r^2 / (sigma^2 + 4*alpha*t)), plus the
    //   same image sum for periodization. This is a grid-independent reference.
    //
    // Minimum reduction factor:
    //   For a smooth Gaussian the actual reduction per doubling is typically
    //   factor 50-500+ (proper spectral convergence). The test enforces factor
    //   10 as a conservative floor that filters out non-convergence without
    //   being fragile.
    // ------------------------------------------------------------

    // --- parameters fixed across all refinement levels ---

    const Real conv_Lx        = Real{2};
    const Real conv_Ly        = Real{2};
    const Real conv_alpha     = Real{1.0};
    const Real conv_amplitude = Real{1.0};
    const Real conv_sigma     = Real{0.20};   // fixed physical width
    const std::size_t conv_image_radius = 2;
    const Real conv_t         = Real{0.10};

    const std::vector<std::size_t> conv_grid_sizes = {8, 16, 32, 64};

    const Real conv_min_reduction = Real{10};

    // exact solution at time t for the periodized Gaussian IC
    // u(x,y,t) = A * (sigma^2 / (sigma^2 + 4*alpha*t)) * exp(-r^2 / (sigma^2 + 4*alpha*t))
    // The prefactor accounts for amplitude decay as the Gaussian spreads.
    auto make_exact_gaussian_solution = [&](std::size_t N, Real time) -> Grid2D<Real> {

        const Real dx = conv_Lx / static_cast<Real>(N);
        const Real dy = conv_Ly / static_cast<Real>(N);
        const Real x_min = -conv_Lx / Real{2};
        const Real y_min = -conv_Ly / Real{2};
        const Real sigma_sq    = conv_sigma * conv_sigma;
        const Real sigma_t_sq  = sigma_sq + Real{4} * conv_alpha * time;

        // The heat equation solution for u0 = A*exp(-r^2/sigma^2) on R^2 is
        //
        //   u(x,y,t) = A * (sigma^2 / (sigma^2 + 4*alpha*t))
        //                * exp(-r^2 / (sigma^2 + 4*alpha*t))
        //
        // The prefactor sigma^2/(sigma^2+4*alpha*t) comes from the heat kernel
        // convolution and accounts for the Gaussian losing amplitude as it
        // spreads. At t=0 the prefactor is 1 and the formula reduces to the IC.
        const Real prefactor   = sigma_sq / sigma_t_sq;

        const std::ptrdiff_t ir = static_cast<std::ptrdiff_t>(conv_image_radius);

        Grid2D<Real> exact(N, N);

        for(std::size_t i = 0; i < N; ++i) {
            const Real x = x_min + static_cast<Real>(i) * dx;

            for(std::size_t j = 0; j < N; ++j) {
                const Real y = y_min + static_cast<Real>(j) * dy;

                Real value = Real{0};

                for(std::ptrdiff_t m = -ir; m <= ir; ++m) {
                    const Real xm = x + static_cast<Real>(m) * conv_Lx;

                    for(std::ptrdiff_t n = -ir; n <= ir; ++n) {
                        const Real yn = y + static_cast<Real>(n) * conv_Ly;

                        value += std::exp(-(xm * xm + yn * yn) / sigma_t_sq);
                    }
                }

                exact(i, j) = conv_amplitude * prefactor * value;
            }
        }

        return exact;
    };

    auto l_inf_error_real = [](const Grid2D<Real>& expected, const Grid2D<Real>& actual) -> Real {
        Real max_err = Real{0};

        for(std::size_t i = 0; i < expected.nx(); ++i) {
            for(std::size_t j = 0; j < expected.ny(); ++j) {
                const Real err = std::abs(expected(i, j) - actual(i, j));
                if(err > max_err) max_err = err;
            }
        }

        return max_err;
    };

    // --- run the convergence sweep ---

    {
        std::cout << "--- Spectral convergence table (Gaussian IC, t=" << conv_t << ") ---\n";
        std::cout << std::setw(6)  << "N"
                  << std::setw(20) << "L-inf error"
                  << std::setw(22) << "reduction factor"
                  << "\n";
        std::cout << std::string(48, '-') << "\n";

        std::vector<Real> conv_errors;
        conv_errors.reserve(conv_grid_sizes.size());

        for(std::size_t level = 0; level < conv_grid_sizes.size(); ++level) {

            const std::size_t N = conv_grid_sizes[level];
            const std::string level_name = "spectral_convergence_N" + std::to_string(N);

            Heat2DConfig cfg;
            cfg.nx           = N;
            cfg.ny           = N;
            cfg.Lx           = conv_Lx;
            cfg.Ly           = conv_Ly;
            cfg.alpha        = conv_alpha;
            cfg.output_times = {conv_t};

            Grid2D<Real> u0 = make_gaussian_ic(
                cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
                conv_amplitude,
                std::optional<Real>{conv_sigma},
                conv_image_radius,
                conv_image_radius,
                ValidationConfig{}
            );

            Heat2DFourierSolver solver(cfg);
            solver.set_initial_condition(u0);
            std::vector<Grid2D<Real>> snapshots = solver.solve();

            Grid2D<Real> exact = make_exact_gaussian_solution(N, conv_t);
            const Real err = l_inf_error_real(exact, snapshots[0]);
            conv_errors.push_back(err);

            if(level == 0) {
                std::cout << std::setw(6)  << N
                          << std::setw(20) << std::scientific << std::setprecision(6) << err
                          << std::setw(22) << "(baseline)"
                          << "\n";
            } else {
                const Real prev_err = conv_errors[level - 1];
                const Real reduction = (err > Real{0})
                    ? (prev_err / err)
                    : std::numeric_limits<Real>::infinity();

                std::cout << std::setw(6)  << N
                          << std::setw(20) << std::scientific << std::setprecision(6) << err
                          << std::setw(22) << std::fixed << std::setprecision(2) << reduction
                          << "\n";

                // Once both errors are below the double-precision floor there is
                // no meaningful signal left. Rounding noise can cause a tiny
                // non-monotone bump (e.g. 4e-17 -> 1e-16) that is not a real
                // regression. Skip the rate checks in that regime.
                const Real fp_floor = Real{1e-14};
                const bool at_fp_floor = (prev_err < fp_floor && err < fp_floor);

                if(at_fp_floor) {
                    std::cout << "  (both errors below fp floor " << fp_floor
                              << ", rate checks skipped)\n";
                }
                else if(err >= prev_err) {
                    std::cout << "[FAIL] " << level_name
                              << ": error did not decrease"
                              << " (prev=" << prev_err << ", current=" << err << ")\n";
                    failed_tests.push_back(level_name + "_error_not_decreasing");
                }
                else if(reduction < conv_min_reduction) {
                    std::cout << "[FAIL] " << level_name
                              << ": reduction factor " << reduction
                              << " is below minimum " << conv_min_reduction
                              << " (expected spectral, not polynomial)\n";
                    failed_tests.push_back(level_name + "_rate_too_slow");
                }
            }
        }

        // Finest-level absolute check. Spectral convergence for this sigma
        // and image radius gives ~1e-12 at N=64 in practice; 1e-6 is the floor.
        {
            const Real finest_err = conv_errors.back();
            const Real finest_threshold = Real{1e-6};
            const std::string finest_name = "spectral_convergence_finest_level_error";

            if(finest_err > finest_threshold) {
                std::cout << "[FAIL] " << finest_name
                          << ": L-inf error at N=" << conv_grid_sizes.back()
                          << " is " << finest_err
                          << ", expected below " << finest_threshold << "\n";
                failed_tests.push_back(finest_name);
            }
        }

        std::cout << "\n";
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