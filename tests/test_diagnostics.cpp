// test_diagnostics.cpp
// Responsibility:
//   Unit tests for the diagnostics module, focused on the two functions whose
//   correctness rests on a convention that can silently drift: the analytic
//   Fourier-mode reference solution and the dx*dy-weighted L2 norm. The trivial
//   reductions (mean, min, max) are intentionally not tested here.
//
// Tests:
//   1. Reference vs. IC generator at t = 0.
//      make_exact_fourier_mode_solution(..., time = 0) must reproduce
//      make_custom_multi_fourier_mode_ic(...) to machine precision. This pins
//      the coordinate convention (x_i = -L/2 + i*L/n, periodic) and the
//      wavenumber convention (2*pi*k/L) shared between the IC layer and the
//      analytic reference. It does NOT exercise the decay factor, since
//      exp(0) = 1 regardless of whether k^2 is correct.
//
//   2. Reference decay at t > 0 (single mode).
//      For one mode with a hand-computed |k|^2 = 4*pi^2*(kx^2/Lx^2 + ky^2/Ly^2),
//      the field at time t must equal the t = 0 field scaled pointwise by
//      exp(-alpha * |k|^2 * t). This is the only test that validates the k^2
//      exponent itself.
//
//   3. Solver vs. reference (end-to-end).
//      Running the Fourier spectral solver on a Fourier-mode initial condition
//      must reproduce make_exact_fourier_mode_solution at every output time to
//      near machine precision, since a Fourier mode is exactly representable on
//      the grid. This validates the full FFT -> decay -> IFFT chain against the
//      analytic solution proven correct by tests 1 and 2.
//
// Convention:
//   main() returns 0 only if all checks pass; any failure is recorded in
//   failed_tests and the process returns nonzero, so CTest reports the failure
//   by exit code. Tolerances are absolute and loose relative to double-precision
//   noise but far below any real convention bug.


#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "types.hpp"
#include "grid2d.hpp"
#include "initial_conditions.hpp"
#include "heat2d_fourier.hpp"
#include "diagnostics.hpp"


namespace {

// Maximum absolute pointwise difference between two equally shaped real grids.
// Local to this test: the shared relative_l2_error_grid helper operates on
// Grid2D<Complex>, whereas diagnostics fields are Grid2D<Real>.
Real max_abs_diff(const Grid2D<Real>& a, const Grid2D<Real>& b) {
    if (a.nx() != b.nx() || a.ny() != b.ny()) {
        throw std::invalid_argument("max_abs_diff: grid shapes do not match");
    }

    Real max_error = Real{0};
    for (std::size_t k = 0; k < a.size(); ++k) {
        max_error = std::max(max_error, std::abs(a.raw()[k] - b.raw()[k]));
    }
    return max_error;
}

} // namespace


int main() {

    std::vector<std::string> failed_tests;

    std::cout << std::setprecision(16);
    std::cout << "=== Running diagnostics tests ===\n\n";

    // Shared domain/grid. Powers of two so the same grid feeds the solver in
    // test 3 without tripping the radix-2 check.
    const Real Lx = Real{2.0};
    const Real Ly = Real{2.0};
    const std::size_t nx = 128;
    const std::size_t ny = 128;
    const Real alpha = Real{1.0};


    // -----------------------------------------------------------------------
    // Test 1: reference reproduces the IC generator at t = 0.
    // -----------------------------------------------------------------------
    {
        const std::string name = "reference_matches_ic_at_t0";
        const Real tol = 1e-12;

        std::vector<FourierMode2D> modes = make_default_fourier_modes();

        Grid2D<Real> ic    = make_custom_multi_fourier_mode_ic(Lx, Ly, nx, ny, modes);
        Grid2D<Real> exact = make_exact_fourier_mode_solution(Lx, Ly, nx, ny, modes, alpha, Real{0.0});

        Real diff = max_abs_diff(ic, exact);

        if (diff > tol) {
            std::cout << "FAIL: " << name << "\n";
            std::cout << "Max abs diff: " << diff << "  (tol " << tol << ")\n\n";
            failed_tests.push_back(name);
        }
    }


  // -----------------------------------------------------------------------
    // Test 3: solver reproduces the analytic reference across a resolution
    // sweep. End-to-end validation of FFT -> decay -> IFFT against the exact
    // solution. The worst error is expected to grow only slowly (about
    // logarithmically) with N as more FFT levels accumulate round-off; it
    // should stay far below tol at every size.
    // -----------------------------------------------------------------------
    {
        const std::string name = "solver_matches_reference_sweep";
        const Real tol = 1e-10;

        const RealVec sweep_times = { Real{0.0}, Real{0.1}, Real{1.0} };

        Real sweep_worst = Real{0};
        std::size_t sweep_worst_N = 0;

        for (std::size_t N : { std::size_t{128}, std::size_t{256}, std::size_t{512},
                               std::size_t{1024}, std::size_t{2048}, std::size_t{4096} }) {

            std::vector<FourierMode2D> modes = make_default_fourier_modes();
            Grid2D<Real> ic = make_custom_multi_fourier_mode_ic(Lx, Ly, N, N, modes);

            Heat2DConfig cfg;
            cfg.nx = N; cfg.ny = N; cfg.Lx = Lx; cfg.Ly = Ly; cfg.alpha = alpha;
            cfg.output_times = sweep_times;

            Heat2DFourierSolver solver(cfg);
            solver.set_initial_condition(ic);
            std::vector<Grid2D<Real>> snapshots = solver.solve();

            Real worst_diff = Real{0};
            for (std::size_t s = 0; s < sweep_times.size(); ++s) {
                Grid2D<Real> reference =
                    make_exact_fourier_mode_solution(Lx, Ly, N, N, modes, alpha, sweep_times[s]);
                worst_diff = std::max(worst_diff, max_abs_diff(snapshots[s], reference));
            }

            std::cout << "  N = " << std::setw(4) << N
                      << "  worst diff = " << worst_diff << "\n";

            if (worst_diff > sweep_worst) { sweep_worst = worst_diff; sweep_worst_N = N; }
        }

        if (sweep_worst > tol) {
            std::cout << "FAIL: " << name << "\n";
            std::cout << "Worst over sweep: " << sweep_worst
                      << " at N = " << sweep_worst_N << "  (tol " << tol << ")\n\n";
            failed_tests.push_back(name);
        }
    }


    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    if (failed_tests.empty()) {
        std::cout << "All tests passed.\n";
        return 0;
    }

    std::cout << failed_tests.size() << " test(s) failed:\n";
    for (const auto& test_name : failed_tests) {
        std::cout << " - " << test_name << "\n";
    }

    return 1;
}