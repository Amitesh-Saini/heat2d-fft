// main.cpp
// Responsibility:
//   Top-level driver for the 2D Fourier heat-equation solver.
//
//   Usage:
//       ./heat2d configs/gaussian_demo.json
//
//   Pipeline:
//       config file -> RunConfig (strict parse + validation)
//                   -> provenance + output file opened (recorded BEFORE the
//                      solve, so a crashed run still documents what was
//                      attempted)
//                   -> grids, initial condition, solver
//                   -> solve, timed
//                   -> snapshots + diagnostics streamed to the writer
//                   -> finalize (diagnostics datasets, wall time, count check)
//
//   All real logic lives in the library; this file only sequences it and
//   converts failures into a clean error message and a nonzero exit code.

#include <exception>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

#include "config_io.hpp"
#include "diagnostics.hpp"
#include "grid2d.hpp"
#include "heat2d_fourier.hpp"
#include "initial_conditions.hpp"
#include "run_config.hpp"
#include "snapshot_writer.hpp"
#include "timer.hpp"
#include "types.hpp"


namespace {

// Recovers the Fourier mode list from the initial-condition variant so the
// analytic reference can be evaluated. Only called when
// config.diagnostics.compute_analytic_error is true, which
// validate_run_config guarantees implies a Fourier-mode IC.
std::vector<FourierMode2D> extract_fourier_modes(const InitialConditionParams& ic){

    if(std::holds_alternative<SingleFourierModeIcParams>(ic)){

        const auto& p = std::get<SingleFourierModeIcParams>(ic);
        return { {p.kx, p.ky, p.amplitude, p.phase} };
    }

    if(std::holds_alternative<MultiFourierModeIcParams>(ic)){

        const auto& p = std::get<MultiFourierModeIcParams>(ic);
        return p.modes.empty() ? make_default_fourier_modes() : p.modes;
    }

    throw std::logic_error(
        "extract_fourier_modes: initial condition is not a Fourier mode "
        "(validate_run_config should have rejected this configuration)");
}

} // namespace


int main(int argc, char** argv){

    if(argc != 2){

        std::cerr << "usage: heat2d <config.json>\n";
        return 1;
    }

    try{

        // 1. Load, strictly parse, and validate the configuration.
        const RunConfig config = load_run_config_from_json(argv[1]);

        // 2. Open the output file and record provenance immediately.
        const RunProvenance provenance = make_run_provenance(config);
        SnapshotWriter writer(config, provenance);

        // 3. Derive the solver configuration and the physical coordinates.
        const Heat2DConfig cfg = make_heat2d_config(config);

        const RealVec x = make_periodic_coordinates(config.solver.Lx, config.solver.nx);
        const RealVec y = make_periodic_coordinates(config.solver.Ly, config.solver.ny);

        writer.write_grids(x, y, cfg.output_times);

        // 4. Build the initial condition and the solver. The solver
        //    constructor runs the authoritative grid/physics validation.
        const Grid2D<Real> initial_condition = make_initial_condition(config);

        Heat2DFourierSolver solver(cfg);
        solver.set_initial_condition(initial_condition);

        // 5. Solve, timed. The wall time covers solve + output, i.e. the
        //    user-visible cost of the run.
        Timer timer;

        const std::vector<Grid2D<Real>> snapshots = solver.solve();

        // 6. Stream snapshots (and per-snapshot diagnostics) to the writer.
        const bool want_analytic_error = config.diagnostics.compute_analytic_error;

        std::vector<FourierMode2D> modes;
        if(want_analytic_error){

            modes = extract_fourier_modes(config.initial_condition);
        }

        for(std::size_t k = 0; k < snapshots.size(); ++k){

            writer.append_snapshot(snapshots[k]);

            if(config.diagnostics.enabled){

                writer.record_diagnostics(compute_snapshot_diagnostics(
                    snapshots[k], cfg.output_times[k], cfg.Lx, cfg.Ly));
            }

            if(want_analytic_error){

                const Grid2D<Real> reference = make_exact_fourier_mode_solution(
                    cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, modes, cfg.alpha, cfg.output_times[k]);

                writer.record_analytic_error(compute_relative_l2_error(
                    snapshots[k], reference, cfg.Lx, cfg.Ly));
            }
        }

        // 7. Finalize: diagnostics datasets, wall time, snapshot-count check.
        writer.finalize(timer.elapsed_seconds());

        std::cout << "wrote " << config.output.output_path
                  << " (" << snapshots.size() << " snapshots, "
                  << timer.elapsed_seconds() << " s)\n";
    }
    catch(const std::exception& e){

        std::cerr << "heat2d: " << e.what() << "\n";
        return 1;
    }

    return 0;
}