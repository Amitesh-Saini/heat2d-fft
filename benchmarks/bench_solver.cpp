// bench_solver.cpp
// Responsibility:
//   Executable driving every solver benchmark: the grid size ladder, the
//   initial-condition comparison, and the full run with HDF5 output.
//
//   This file owns the configurations. bench_common owns the machinery that
//   measures one of them, and bench_csv owns turning a result into a row.
//
// Structure:
//   The same three phases as bench_transforms.
//
//   1. Build the configuration list. Each configuration validates its own
//      shape and, where the initial condition has a closed form, records the
//      analytic error. A configuration that fails aborts the run rather than
//      emitting timings from a binary that computes the wrong answer.
//
//   2. Expand configurations into one job per trial, then shuffle. Running in
//      declaration order would leave the machine hotter at the large grid
//      sizes than the small ones, so thermal drift would be
//      indistinguishable from scaling.
//
//   3. Execute the shuffled list, writing each row as it is produced.
//
// Experiments:
//   Numeric_solver     the 128 to 1024 ladder, multi-mode IC, no output.
//                      The phase breakdown comes along automatically, since
//                      the timing registry populates it on every solve, so
//                      there is no separate profiling experiment.
//
//   Solver_ic_compare  all three initial conditions at 512 only. Solver cost
//                      is IC-independent by construction: the same
//                      transforms on the same array sizes regardless of what
//                      the field contains. Confirming that once is what
//                      licenses the ladder above to use a single IC, and if
//                      the three do NOT agree within spread, something is
//                      wrong.
//
//   Full_solver        the same ladder with HDF5 output, at two compression
//                      levels. gzip sits inside the measured I/O time and on
//                      a smooth field its CPU cost can rival the write, so
//                      running level 0 alongside level 4 separates
//                      compressing from writing.
//
// Instrumentation:
//   The phase columns come from the timing registry, which heat2d_fourier.cpp
//   populates from regions annotated inside the real solve. They read as zero
//   in a build without HEAT2D_ENABLE_TIMING; this executable warns at startup
//   rather than silently producing a profile of nothing.
//
// Scaffolding to be replaced:
//   - run_id is hardcoded. The harness script will supply it.
//   - machine_info and build_provenance are left empty, pending
//     bench_platform and bench_metadata.
//
// Output:
//   benchmarks/solver_timings.csv, overwritten on each run. The HDF5 files
//   written by the I/O experiment are removed after measurement.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "bench_common.hpp"
#include "bench_csv.hpp"
#include "benchmark_types.hpp"
#include "diagnostics.hpp"
#include "grid2d.hpp"
#include "heat2d_fourier.hpp"
#include "initial_conditions.hpp"
#include "run_config.hpp"
#include "timing_registry.hpp"
#include "types.hpp"
#include "bench_platform.hpp"
#include "bench_metadata.hpp"
#include "snapshot_writer.hpp"

// One measured configuration. The analytic error is established once, before
// any timing, and shared by every trial so the rows stay comparable.
struct configuration {

    Benchmark_name name;
    grid_size size;

    InitialConditionParams ic;
    std::string ic_name;
    std::size_t reps = 1;

    bool write_output = false;
    int gzip_level = 0;

    Real error = Real{0};
    bool has_error = false;
};


struct job {

    std::size_t configuration_index = 0;
    std::size_t trial = 0;
};


int main(int argc, char** argv){

    try{

        // -------------------------------------------------------------------
        // Run context
        // -------------------------------------------------------------------

        run_context context;

        // Supplied by the harness so every file in a session shares one
        // identifier. Defaulted for direct invocation during development.
        context.run_id = (argc >= 2) ? argv[1] : "transforms";
        context.output_dir = "benchmarks/results";
        context.base_seed = 123456789;

        context.machine = query_machine_info();

        std::cout << context.machine.hostname
                  << " / " << context.machine.os_name
                  << " / " << context.machine.cpu_model << "\n";

        context.build_provenance = make_run_provenance();

        std::filesystem::create_directories(context.output_dir);

        SolverCsvWriter writer(context.output_dir + "/solver_timings.csv", context);

        if(!timing::enabled()){

            std::cerr << "warning: built without HEAT2D_ENABLE_TIMING, so the phase\n"
                      << "         columns will all read zero. Reconfigure with\n"
                      << "         -DHEAT2D_ENABLE_TIMING=ON for the profile.\n";
        }

        // Seven trials, matching the transform suite.
        const std::size_t trials = 7;
 
        // Reps are batched only where a single solve is too short to measure
        // cleanly. At 512 and above one solve runs for hundreds of
        // milliseconds and repetition is carried by trials alone; at 128 a
        // solve is about 29 ms, short enough that one scheduling interruption
        // is a large fraction of it.
        //
        // The floor is well below the 50 ms used for the transforms because a
        // solver rep is far more expensive: clearing 200 ms at 128 takes
        // about seven solves, while clearing it at 256 takes three.
        const std::uint64_t solve_floor_ns = 200'000'000;
        const std::size_t max_solve_reps = 16;
 
        // The cooldown is retained at zero: the first run used a pause
        // proportional to each job's duration on the theory that the spread
        // was thermal. It was not. Closing background applications took the
        // spread at 1024 from 55 percent to under 1 percent, and the pause
        // changed nothing. Left here as a record of what was tried.
        const std::uint64_t cooldown_ns_cap = 0;

        // -------------------------------------------------------------------
        // Physical setup, identical across every configuration
        // -------------------------------------------------------------------

        const Real Lx = Real{2};
        const Real Ly = Real{2};
        const Real alpha = Real{1};

        // Ten output times rather than the production default of fifty. Every
        // snapshot is a full inverse transform, so M multiplies the whole
        // sweep: at 1024 with fifty snapshots one solve runs for seconds and
        // the ladder becomes a half-hour job. Ten is enough to measure the
        // scaling and keeps the suite runnable while iterating.
        //
        // M is fixed across the ladder. If it varied with resolution, total
        // runtime would scale for two reasons at once and the fitted exponent
        // would mean nothing.
        UniformTimeSpec time_spec;

        time_spec.t_start = Real{0};
        time_spec.t_end = Real{0.5};
        time_spec.num_snapshots = 10;

        const std::vector<std::size_t> sizes_ladder = {128, 256, 512, 1024};

        const std::size_t ic_compare_size = 512;

        // -------------------------------------------------------------------
        // Initial conditions
        // -------------------------------------------------------------------

        // Five low-frequency modes, well inside the Nyquist limit at every
        // grid size in the sweep. Band-limited, so the spectral solver
        // represents it exactly and the analytic solution is available: these
        // are the only rows that can carry a correctness number.
        MultiFourierModeIcParams multi_mode;

        multi_mode.modes = {
            {std::ptrdiff_t{1},  std::ptrdiff_t{1},  Real{1.00}, Real{0}},
            {std::ptrdiff_t{-1}, std::ptrdiff_t{2},  Real{0.50}, -PI / Real{2}},
            {std::ptrdiff_t{2},  std::ptrdiff_t{0},  Real{0.25}, PI / Real{2}},
            {std::ptrdiff_t{0},  std::ptrdiff_t{-3}, Real{0.75}, PI},
            {std::ptrdiff_t{3},  std::ptrdiff_t{2},  Real{0.40}, PI / Real{4}}};

        GaussianIcParams gaussian;

        HotSquareIcParams hot_square;

        // Pinned in physical units. The generator's default smoothing width is
        // min(3*dx, 0.10*width), which becomes grid-dependent above roughly
        // nx = 150: the transition would sharpen as the grid refines, so each
        // size would be solving a different problem. Fixing it here keeps one
        // physical initial condition across the whole sweep.
        hot_square.smooth_width_x = Real{0.02};
        hot_square.smooth_width_y = Real{0.02};

        // -------------------------------------------------------------------
        // Phase 1: configurations
        // -------------------------------------------------------------------

        std::vector<configuration> configurations;

        // The ladder uses multi-mode rather than a Gaussian. Solver cost is
        // IC-independent, so the choice is free in timing terms, and
        // multi-mode is the only one with a closed form: every row on the
        // main scaling curve gets a correctness number at no cost.
        for(const std::size_t n : sizes_ladder){

            configuration config;

            config.name = Benchmark_name::Numeric_solver;
            config.size = grid_size{n, n};
            config.ic = multi_mode;
            config.ic_name = initial_condition_type_name(multi_mode);

            configurations.push_back(config);
        }

        // All three initial conditions at one size.
        {
            const std::vector<InitialConditionParams> ic_set = {
                multi_mode, gaussian, hot_square};

            for(const InitialConditionParams& ic : ic_set){

                configuration config;

                config.name = Benchmark_name::Solver_ic_compare;
                config.size = grid_size{ic_compare_size, ic_compare_size};
                config.ic = ic;
                config.ic_name = initial_condition_type_name(ic);

                configurations.push_back(config);
            }
        }

        // The same ladder with output, at two compression levels.
        for(const int gzip_level : {0, 1, 4}){

            for(const std::size_t n : sizes_ladder){

                configuration config;

                config.name = Benchmark_name::Full_solver;
                config.size = grid_size{n, n};
                config.ic = multi_mode;
                config.ic_name = initial_condition_type_name(multi_mode);
                config.write_output = true;
                config.gzip_level = gzip_level;

                configurations.push_back(config);
            }
        }

        std::cout << "preparing " << configurations.size() << " configurations\n";

        for(configuration& config : configurations){

            SolverSettings solver_settings;

            solver_settings.nx = config.size.nx;
            solver_settings.ny = config.size.ny;
            solver_settings.Lx = Lx;
            solver_settings.Ly = Ly;
            solver_settings.alpha = alpha;

            RunConfig run_config;

            run_config.schema_version = current_schema_version;
            run_config.solver = solver_settings;
            run_config.time_spec = time_spec;
            run_config.initial_condition = config.ic;

            const Heat2DConfig solver_config = make_heat2d_config(run_config);

            const Grid2D<Real> initial_condition = make_initial_condition(run_config);

            // Correctness check, computed once per configuration outside every
            // timed region. Only the Fourier-mode initial condition has a
            // closed form to compare against; the Gaussian and the hot square
            // leave the error column empty rather than carrying a number that
            // means something different.
            if(std::holds_alternative<MultiFourierModeIcParams>(config.ic)){

                std::vector<Grid2D<Real>> snapshots;

                time_solve(initial_condition, solver_config, snapshots);

                // Checked at a mid-run output time rather than the last.
                // compute_relative_l2_error falls back to an ABSOLUTE error
                // when the reference norm drops below a small floor, so that
                // the tail of a validation run does not divide by nearly
                // zero. That fallback silently changes what the returned
                // number means, and the threshold below is written for the
                // relative form.
                //
                // This initial condition has zero mean and decays under
                // diffusion: by t_end its slowest mode has fallen by four
                // orders of magnitude, close enough to the floor to be worth
                // avoiding. Mid-run the field is still comfortably clear of
                // it, and the decay has still been applied, so this remains a
                // real test rather than a weaker one.
                const std::size_t check_index = solver_config.output_times.size() / 2;

                const Grid2D<Real> reference = make_exact_fourier_mode_solution(
                    solver_config.Lx, solver_config.Ly,
                    solver_config.nx, solver_config.ny,
                    multi_mode.modes, solver_config.alpha,
                    solver_config.output_times[check_index]);

                config.error = compute_relative_l2_error(
                    snapshots[check_index], reference, solver_config.Lx, solver_config.Ly);

                config.has_error = true;

                // The spectral solver evaluates the decay exactly and a
                // band-limited initial condition is represented exactly, so
                // the only error here is floating-point roundoff. A threshold
                // well above machine precision but far below anything a real
                // bug produces.
                if(config.error > Real{1e-10}){

                    throw std::runtime_error(
                        "bench_solver: analytic error exceeds tolerance");
                }
            }


            // Probe once, outside every timed region, and reuse the count for
            // every trial of this configuration so its rows stay comparable.
            {
                std::vector<Grid2D<Real>> probe_snapshots;
 
                const solver_timing probe =
                    time_solve(initial_condition, solver_config, probe_snapshots);
 
                if(probe.solve_time_ns >= solve_floor_ns){
 
                    config.reps = 1;
                }
                else if(probe.solve_time_ns == 0){
 
                    config.reps = max_solve_reps;
                }
                else{
 
                    const std::uint64_t needed =
                        (solve_floor_ns + probe.solve_time_ns - 1) / probe.solve_time_ns;
 
                    config.reps = static_cast<std::size_t>(
                        std::min(needed, static_cast<std::uint64_t>(max_solve_reps)));
                }
            }
        }

        // -------------------------------------------------------------------
        // Phase 2: job list
        // -------------------------------------------------------------------

        std::vector<job> jobs;

        jobs.reserve(configurations.size() * trials);

        for(std::size_t index = 0; index < configurations.size(); ++index){

            for(std::size_t trial = 0; trial < trials; ++trial){

                jobs.push_back({index, trial});
            }
        }

        std::mt19937 order_gen(context.base_seed);

        std::shuffle(jobs.begin(), jobs.end(), order_gen);

        std::cout << "running " << jobs.size() << " jobs\n";

        // Solve times per configuration, for the spread summary below.
        std::vector<std::vector<std::uint64_t>> observed(configurations.size());

        // -------------------------------------------------------------------
        // Phase 3: execute
        // -------------------------------------------------------------------

        for(const job& item : jobs){

            const configuration& config = configurations[item.configuration_index];

            SolverSettings solver_settings;

            solver_settings.nx = config.size.nx;
            solver_settings.ny = config.size.ny;
            solver_settings.Lx = Lx;
            solver_settings.Ly = Ly;
            solver_settings.alpha = alpha;

            RunConfig run_config;

            run_config.schema_version = current_schema_version;
            run_config.solver = solver_settings;
            run_config.time_spec = time_spec;
            run_config.initial_condition = config.ic;

            const Heat2DConfig solver_config = make_heat2d_config(run_config);

            // Rebuilt per job rather than cached: the shuffled order visits
            // configurations arbitrarily, so caching would mean holding every
            // initial condition at once. Construction is outside the timed
            // region, and writing the values also faults in the pages.
            const Grid2D<Real> initial_condition = make_initial_condition(run_config);

            std::vector<Grid2D<Real>> snapshots;
 
            solver_timing timing;
 
            if(config.reps == 1){
 
                timing = time_solve(initial_condition, solver_config, snapshots);
            }
            else{
 
                // Accumulate across the batch, then divide, so the recorded
                // row describes one solve while the measurement itself ran
                // long enough for interrupt noise to be a small fraction.
                //
                // The registry accumulates process-wide and time_solve clears
                // it on entry, so each rep's phases have to be summed here
                // rather than read once at the end.
                for(std::size_t rep = 0; rep < config.reps; ++rep){
 
                    const solver_timing one =
                        time_solve(initial_condition, solver_config, snapshots);
 
                    timing.solve_time_ns += one.solve_time_ns;
                    timing.forward_transform_time_ns += one.forward_transform_time_ns;
                    timing.spectral_copy_time_ns += one.spectral_copy_time_ns;
                    timing.decay_time_ns += one.decay_time_ns;
                    timing.inverse_transform_time_ns += one.inverse_transform_time_ns;
                }
 
                const std::uint64_t reps = static_cast<std::uint64_t>(config.reps);
 
                timing.solve_time_ns /= reps;
                timing.forward_transform_time_ns /= reps;
                timing.spectral_copy_time_ns /= reps;
                timing.decay_time_ns /= reps;
                timing.inverse_transform_time_ns /= reps;
            }

            Solver_Result result;

            result.name = config.name;
            result.fft_backend = FftBackend::custom;
            result.ic_name = config.ic_name;
            result.nx = config.size.nx;
            result.ny = config.size.ny;
            result.trial = item.trial;
            result.reps_used = config.reps;
 
            result.total_time_ns = timing.solve_time_ns;
            result.forward_transform_time_ns = timing.forward_transform_time_ns;
            result.spectral_copy_time_ns = timing.spectral_copy_time_ns;
            result.decay_time_ns = timing.decay_time_ns;
            result.inverse_transform_time_ns = timing.inverse_transform_time_ns;

            if(config.has_error){

                result.error = config.error;
            }

            if(config.write_output){

                // A distinct path per job. Each measurement writes into free
                // space rather than over an existing large file, which is the
                // reproducible case: overwriting may or may not reuse blocks
                // depending on what ran before.
                const std::string path =
                    context.output_dir + "/io_" + std::to_string(config.size.nx) +
                    "_gz" + std::to_string(config.gzip_level) +
                    "_t" + std::to_string(item.trial) + ".h5";

                const RealVec x = make_periodic_coordinates(Lx, config.size.nx);
                const RealVec y = make_periodic_coordinates(Ly, config.size.ny);

                const io_timing io = time_snapshot_write(
                    snapshots, x, y, solver_config.output_times, path, config.gzip_level);

                result.io_time_ns = io.io_time_ns;
                result.finalize_time_ns = io.finalize_time_ns;
                result.bytes_written = io.bytes_written;
                result.gzip_level = config.gzip_level;
            }

            writer.write(result);
            observed[item.configuration_index].push_back(timing.solve_time_ns);
        }

        // -------------------------------------------------------------------
        // Spread summary
        //
        // Printed rather than written to the CSV: it is derived from rows that
        // are already there, and the analysis layer recomputes it anyway. It
        // is here so a wide spread is visible immediately rather than only
        // after plotting.
        // -------------------------------------------------------------------

        std::cout << "\nsolve time spread by configuration (ms)\n";

        for(std::size_t index = 0; index < configurations.size(); ++index){

            std::vector<std::uint64_t>& samples = observed[index];

            if(samples.empty()){
                continue;
            }

            std::sort(samples.begin(), samples.end());

            const double smallest = static_cast<double>(samples.front()) / 1e6;
            const double largest = static_cast<double>(samples.back()) / 1e6;
            const double middle = static_cast<double>(samples[samples.size() / 2]) / 1e6;

            const configuration& config = configurations[index];

            std::cout << "  " << benchmark_name_to_string(config.name)
                      << "  " << config.ic_name
                      << "  " << config.size.nx << "x" << config.size.ny;

            if(config.write_output){
                std::cout << "  gzip" << config.gzip_level;
            }

            std::cout << "   median " << middle
                      << "   min " << smallest
                      << "   max " << largest
                      << "   spread " << 100.0 * (largest - smallest) / middle << "%\n";
        }

        // Written after the last row: a metadata file describing a run that
        // did not finish would be misleading.
        std::map<std::string, std::string> benchmark_config;

        benchmark_config["trials"] = std::to_string(trials);
        benchmark_config["sizes_ladder"] = "128, 256, 512, 1024";
        benchmark_config["ic_compare_size"] = std::to_string(ic_compare_size);
        benchmark_config["gzip_levels"] = "0, 1, 4";
        benchmark_config["num_snapshots"] = std::to_string(time_spec.num_snapshots);
        benchmark_config["t_start"] = std::to_string(time_spec.t_start);
        benchmark_config["t_end"] = std::to_string(time_spec.t_end);
        benchmark_config["alpha"] = std::to_string(alpha);
        benchmark_config["Lx"] = std::to_string(Lx);
        benchmark_config["Ly"] = std::to_string(Ly);
        benchmark_config["solve_floor_ns"] = std::to_string(solve_floor_ns);
        benchmark_config["max_solve_reps"] = std::to_string(max_solve_reps);
        benchmark_config["configurations"] = std::to_string(configurations.size());
        benchmark_config["jobs"] = std::to_string(jobs.size());

        write_run_metadata(context.output_dir + "/solver_metadata.json",
                           context, benchmark_config);

        std::cout << "\nwrote " << context.output_dir << "/solver_timings.csv\n";
    }
    catch(const std::exception& e){

        std::cerr << "bench_solver: " << e.what() << "\n";
        return 1;
    }

    return 0;
}