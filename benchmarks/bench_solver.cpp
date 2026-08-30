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


// One measured configuration. The analytic error is established once, before
// any timing, and shared by every trial so the rows stay comparable.
struct configuration {

    Benchmark_name name;
    grid_size size;

    InitialConditionParams ic;
    std::string ic_name;

    bool write_output = false;
    int gzip_level = 0;

    Real error = Real{0};
    bool has_error = false;
};


struct job {

    std::size_t configuration_index = 0;
    std::size_t trial = 0;
};


int main(){

    try{

        // -------------------------------------------------------------------
        // Run context
        // -------------------------------------------------------------------

        run_context context;

        context.run_id = "solver";
        context.output_dir = "benchmarks/results";
        context.base_seed = 123456789;

        std::filesystem::create_directories(context.output_dir);

        SolverCsvWriter writer(context.output_dir + "/solver_timings.csv", context);

        if(!timing::enabled()){

            std::cerr << "warning: built without HEAT2D_ENABLE_TIMING, so the phase\n"
                      << "         columns will all read zero. Reconfigure with\n"
                      << "         -DHEAT2D_ENABLE_TIMING=ON for the profile.\n";
        }

        // Repetition is carried entirely by trials: a solve at any benchmarked
        // grid size is far above the timing floor, so reps are always 1 and no
        // rep probe is needed.
        //
        // Seven rather than five, matching the transform suite. The median of
        // five is fragile when the spread is wide, and solver jobs vary far
        // more than the transform batches did: a single solve runs for
        // hundreds of milliseconds to seconds, long enough for the machine's
        // thermal state to matter.
        const std::size_t trials = 7;

        // Cooldown between jobs, as a fraction of the job's own measured
        // duration and capped so a long job cannot stall the sweep.
        //
        // The transform suite did not need this: each measurement was a 50 ms
        // batch, short enough that the machine stayed near steady state. Here
        // the jobs are seconds long and run back to back, so without a gap the
        // processor heats continuously and later jobs run throttled. Shuffling
        // decorrelates that drift from problem size, but it does not remove
        // the variance it adds, and the first run showed a 57 percent spread
        // across trials at the largest grid.
        const double cooldown_fraction = 0.5;
        const std::uint64_t cooldown_cap_ns = 2'000'000'000;

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
        for(const int gzip_level : {0, 4}){

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

            const solver_timing timing =
                time_solve(initial_condition, solver_config, snapshots);

            Solver_Result result;

            result.name = config.name;
            result.fft_backend = FftBackend::custom;
            result.ic_name = config.ic_name;
            result.nx = config.size.nx;
            result.ny = config.size.ny;
            result.trial = item.trial;
            result.reps_used = 1;

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
            }

            writer.write(result);

            // Recorded so the terminal summary can report the spread, which is
            // the number that says whether the cooldown is doing its job.
            observed[item.configuration_index].push_back(timing.solve_time_ns);

            // The pause is outside every timed region and after the row is
            // written, so a session interrupted during a cooldown has already
            // saved its measurement.
            const std::uint64_t cooldown_ns = std::min(
                static_cast<std::uint64_t>(cooldown_fraction *
                    static_cast<double>(timing.solve_time_ns)),
                cooldown_cap_ns);

            std::this_thread::sleep_for(std::chrono::nanoseconds(cooldown_ns));
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

        std::cout << "\nwrote " << context.output_dir << "/solver_timings.csv\n";
    }
    catch(const std::exception& e){

        std::cerr << "bench_solver: " << e.what() << "\n";
        return 1;
    }

    return 0;
}