// bench_memory.cpp
// Responsibility:
//   Executable measuring one solve's memory footprint: the analytic working
//   set model against the peak resident set size actually observed.
//
//   One configuration per invocation, and exactly one row written. Peak RSS
//   is a whole-process high-water mark that never decreases, so a binary
//   sweeping several sizes would report the largest configuration's footprint
//   on every row. The sweep is therefore driven by the harness script, which
//   deletes the CSV once and then invokes this binary per configuration.
//
// What it measures:
//   The baseline is sampled before any solver array exists, then the solve
//   runs, then the peak is sampled again. The difference is what the arrays
//   cost; the baseline is the process overhead the model does not describe.
//   Reporting both is what makes the ratio interpretable rather than
//   systematically inflated by a constant.
//
//   Measured after the solve rather than after allocation, because resident
//   set counts pages that have been WRITTEN, not merely reserved. The
//   snapshot vector is reserved up front but only faulted in as the solve
//   fills it, so a measurement taken earlier would fall short of the model
//   for a reason that has nothing to do with the model being wrong.
//
// Why the model can differ:
//   Measured RSS also includes the binary, allocator overhead and
//   fragmentation, and FFTW or HDF5 internals. It is expected to exceed the
//   model. The useful quantity is the RATIO and its trend across the sweep:
//   fixed overheads dominate at small sizes and become negligible at large
//   ones, so the ratio should approach one as the grid grows. A ratio that
//   stays flat or widens indicates per-element overhead, an unintended array
//   copy or a temporary that scales with the problem, which is the specific
//   failure this benchmark exists to detect.
//
// Usage:
//   bench_memory <nx> <num_snapshots> [run_id]
//
//   nx is the grid size; the grid is square. run_id defaults to "memory" and
//   is supplied by the harness so every file in a session shares one.
 
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
 
#include "bench_common.hpp"
#include "bench_csv.hpp"
#include "bench_metadata.hpp"
#include "bench_platform.hpp"
#include "benchmark_types.hpp"
#include "grid2d.hpp"
#include "heat2d_fourier.hpp"
#include "initial_conditions.hpp"
#include "run_config.hpp"
#include "snapshot_writer.hpp"
#include "types.hpp"
 
 
int main(int argc, char** argv){
 
    if(argc < 3 || argc > 4){
 
        std::cerr << "usage: bench_memory <nx> <num_snapshots> [run_id]\n";
        return 1;
    }
 
    try{
 
        const std::size_t nx = static_cast<std::size_t>(std::stoul(argv[1]));
        const std::size_t num_snapshots = static_cast<std::size_t>(std::stoul(argv[2]));
 
        // -------------------------------------------------------------------
        // Baseline, sampled before anything solver-related is constructed.
        //
        // Taken first so the number describes the process as it starts:
        // binary, static data, and the allocator's initial arena, with none of
        // the solver's arrays in it yet.
        // -------------------------------------------------------------------
 
        const std::optional<std::uint64_t> baseline = peak_process_memory_bytes();
 
        run_context context;
 
        context.run_id = (argc == 4) ? argv[3] : "memory";
        context.output_dir = "benchmarks/results";
        context.machine = query_machine_info();
        context.build_provenance = make_run_provenance();
 
        std::filesystem::create_directories(context.output_dir);
 
        // -------------------------------------------------------------------
        // Configuration, matching the solver benchmark so the two are
        // comparable: same domain, same physics, same band-limited initial
        // condition.
        // -------------------------------------------------------------------
 
        const Real Lx = Real{2};
        const Real Ly = Real{2};
        const Real alpha = Real{1};
 
        MultiFourierModeIcParams multi_mode;
 
        multi_mode.modes = {
            {std::ptrdiff_t{1},  std::ptrdiff_t{1},  Real{1.00}, Real{0}},
            {std::ptrdiff_t{-1}, std::ptrdiff_t{2},  Real{0.50}, -PI / Real{2}},
            {std::ptrdiff_t{2},  std::ptrdiff_t{0},  Real{0.25}, PI / Real{2}},
            {std::ptrdiff_t{0},  std::ptrdiff_t{-3}, Real{0.75}, PI},
            {std::ptrdiff_t{3},  std::ptrdiff_t{2},  Real{0.40}, PI / Real{4}}};
 
        UniformTimeSpec time_spec;
 
        time_spec.t_start = Real{0};
        time_spec.t_end = Real{0.5};
        time_spec.num_snapshots = num_snapshots;
 
        SolverSettings solver_settings;
 
        solver_settings.nx = nx;
        solver_settings.ny = nx;
        solver_settings.Lx = Lx;
        solver_settings.Ly = Ly;
        solver_settings.alpha = alpha;
 
        RunConfig run_config;
 
        run_config.schema_version = current_schema_version;
        run_config.solver = solver_settings;
        run_config.time_spec = time_spec;
        run_config.initial_condition = multi_mode;
 
        const Heat2DConfig solver_config = make_heat2d_config(run_config);
 
        const Grid2D<Real> initial_condition = make_initial_condition(run_config);
 
        // -------------------------------------------------------------------
        // The solve. Untimed: this benchmark measures space, and the solver
        // benchmark already measures time on the same configuration.
        //
        // The snapshots are held until after the measurement so the vector is
        // still resident when the peak is sampled.
        // -------------------------------------------------------------------
 
        std::vector<Grid2D<Real>> snapshots;
 
        time_solve(initial_condition, solver_config, snapshots);
 
        const std::optional<std::uint64_t> peak = peak_process_memory_bytes();
 
        Memory_Result result;
 
        result.nx = nx;
        result.ny = nx;
        result.num_snapshots = solver_config.output_times.size();
        result.theoretical_bytes =
            solver_working_set_bytes(nx, nx, solver_config.output_times.size());
        result.baseline_rss_bytes = baseline;
        result.peak_rss_bytes = peak;
 
        {
            MemoryCsvWriter writer(context.output_dir + "/memory.csv", context);
 
            writer.write(result);
        }
 
        // A read after the measurement, so nothing above can be optimized
        // away as unobserved.
        if(!snapshots.empty() && !snapshots.back().raw().empty()){
 
            volatile Real guard = snapshots.back().raw()[0];
            (void)guard;
        }
 
        // -------------------------------------------------------------------
        // Metadata, rewritten by every invocation.
        //
        // The machine and build fields are identical across the sweep, so the
        // last writer wins and the file describes the session correctly. The
        // per-configuration values live in the CSV rows, not here.
        // -------------------------------------------------------------------
 
        std::map<std::string, std::string> benchmark_config;
 
        benchmark_config["invocation"] = "one process per configuration";
        benchmark_config["Lx"] = std::to_string(Lx);
        benchmark_config["Ly"] = std::to_string(Ly);
        benchmark_config["alpha"] = std::to_string(alpha);
        benchmark_config["t_start"] = std::to_string(time_spec.t_start);
        benchmark_config["t_end"] = std::to_string(time_spec.t_end);
        benchmark_config["initial_condition"] = "multi_fourier_mode, 5 modes";
 
        write_run_metadata(context.output_dir + "/memory_metadata.json",
                           context, benchmark_config);
 
        const double model_mib =
            static_cast<double>(result.theoretical_bytes) / (1024.0 * 1024.0);
 
        std::cout << "n=" << nx << "  snapshots=" << result.num_snapshots
                  << "  model " << model_mib << " MiB";
 
        if(peak.has_value() && baseline.has_value()){
 
            const double measured_mib =
                static_cast<double>(*peak - *baseline) / (1024.0 * 1024.0);
 
            std::cout << "  measured " << measured_mib << " MiB"
                      << "  ratio " << measured_mib / model_mib;
        }
 
        std::cout << "\n";
    }
    catch(const std::exception& e){
 
        std::cerr << "bench_memory: " << e.what() << "\n";
        return 1;
    }
 
    return 0;
}
 