// bench_transforms.cpp
// Responsibility:
//   Executable driving every transform benchmark: the 1D FFT, DFT and FFTW
//   sweeps, the 2D square sweep for both the custom kernel and FFTW, and the
//   rectangular aspect pair.
//
//   This file owns the configurations. bench_common owns the machinery that
//   measures one of them, and bench_csv owns turning a result into a row.
//
// Structure:
//   Three phases, in order.
//
//   1. Build the configuration list. Each configuration runs its correctness
//      check against FFTW and probes for a rep count, both once, outside
//      every timed region. A configuration that fails its check aborts the
//      run rather than emitting timings from a binary that computes the wrong
//      answer.
//
//   2. Expand configurations into one job per trial, then shuffle. Running in
//      declaration order would mean the machine is hotter at the large sizes
//      than the small ones, so thermal drift would be indistinguishable from
//      scaling. Shuffling spreads the drift evenly across sizes instead. This
//      matters on a laptop; a dedicated compute node is stable enough that
//      most published benchmarks skip it.
//
//   3. Execute the shuffled list, writing each row as it is produced.
//
// FFTW plans:
//   Built before anything is timed and held for the whole run, because
//   planning with FFTW_MEASURE takes seconds at the larger sizes and could
//   never sit inside a shuffled loop. Holding every 2D plan at once costs
//   roughly 360 MB of buffers, dominated by the 4096 entry.
//
// Scaffolding to be replaced:
//   - run_id is hardcoded. The harness script will supply it on the command
//     line, since the memory benchmark spans several processes that must
//     share one.
//   - machine_info is left empty. bench_platform will fill it.
//   - build_provenance is left empty. make_run_provenance currently requires
//     a RunConfig, which this executable has no reason to build.
//
// Output:
//   benchmarks/transform_timings.csv, overwritten on each run, plus
//   benchmarks/fftw_wisdom.dat, which persists across runs.

#include <algorithm>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <fftw3.h>

#include "bench_common.hpp"
#include "bench_csv.hpp"
#include "benchmark_types.hpp"
#include "dft1d.hpp"
#include "fft1d.hpp"
#include "fft2d.hpp"
#include "grid2d.hpp"
#include "types.hpp"
#include "bench_platform.hpp"
#include "bench_metadata.hpp"
#include "snapshot_writer.hpp"


// One measured configuration. The error and the rep count are established
// once, before any timing, and shared by every trial of that configuration so
// its rows stay comparable.
struct configuration {

    Benchmark_name name;
    transform transform_type;
    grid_size size;

    Real error = Real{0};
    Real roundtrip_error = Real{0};
    std::size_t reps = 1;
};


// One unit of work: a configuration and which trial of it this is.
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

        // Queried once at startup: hostname, OS, CPU model, and the cache
        // sizes that annotate the cache-cliff plot. Recorded in the run
        // metadata so a result can be attributed to the machine that produced
        // it rather than to an unnamed laptop.
        context.machine = query_machine_info();
        std::cout << context.machine.hostname
                  << " / " << context.machine.os_name
                  << " / " << context.machine.cpu_model << "\n";

        context.build_provenance = make_run_provenance();

        context.fftw_planner_flag = "FFTW_MEASURE";

        // create_directories succeeds silently when the directory already
        // exists, and the CSV writers open with truncation, so a repeated run
        // overwrites the rows rather than appending to them.
        std::filesystem::create_directories(context.output_dir);

        const std::string wisdom_path = context.output_dir + "/fftw_wisdom.dat";

        // FFTW_MEASURE picks its algorithm by timing candidates on the day, so
        // two sessions can select different decompositions for the same size
        // and differ by double digits for no visible reason. Importing pins
        // the choice to whatever the first session found.
        //
        // A missing file is the ordinary first-run case, not an error.
        const bool wisdom_imported = import_fftw_wisdom(wisdom_path);

        context.fftw_wisdom_policy = wisdom_imported ? "imported" : "generated";

        TransformCsvWriter writer(context.output_dir + "/transform_timings.csv", context);

        const std::size_t trials = 7;

        // Provisional. The custom FFT computes its twiddle factors by phase
        // recurrence rather than from a table, so its roundoff accumulates
        // faster than the eps*log2(N) model predicts: at n = 1024 it sits near
        // 68*eps where the model says about 10*eps. This factor is loose
        // enough not to fail correct code while still being orders of
        // magnitude below what a real bug produces. Calibrate it against the
        // measured sweep rather than leaving it guessed.
        const Real safety_factor = Real{1000};

        // -------------------------------------------------------------------
        // Size ladders
        // -------------------------------------------------------------------

        // The DFT stops at 128 because it is O(n^2). The benchmark's purpose
        // is showing the divergence from the FFT, not measuring how long a
        // quadratic algorithm can be made to run.
        const std::vector<std::size_t> sizes_1d_dft = {8, 16, 32, 64, 128};

        const std::vector<std::size_t> sizes_1d = {
            8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

        // 4096 is a single extended point past the regular sweep. Each grid at
        // that size is 268 MB, and time_transform_2d holds two of them because
        // it measures the composed transform and the row/column split in
        // separate loops.
        const std::vector<std::size_t> sizes_2d = {
            16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

        // The aspect pair. Both shapes have identical flop counts and
        // identical memory footprints, since 5*nx*ny*log2(nx*ny) is symmetric
        // under swapping the dimensions. The only difference is which
        // dimension is contiguous, so any timing gap between them is memory
        // access behaviour and nothing else. Tagged separately so it never
        // enters the scaling fit, where mixed aspect ratios would make a
        // fitted exponent meaningless.
        const std::vector<grid_size> sizes_aspect = {
            grid_size{512, 2048},
            grid_size{2048, 512}};

        // -------------------------------------------------------------------
        // FFTW plans
        //
        // Plans are non-copyable, so they are held by pointer.
        // -------------------------------------------------------------------

        std::map<std::size_t, std::unique_ptr<Fftw1dPlan>> plans_1d;

        for(const std::size_t n : sizes_1d){

            plans_1d[n] = std::make_unique<Fftw1dPlan>(n, FFTW_MEASURE);
        }

        std::map<std::pair<std::size_t, std::size_t>, std::unique_ptr<Fftw2dPlan>> plans_2d;

        for(const std::size_t n : sizes_2d){

            plans_2d[{n, n}] = std::make_unique<Fftw2dPlan>(n, n, FFTW_MEASURE);
        }

        for(const grid_size& size : sizes_aspect){

            plans_2d[{size.nx, size.ny}] =
                std::make_unique<Fftw2dPlan>(size.nx, size.ny, FFTW_MEASURE);
        }

        // -------------------------------------------------------------------
        // Phase 1: configurations
        // -------------------------------------------------------------------

        std::vector<configuration> configurations;

        for(const std::size_t n : sizes_1d){

            configurations.push_back({Benchmark_name::FFT_1d_time, transform::FFT, grid_size{n, 1}});
            configurations.push_back({Benchmark_name::FFTW_1d_time, transform::FFTW, grid_size{n, 1}});
        }

        for(const std::size_t n : sizes_1d_dft){

            configurations.push_back({Benchmark_name::DFT_1d_time, transform::DFT, grid_size{n, 1}});
        }

        for(const std::size_t n : sizes_2d){

            configurations.push_back({Benchmark_name::FFT_2d_time, transform::FFT_2d, grid_size{n, n}});
            configurations.push_back({Benchmark_name::FFTW_2d_time, transform::FFTW_2d, grid_size{n, n}});
        }

        for(const grid_size& size : sizes_aspect){

            configurations.push_back({Benchmark_name::FFT_2d_aspect, transform::FFT_2d, size});
        }

        std::cout << "preparing " << configurations.size() << " configurations\n";

        for(configuration& config : configurations){

            // Derived from the session seed and the grid shape only, so every
            // transform benchmarked at a given size sees identical data and
            // their spectra stay comparable offline.
            const std::uint32_t seed =
                derive_seed(context.base_seed, config.size.nx, config.size.ny);

            if(config.size.ny == 1){

                const ComplexVec input = make_random_input_1d(config.size.nx, seed);

                Fftw1dPlan& plan = *plans_1d.at(config.size.nx);

                // Loaded after construction, never before: planning with
                // FFTW_MEASURE overwrites the buffer while timing candidate
                // algorithms.
                plan.load(input);
                plan.execute_forward();

                const ComplexVec reference = plan.store();

                // Both forward transforms are unnormalized, so the comparison
                // is direct with no scaling applied to either side.
                //
                // FFTW rows carry the FFT-against-FFTW difference: the error
                // belongs to the comparison rather than to either
                // implementation alone.
                ComplexVec computed;

                if(config.transform_type == transform::DFT){

                    computed = dft_1d(input);
                }
                else{

                    computed = input;

                    fft_1d_inplace(computed);
                }

                config.error = relative_linf_error(computed, reference);

                const transform tolerance_model =
                    (config.transform_type == transform::DFT) ? transform::DFT : transform::FFT;

                const Real tolerance =
                    error_tolerance(config.size.nx, config.size.ny, tolerance_model, safety_factor);

                if(config.error > tolerance){

                    throw std::runtime_error(
                        "bench_transforms: 1D transform disagrees with FFTW beyond tolerance");
                }

                // Self-contained accuracy check, computed once per
                // configuration outside every timed region. Only the custom
                // FFT has a round trip to measure: the DFT has no inverse
                // here, and FFTW's directions are both unnormalized so its
                // round trip would measure a different quantity.
                if(config.transform_type != transform::DFT){
 
                    config.roundtrip_error = round_trip_error_1d(input);
 
                    if(config.roundtrip_error > tolerance){
 
                        throw std::runtime_error(
                            "bench_transforms: 1D round trip exceeds tolerance");
                    }
                }

                // Probed once and reused across every trial, so all of a
                // configuration's rows share one rep count.
                if(config.transform_type == transform::FFTW){

                    plan.load(input);

                    config.reps = choose_reps(time_fftw_1d(plan, 1).total_time_ns, context.policy);
                }
                else{

                    config.reps = choose_reps(
                        time_transform_1d(input, config.transform_type, 1).total_time_ns,
                        context.policy);
                }
            }
            else{

                const Grid2D<Complex> input =
                    make_random_input_2d(config.size.nx, config.size.ny, seed);

                Fftw2dPlan& plan = *plans_2d.at({config.size.nx, config.size.ny});

                plan.load(input);
                plan.execute_forward();

                const Grid2D<Complex> reference = plan.store();

                Grid2D<Complex> computed = input;

                fft_2d_inplace(computed);

                config.error = relative_linf_error(computed, reference);

                const Real tolerance = error_tolerance(
                    config.size.nx, config.size.ny, transform::FFT_2d, safety_factor);

                if(config.error > tolerance){

                    throw std::runtime_error(
                        "bench_transforms: 2D transform disagrees with FFTW beyond tolerance");
                }

                config.roundtrip_error = round_trip_error_2d(input);
 
                if(config.roundtrip_error > tolerance){
 
                    throw std::runtime_error(
                        "bench_transforms: 2D round trip exceeds tolerance");
                }

                if(config.transform_type == transform::FFTW_2d){

                    plan.load(input);

                    config.reps = choose_reps(time_fftw_2d(plan, 1).total_time_ns, context.policy);
                }
                else{

                    config.reps = choose_reps(
                        time_transform_2d(input, config.transform_type, 1).total_time_ns,
                        context.policy);
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

        // Shuffled with the session seed so the order is reproducible from the
        // metadata rather than merely random.
        std::mt19937 order_gen(context.base_seed);

        std::shuffle(jobs.begin(), jobs.end(), order_gen);

        std::cout << "running " << jobs.size() << " jobs\n";

        // -------------------------------------------------------------------
        // Phase 3: execute
        // -------------------------------------------------------------------

        for(const job& item : jobs){

            const configuration& config = configurations[item.configuration_index];

            const std::uint32_t seed =
                derive_seed(context.base_seed, config.size.nx, config.size.ny);

            Transform_Result result;

            result.name = config.name;
            result.transform_type = config.transform_type;
            result.nx = config.size.nx;
            result.ny = config.size.ny;
            result.trial = item.trial;
            result.error = config.error;
            result.roundtrip_error = config.roundtrip_error;
 

            if(config.size.ny == 1){

                // Regenerated per job rather than cached: the shuffled order
                // visits sizes arbitrarily, so caching would mean holding
                // every input at once. Generation is outside the timed region,
                // and writing the values also faults in the pages, so the
                // measurement does not pay for first touch.
                const ComplexVec input = make_random_input_1d(config.size.nx, seed);

                if(config.transform_type == transform::FFTW){

                    Fftw1dPlan& plan = *plans_1d.at(config.size.nx);

                    // A discarded warm-up batch: the first pass over a cold
                    // cache is not representative of the steady state the
                    // measured batch runs in.
                    plan.load(input);
                    time_fftw_1d(plan, config.reps);

                    plan.load(input);

                    const timed_batch batch = time_fftw_1d(plan, config.reps);

                    result.reps_used = batch.reps_used;
                    result.paired = batch.paired;
                    result.total_time_ns = batch.total_time_ns;
                }
                else{

                    time_transform_1d(input, config.transform_type, config.reps);

                    const timed_batch batch =
                        time_transform_1d(input, config.transform_type, config.reps);

                    result.reps_used = batch.reps_used;
                    result.paired = batch.paired;
                    result.total_time_ns = batch.total_time_ns;
                }

                // row_time_ns and col_time_ns stay absent: a 1D transform has
                // no pass structure to report, and an empty field says that
                // where a zero would look like a measurement.
            }
            else{

                const Grid2D<Complex> input =
                    make_random_input_2d(config.size.nx, config.size.ny, seed);

                if(config.transform_type == transform::FFTW_2d){

                    Fftw2dPlan& plan = *plans_2d.at({config.size.nx, config.size.ny});

                    plan.load(input);
                    time_fftw_2d(plan, config.reps);

                    plan.load(input);

                    const timed_batch batch = time_fftw_2d(plan, config.reps);

                    result.reps_used = batch.reps_used;
                    result.paired = batch.paired;
                    result.total_time_ns = batch.total_time_ns;

                    // FFTW does not expose its internal pass structure, so the
                    // row and column columns stay absent for these rows.
                }
                else{

                    time_transform_2d(input, config.transform_type, config.reps);

                    const timed_batch_2d batch =
                        time_transform_2d(input, config.transform_type, config.reps);

                    result.reps_used = batch.reps_used;
                    result.paired = batch.paired;
                    result.total_time_ns = batch.total_time_ns;
                    result.row_time_ns = batch.row_time_ns;
                    result.col_time_ns = batch.col_time_ns;
                }
            }

            writer.write(result);
        }

        // Exported after every plan exists, so the file records the whole run
        // rather than a partial set that later sessions would only partly
        // benefit from.
        export_fftw_wisdom(wisdom_path);

        // Written after the last row: a metadata file describing a run that
        // did not finish would be misleading.
        std::map<std::string, std::string> benchmark_config;

        benchmark_config["trials"] = std::to_string(trials);
        benchmark_config["safety_factor"] = std::to_string(safety_factor);
        benchmark_config["sizes_1d"] = "8..32768 powers of two";
        benchmark_config["sizes_1d_dft"] = "8..128 powers of two";
        benchmark_config["sizes_2d"] = "16..4096 powers of two";
        benchmark_config["sizes_aspect"] = "512x2048, 2048x512";
        benchmark_config["configurations"] = std::to_string(configurations.size());
        benchmark_config["jobs"] = std::to_string(jobs.size());

        write_run_metadata(context.output_dir + "/transform_metadata.json",
                           context, benchmark_config);

        std::cout << "wrote " << context.output_dir << "/transform_timings.csv\n";
    }
    catch(const std::exception& e){

        std::cerr << "bench_transforms: " << e.what() << "\n";
        return 1;
    }

    return 0;
}