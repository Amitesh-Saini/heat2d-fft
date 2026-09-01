#pragma once
// benchmark_types.hpp
// Responsibility:
//   Define the configuration, result, and run-context types shared by every
//   benchmark executable in benchmarks/.
//
//   This header is the schema layer of the benchmark suite. It describes:
//     - which experiments exist (Benchmark_name) and which transform each
//       row was produced by (transform),
//     - what a benchmark is asked to measure (transform_benchmark_config,
//       solver_benchmark_config),
//     - what one measurement produces (Transform_Result, Solver_Result,
//       Memory_Result),
//     - what identifies and describes a whole measurement session
//       (run_context, machine_info, rep_policy, validation_report).
//
// Conventions this header assumes:
//   - All durations are uint64_t nanoseconds from the steady-clock timer.
//   - A rep is one transform inside the timed region; a trial is an
//     independent re-measurement of the whole rep batch. total_time_ns is
//     the batch total, so per-transform time is total_time_ns / reps_used.
//     Reps are chosen adaptively (see rep_policy) and must therefore be
//     recorded on every row.
//   - One row per (configuration, trial). Nothing is aggregated in C++;
//     medians, spreads, GFLOP/s, GB/s, and all normalizations are derived
//     downstream in Python from the raw rows.
//   - std::optional means "not applicable to this row", written as an empty
//     CSV field. A zero is a measurement, not an absent value.
//   - Error metrics are computed once per configuration, outside the timed
//     region, and replicated onto that configuration's rows.
//   - run_id is not stored on result rows; the CSV writer stamps it from the
//     run_context it already holds.
//
// Reuse:
//   Initial conditions, the FFT backend selection, and the output-time
//   specification are NOT redefined here. They come from run_config.hpp so
//   that a benchmarked solve is configured through exactly the same types as
//   a production solve driven by main.cpp.
//
// Note:
//   RunProvenance currently lives in snapshot_writer.hpp, which drags the
//   HDF5 headers into every benchmark translation unit. If that becomes a
//   compile-time nuisance, move RunProvenance and its factory into their own
//   provenance.hpp; nothing here changes.

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "run_config.hpp"
#include "snapshot_writer.hpp"
#include "types.hpp"


// ---------------------------------------------------------------------------
// Experiment and transform identity
// ---------------------------------------------------------------------------

// Which transform produced a row. Each transform is timed in its own
// benchmark run; comparisons such as FFT vs DFT are made downstream by
// filtering rows, not by timing two transforms in one region.
//
// The radix-2 backend requires power-of-two nx and ny, so every size in
// every sweep must be a power of two. Rectangular grids are allowed.
enum class transform {

    DFT, FFT, FFTW, FFT_2d, FFTW_2d
};


// Which experiment a row belongs to. This is the column plotting scripts
// filter on, so experiments that share a transform still need distinct
// names: FFT_2d_aspect must be separable from FFT_2d_time or the
// rectangular pair contaminates the fitted scaling slope.
enum class Benchmark_name {

    FFT_1d_time, DFT_1d_time, FFTW_1d_time,
    FFT_2d_time, FFTW_2d_time, FFT_2d_aspect,
    Numeric_solver, Solver_ic_compare, Full_solver, Numeric_profile,
    Memory_footprint
};

// ---------------------------------------------------------------------------
// Grid sizes
// ---------------------------------------------------------------------------

// One benchmarked grid shape. For 1D transforms ny is 1, so nx * ny is
// always the point count with no special casing. Grid2D maps (i,j) to
// data[i * ny + j], so j is the contiguous index: a row pass runs over ny
// contiguous elements at fixed i, and a column pass strides by ny.
struct grid_size {

    std::size_t nx = 0;
    std::size_t ny = 1;

    std::size_t total_points() const {
        return nx * ny;
    }
};


// ---------------------------------------------------------------------------
// Transform benchmarks
// ---------------------------------------------------------------------------

// One transform sweep. The harness loops over sizes, computes the rep count
// for each size from the run's rep_policy, and emits one row per trial.
struct transform_benchmark_config {

    Benchmark_name name;
    transform transform_type;

    std::vector<grid_size> sizes;

    std::size_t trials = 7;
    std::uint32_t base_seed = 123456789;
};


// One timed batch of a transform benchmark.
struct Transform_Result {

    Benchmark_name name;
    transform transform_type;

    std::size_t nx = 0;
    std::size_t ny = 1;

    std::size_t trial = 0;
    std::size_t reps_used = 0;

    // How the batch was run. With reps == 1 the input is transformed once
    // from a fresh buffer. With reps > 1 the batch alternates forward and
    // inverse so the buffer restores itself, which makes the row a
    // forward/inverse average rather than a forward. Carried into the CSV
    // rather than hidden, since it changes what the time means.
    bool paired = false;

    std::uint64_t total_time_ns = 0;

    // Row and column passes of the 2D transform, timed separately inside
    // fft_2d_inplace. Absent for 1D transforms. total_time_ns is measured as
    // its own region rather than summed, so total minus row minus column is
    // a visible residual covering bit-reversal, gather/scatter, and any
    // allocation the passes do not account for.
    std::optional<std::uint64_t> row_time_ns;
    std::optional<std::uint64_t> col_time_ns;

    // Relative L-infinity difference against the FFTW reference for 1D
    // transforms, relative round-trip error for 2D. Computed once per
    // configuration outside the timed region; the tolerance it is checked
    // against scales with size and algorithm (roughly eps * log2(N) for the
    // FFT and eps * N for the naive DFT), so a single constant threshold is
    // wrong at both ends of the sweep.
    Real error = Real{0};


    // Relative L-infinity error of a forward transform followed by an
    // inverse, measured against the original input.
    //
    // Independent of FFTW: the error column above rests entirely on FFTW
    // being correct, while this one is self-contained. It is also the only
    // check that exercises the 1/n normalization, which lives entirely in
    // the inverse and which a forward-only comparison never touches.
    //
    // Its blind spot is the mirror image: a fault that cancels between the
    // forward and inverse transforms round-trips perfectly. A conjugated
    // transform, for instance, passes this and fails the FFTW comparison.
    // The two checks are complementary rather than redundant.
    Real roundtrip_error = Real{0};
};


// ---------------------------------------------------------------------------
// Solver benchmarks
// ---------------------------------------------------------------------------

// One solver sweep. Reps are always 1: a solve at any benchmarked grid size
// is far above the timing floor, so repetition is carried entirely by
// trials.
//
// initial_conditions holds one entry for the size-ladder run and three for
// the IC-comparison run at a single size. The parameters are the same
// variant main.cpp uses, so a benchmarked IC is built by exactly the same
// code path as a production one.
//
// time_spec fixes the output times. The number of snapshots alone is not
// enough: the decay factors depend on the actual times, and snapshot count
// also drives both the I/O volume and the peak resident set, since solve()
// returns all snapshots at once.
struct solver_benchmark_config {

    Benchmark_name name;
    FftBackend fft_backend = FftBackend::custom;

    std::vector<grid_size> sizes;
    std::vector<InitialConditionParams> initial_conditions;

    Real Lx = Real{2};
    Real Ly = Real{2};
    Real alpha = Real{1};

    TimeSpec time_spec = UniformTimeSpec{};

    std::size_t trials = 5;
    std::uint32_t base_seed = 123456789;

    // false runs the solve alone; true runs the solve and writes snapshots
    // through SnapshotWriter so compute and I/O can be separated.
    bool write_output = false;

    // Compression is inside the measured I/O time. Sweeping this from 0
    // isolates gzip cost from the write itself, so it must be recorded.
    int gzip_level = 4;

    std::string output_dir;
};


// One solve. The phase times decompose total_time_ns; each is measured as
// its own region so no work is attributed to the wrong phase.
struct Solver_Result {

    Benchmark_name name;
    FftBackend fft_backend = FftBackend::custom;

    // From initial_condition_type_name(), so benchmark rows and HDF5
    // metadata use the same IC names.
    std::string ic_name;

    std::size_t nx = 0;
    std::size_t ny = 0;

    std::size_t trial = 0;
    std::size_t reps_used = 1;

    std::uint64_t total_time_ns = 0;

    // Forward transform of the initial condition, once per solve.
    std::uint64_t forward_transform_time_ns = 0;

    // make_snapshot_at_time copies the full spectral grid before applying
    // the decay factor. That copy is neither decay nor inverse transform,
    // so it is timed separately rather than inflating the decay pass.
    std::uint64_t spectral_copy_time_ns = 0;

    // Summed over all snapshots. The decay pass is memory-bound by
    // construction (2 flops per 32 bytes touched), so it is reported
    // downstream as GB/s against measured STREAM bandwidth, not as GFLOP/s.
    std::uint64_t decay_time_ns = 0;
    std::uint64_t inverse_transform_time_ns = 0;

    // Present only when write_output is true. Includes the flush, so the
    // number reflects data handed to the OS rather than a returned write.
    std::optional<std::uint64_t> io_time_ns;

    // The closing cost on its own: finalize() writes the remaining datasets,
    // verifies the snapshot count, and flushes. Included in io_time_ns as
    // well, so the per-snapshot appends are io_time_ns minus this. Present
    // only on rows from runs that wrote output.
    std::optional<std::uint64_t> finalize_time_ns;

    std::optional<std::uint64_t> bytes_written;

    // Compression level the output was written at. Present only on rows from
    // runs that wrote output, and recorded rather than inferred: with more
    // than two levels in the sweep the byte count no longer identifies which
    // one produced it.
    std::optional<int> gzip_level;

    // Relative L-infinity error against the analytic solution. Present only
    // for Fourier-mode initial conditions, which are the only ones with a
    // closed form.
    std::optional<Real> error;
};


// ---------------------------------------------------------------------------
// Memory benchmark
// ---------------------------------------------------------------------------

// One measurement of a solve's memory footprint.
//
// The binary is invoked once per configuration, because peak resident set
// size is a whole-process high-water mark: a process that swept several sizes
// would report the largest one on every row.
//
// num_snapshots is recorded alongside the grid size because the solver holds
// every snapshot in memory at once. The snapshot vector is the dominant term
// in the footprint, so the model scales with snapshot count as well as with
// grid area, and a sweep that varied both at once would measure neither.
struct Memory_Result {
 
    Benchmark_name name = Benchmark_name::Memory_footprint;
 
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t num_snapshots = 0;
 
    // Array working set only: element size times point count times the number
    // of simultaneously live grids, plus the accumulated snapshots.
    std::uint64_t theoretical_bytes = 0;
 
    // Peak resident set before any solver array is allocated: the binary, its
    // static data, the allocator's initial arena, and whatever the runtime
    // brought in. The model does not include any of this, so subtracting it
    // is what makes the comparison meaningful rather than systematically off
    // by a constant.
    std::optional<std::uint64_t> baseline_rss_bytes;
 
    // Peak resident set after the solve, normalized to bytes at the point of
    // measurement: ru_maxrss is kilobytes on Linux and bytes on macOS.
    std::optional<std::uint64_t> peak_rss_bytes;
};


// ---------------------------------------------------------------------------
// Run context
// ---------------------------------------------------------------------------

// Parameters of the adaptive rep policy. Recorded in metadata because
// reps_used cannot be interpreted without them.
struct rep_policy {

    // Above this, one transform is already a valid measurement on its own:
    // choose_reps returns 1, repetition comes from trials alone, and the
    // buffer is transformed once from fresh with nothing to restore.
    std::uint64_t single_call_ns = 1'000'000;   // 1 ms

    std::uint64_t min_timed_ns = 50'000'000;   // rep floor: 50 ms
    std::size_t max_reps = 100000;
    std::size_t warmup_reps = 1;               // discarded, not recorded

};


// Machine description, queried at run start.
struct machine_info {

    std::string hostname;
    std::string os_name;
    std::string cpu_model;

    // Cache sizes annotate the working-set axis of the cache-cliff plot.
    // Apple Silicon has no conventional L3, so l3_bytes may be absent.
    std::uint64_t l1d_bytes = 0;
    std::uint64_t l2_bytes = 0;
    std::optional<std::uint64_t> l3_bytes;

    // A CSV that does not record its precision is not reproducible.
    std::size_t sizeof_real = sizeof(Real);
};


// One correctness check run by the validation gate.
struct validation_check {

    std::string name;
    Real residual = Real{0};
    Real tolerance = Real{0};
    bool passed = false;
};


// Result of the validation gate. The gate runs once per session, on the same
// binary and flags as the timings it vouches for, at the smallest and the
// largest size in the sweep: a blocked transpose can be correct when the
// whole grid fits in one block and wrong at a block boundary.
//
// Residuals are kept rather than only the pass flag, so drift across commits
// is detectable. No timings are emitted when passed is false.
struct validation_report {

    std::vector<validation_check> checks;
    bool passed = false;
};


// Everything constant across one measurement session. Constructed once at
// startup, held const, and passed by reference to every benchmark. Grouped
// by where the values come from: build_provenance is injected by CMake,
// machine is queried at runtime, and the rest arrives from the harness.
//
// run_id identifies one invocation of the harness script, not one process:
// the memory benchmark spans several processes that must share it, so it is
// passed in on the command line rather than generated internally.
struct run_context {

    std::string run_id;
    std::string output_dir;

    RunProvenance build_provenance;
    machine_info machine;
    rep_policy policy;
    validation_report validation;

    std::uint32_t base_seed = 123456789;

    // FFTW is planned once outside every timed region, with the planner flag
    // and wisdom policy recorded so the comparison against the custom FFT is
    // auditable.
    std::string fftw_planner_flag = "FFTW_MEASURE";
    std::string fftw_wisdom_policy = "none";

    // Free-form notes for things the schema cannot anticipate: ambient
    // conditions, why a sweep was rerun, whether the machine was on mains
    // power. Serialized verbatim into the run metadata.
    std::map<std::string, std::string> notes;
};