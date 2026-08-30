#pragma once
// bench_common.hpp
// Responsibility:
//   Declare the shared machinery used by every benchmark executable:
//   enum-to-string conversion, seeded input generation, the FFTW reference
//   plans, the adaptive rep policy, the timed regions themselves, and the
//   flop / byte / memory models used to derive rates downstream.
//
//   benchmark_types.hpp describes what ends up in a CSV row. This header
//   describes the machinery that produces one. The small structs declared
//   below are intermediate return values handed to the benchmark files,
//   which attach the trial number and benchmark name and assemble them into
//   Transform_Result / Solver_Result rows; they are not part of the on-disk
//   schema.
//
// Conventions:
//   - Every duration is uint64_t nanoseconds from the steady-clock Timer.
//   - A rep is one transform inside the timed region; a trial is an
//     independent re-measurement of the whole batch. Reps are chosen once
//     per configuration by choose_reps and held constant across that
//     configuration's trials, so the rows are comparable.
//   - Nothing here computes a rate. The flop and byte models return counts;
//     GFLOP/s and GB/s are derived in the analysis layer from those counts,
//     total_time_ns, and reps_used. Storing a rate in C++ would put the
//     convention in two places and hide the division by reps_used.
//   - There is no success flag. An invalid configuration throws, and
//     numerical correctness is the job of the error metrics below plus the
//     validation gate that consumes them.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "benchmark_types.hpp"
#include "grid2d.hpp"
#include "heat2d_fourier.hpp"
#include "run_config.hpp"
#include "types.hpp"
#include <fftw3.h>


// ---------------------------------------------------------------------------
// Enum to string
// ---------------------------------------------------------------------------

// Canonical name of a transform, as emitted in CSV rows and run metadata.
// Single source of truth so the enumerators and the written strings cannot
// drift apart.
std::string transform_to_string(transform transform_type);

// Canonical name of an experiment. This is the column the plotting scripts
// filter on, so experiments sharing a transform still resolve to distinct
// strings.
std::string benchmark_name_to_string(Benchmark_name name);


// ---------------------------------------------------------------------------
// Input generation
// ---------------------------------------------------------------------------

// Derives the per-configuration seed from the session seed and the grid
// shape. Deliberately independent of the transform: FFT and DFT run as
// separate benchmarks, so deriving from the transform would hand them
// different inputs at the same size and their spectra could no longer be
// compared offline. Also constant across a configuration's trials, since the
// error check is computed once per configuration rather than per trial.
std::uint32_t derive_seed(std::uint32_t base_seed, std::size_t nx, std::size_t ny);

// Fills a complex vector with seeded pseudo-random values in [-1, 1].
//
// Values are drawn from mt19937 and scaled by hand rather than through a
// std distribution: the engine is fully specified by the standard, the
// distributions are not, so this is the only way the "seeded inputs are
// reproducible" claim survives moving between libc++ and libstdc++.
// Transform timing is data independent, so the seed matters for reproducing
// a failing error check, not for the timings themselves.
// Divies by gen.max then uses 2x-1 to distribute across -1,1
ComplexVec make_random_input_1d(std::size_t n, std::uint32_t seed);

// As above, for a 2D grid.
Grid2D<Complex> make_random_input_2d(std::size_t nx, std::size_t ny, std::uint32_t seed);


Real relative_linf_error(const ComplexVec& computed, const ComplexVec& reference);
Real relative_linf_error(const Grid2D<Complex>& computed, const Grid2D<Complex>& reference);


// ---------------------------------------------------------------------------
// FFTW reference plans
// ---------------------------------------------------------------------------
// FFTW is both the performance reference and the single correctness
// reference for every transform benchmark. Its execution model differs from
// the custom FFT: a plan is bound to a specific buffer and to its alignment,
// and planning with FFTW_MEASURE is expensive (seconds at large sizes)
// because it runs and times candidate algorithms.
//
// Three consequences drive the design below:
//
//   - A plan is created once per configuration and reused across every trial
//     and rep. It must never be constructed inside a timed region or a trial
//     loop.
//   - FFTW_MEASURE OVERWRITES the buffer during planning. The order is
//     therefore always: construct (allocate + plan), then load(), then
//     execute. Loading before planning silently leaves garbage in the
//     buffer, and the transform still runs, so the mistake produces
//     plausible numbers rather than an error.
//   - The transform is planned IN PLACE, with the same buffer as input and
//     output, so that it matches fft_2d_kernel: one array live, the spectrum
//     overwriting the input. In-place is generally FFTW's slower mode,
//     because it must reorder internally rather than writing linearly to a
//     separate output. That cost is accepted deliberately: the alternative
//     compares against a different algorithm on twice the memory. The README
//     records that the reported fraction-of-FFTW is measured against FFTW's
//     in-place performance, not its best.
//
// load() and store() are the pack/unpack between ComplexVec and FFTW's
// interleaved fftw_complex buffer. They are separate from the execute calls
// precisely so they can sit outside the clock: timing them would charge FFTW
// for O(N) copies the custom FFT never pays.
//
// Alignment asymmetry, recorded in the run metadata and stated in the
// README: this buffer comes from fftw_alloc_complex and is aligned for the
// widest SIMD the build supports, while Grid2D and ComplexVec use the
// default std::vector allocator. This favours FFTW. Removing it would mean
// templating Grid2D on an allocator, which is not worth changing in a tested
// core type.

// Owns the buffer and the forward/inverse plans for one 1D transform size.
// Non-copyable: the plans hold a raw pointer to the buffer.
class Fftw1dPlan {

public:

    // Allocates the buffer and creates the forward and inverse plans with
    // the given planner flag, both in place on that buffer. Throws
    // std::invalid_argument if n is zero or exceeds FFTW's int dimension
    // limit, and std::runtime_error on allocation or plan-creation failure.
    Fftw1dPlan(std::size_t n, unsigned planner_flag);
    ~Fftw1dPlan();

    Fftw1dPlan(const Fftw1dPlan&) = delete;
    Fftw1dPlan& operator=(const Fftw1dPlan&) = delete;

    // Packs input into the working buffer. Call after construction, never
    // before: planning destroys buffer contents.
    void load(const ComplexVec& input);

    // Copies the working buffer out. Before any execute call this returns
    // what load() put in; after execute_forward() it returns the spectrum,
    // since the transform is in place.
    ComplexVec store() const;

    // Runs the pre-made forward plan on the working buffer. This is the only
    // call that belongs inside a timed region.
    void execute_forward();

    // Runs the inverse plan. FFTW's backward transform is unnormalized, so
    // this does not divide by n; apply the normalization outside when a
    // round trip is wanted.
    void execute_inverse();

    // Divides every element of the working buffer by n.
    // FFTW's backward transform is unnormalized, so a forward/inverse pair
    // scales the data by n rather than returning it. The custom
    // ifft_1d_inplace applies its 1/n pass internally as part of the
    // inverse, so calling this immediately after execute_inverse() makes
    // the two inverses equivalent in both result and work done. It belongs
    // inside the timed region for that reason.
    void normalize_inverse();

    std::size_t size() const;
    

private:

    std::size_t n_;
    fftw_complex* buffer_;
    fftw_plan forward_;
    fftw_plan inverse_;
};

// Same contract as Fftw1dPlan, for a 2D transform of shape (nx, ny): one
// in-place buffer, plans created once and reused, load() and store() outside
// any timed region.
//
// fftw_plan_dft_2d(n0, n1, ...) treats n1 as the contiguous dimension, and
// Grid2D maps (i,j) to i*ny + j, so ny is contiguous in both. The two
// layouts already agree: load() and store() are flat copies over raw(), with
// no transpose and no index arithmetic.
class Fftw2dPlan {

public:

    // Allocates the buffer and creates the forward and inverse plans with
    // the given planner flag, both in place on that buffer. Throws
    // std::invalid_argument if either dimension is zero or exceeds FFTW's
    // int dimension limit, and std::runtime_error on allocation or
    // plan-creation failure.
    Fftw2dPlan(std::size_t nx, std::size_t ny, unsigned planner_flag);
    ~Fftw2dPlan();

    Fftw2dPlan(const Fftw2dPlan&) = delete;
    Fftw2dPlan& operator=(const Fftw2dPlan&) = delete;

    // Packs input into the working buffer. Call after construction, never
    // before: planning destroys buffer contents.
    void load(const Grid2D<Complex>& input);

    // Copies the working buffer out. Before any execute call this returns
    // what load() put in; after execute_forward() it returns the spectrum,
    // since the transform is in place.
    Grid2D<Complex> store() const;

    // Runs the pre-made forward plan on the working buffer. This is the only
    // call that belongs inside a timed region.
    void execute_forward();

    // Runs the inverse plan. FFTW's backward transform is unnormalized, so
    // this does not divide by nx*ny; apply the normalization outside when a
    // round trip is wanted.
    void execute_inverse();

    // Divides every element of the working buffer by nx*ny.
    // FFTW's backward transform is unnormalized, so a forward/inverse pair
    // scales the data by nx*ny rather than returning it. The custom
    // ifft_2d_inplace applies its 1/(nx*ny) pass internally as part of the
    // inverse, so calling this immediately after execute_inverse() makes
    // the two inverses equivalent in both result and work done. It belongs
    // inside the timed region for that reason.
    void normalize_inverse();

    // Direct access to the working buffer, viewed as Complex.
    //
    // std::complex<double> is guaranteed by the standard to have the layout
    // of double[2] with the real part first, which is exactly fftw_complex,
    // so this is a well-defined view rather than a reinterpretation trick.
    // FFTW's own documentation recommends it for C++ interoperation.
    //
    // This exists so a pointwise pass can operate on the buffer in place,
    // without a load/store round trip. The solver's decay pass is the case
    // that needs it: an FFTW-backed solver would keep its data in FFTW's
    // buffer across the whole solve, so charging the benchmark for copies it
    // would never make would misreport the backend rather than measure it.
    //
    // The buffer holds nx*ny elements in the same row-major order as Grid2D,
    // so element (i,j) is at index i*ny + j.
    Complex* data();
    const Complex* data() const;

    std::size_t nx() const;
    std::size_t ny() const;


private:

    std::size_t nx_;
    std::size_t ny_;
    fftw_complex* buffer_;
    fftw_plan forward_;
    fftw_plan inverse_;
};


// FFTW_MEASURE selects its algorithm by timing candidates on the day, so two
// sessions can pick different plans for the same size and differ by double
// digit percentages for no visible reason. Exporting wisdom after the first
// session and importing it on later ones pins the choice.
//
// import_fftw_wisdom returns false when the file is absent or unreadable,
// which is not an error: the caller records in the run metadata whether the
// session ran on imported or freshly generated wisdom.
bool import_fftw_wisdom(const std::string& path);
void export_fftw_wisdom(const std::string& path);


// ---------------------------------------------------------------------------
// Rep policy
// ---------------------------------------------------------------------------

// Chooses the rep count for one configuration from a single untimed probe
// transform.
//
// Returns 1 when a single transform already exceeds policy.single_call_ns.
// That covers the whole solver benchmark and the upper half of the transform
// sweeps: repetition is then carried entirely by trials, each measurement
// transforms a fresh buffer once, and there is nothing to restore. Below the
// threshold it returns the smallest count whose batch clears
// policy.min_timed_ns, capped at policy.max_reps.
//
// The result is used for every trial of the configuration, so all its rows
// share one rep count and remain comparable.
std::size_t choose_reps(std::uint64_t single_transform_ns, const rep_policy& policy);


// ---------------------------------------------------------------------------
// Timed regions: transforms
// ---------------------------------------------------------------------------

// One timed batch of a 1D transform.
//
// paired records how the batch was run. With reps == 1 the input is
// transformed once from a fresh buffer and paired is false. With reps > 1
// the batch alternates forward and inverse: because the normalization lives
// in the inverse, each pair returns the buffer exactly to its original
// state, so no restoring copy is needed inside the timed region. The cost is
// that such a row is a forward/inverse average rather than a forward, which
// is why the flag is carried through to the CSV rather than hidden.
struct timed_batch {

    std::uint64_t total_time_ns = 0;
    std::size_t reps_used = 0;
    bool paired = false;
};


// One timed batch of a 2D transform, with the row and column passes timed
// separately inside it.
//
// total_time_ns is measured as its own region rather than summed from the
// passes, so total minus row minus column is a visible residual covering bit
// reversal, the column gather/scatter, and any allocation the passes do not
// account for. A large residual is a finding about where the time goes, not
// an error.
struct timed_batch_2d {

    std::uint64_t total_time_ns = 0;
    std::uint64_t row_time_ns = 0;
    std::uint64_t col_time_ns = 0;
    std::size_t reps_used = 0;
    bool paired = false;
};


// Times reps transforms of input using the custom FFT or the naive DFT.
// input is not modified; the working buffer is held internally.
timed_batch time_transform_1d(const ComplexVec& input, transform transform_type, std::size_t reps);

// Times reps executions of a pre-made FFTW plan. The plan must already be
// loaded: load() is deliberately outside the timed region so FFTW is not
// charged for pack/unpack work the custom FFT does not do.
timed_batch time_fftw_1d(Fftw1dPlan& plan, std::size_t reps);

// Times reps 2D transforms of grid, and separately the row and column passes
// that compose them.
timed_batch_2d time_transform_2d(const Grid2D<Complex>& grid, transform transform_type, std::size_t reps);

// Times reps executions of a pre-made, pre-loaded 2D FFTW plan. FFTW does
// not expose its internal pass structure, so row and column times have no
// meaning for these rows and are left absent.
timed_batch time_fftw_2d(Fftw2dPlan& plan, std::size_t reps);

 
// Phase breakdown of one solve. The phase times decompose solve_time_ns, but
// do not sum to it: the wavenumber grid construction, the physical-to-complex
// conversion, the real-part extraction after each inverse, and the copies
// into the snapshot vector are all left out, for the same reason the flop
// model excludes them. The difference is a visible residual, and a large one
// at the smaller grid sizes is setup overhead showing up.
//
// spectral_copy_time_ns exists because the initial spectrum is needed again
// for every later output time, so it is copied before the decay factor
// overwrites it. That copy is neither decay nor inverse transform, and
// folding it into the decay pass would overstate the cost of the
// exponentials, which is the quantity the profile benchmark exists to
// measure.
//
// decay_time_ns and inverse_transform_time_ns are summed over all snapshots;
// forward_transform_time_ns is a single transform of the initial condition.
struct solver_timing {
 
    std::uint64_t solve_time_ns = 0;
 
    std::uint64_t forward_transform_time_ns = 0;
    std::uint64_t spectral_copy_time_ns = 0;
    std::uint64_t decay_time_ns = 0;
    std::uint64_t inverse_transform_time_ns = 0;
};
 
 
// Runs one solve from the given initial condition, returns its total time and
// phase breakdown, and hands back every snapshot so the caller can pass them
// to time_snapshot_write.
//
// The phase times are not measured here. They are read out of the timing
// registry, which heat2d_fourier.cpp populates from regions annotated inside
// the real solve. Reimplementing the loop here with its own clocks would
// duplicate solver logic, and the copy would drift the moment one side is
// optimized and the other is not, leaving the benchmark reporting the
// performance of code nobody runs.
//
// The phases come back as zero when the build does not define
// HEAT2D_ENABLE_TIMING. That is a legitimate state, not an error: the caller
// records timing::enabled() in the run metadata, since an instrumented build
// is not the same binary as a plain one.
//
// Reps are always 1: a solve at any benchmarked grid size is far above the
// timing floor, so repetition is carried by trials alone.
//
// Custom radix-2 backend only. Heat2DFourierSolver has no FFTW path to
// instrument, and writing one out here by hand would reintroduce exactly the
// duplication the registry removes. When the solver gains a real backend
// switch, the annotations already in place cover both paths and this function
// does not change at all.
solver_timing time_solve(const Grid2D<Real>& initial_condition, const Heat2DConfig& solver_config,
    std::vector<Grid2D<Real>>& snapshots_out);

// Result of writing one run's snapshots.
//
// io_time_ns is the total and includes finalize(): an HDF5 file that has not
// been finalized has no diagnostics datasets and no guaranteed flush, so a
// number that excluded it would not correspond to a usable result.
//
// finalize_time_ns is that closing cost on its own, so the breakdown answers
// how much of the I/O is the per-snapshot appends and how much is the final
// flush. Appends are io_time_ns minus finalize_time_ns.
//
// The times include gzip compression, which is why the benchmark sweeps
// gzip_level: on a smooth field the CPU cost of compressing can rival the
// write itself. SnapshotWriter also flushes after every append, so the cost
// scales with snapshot count in a way a single bulk write would not.
//
// These are page-cache write-back numbers on a warm system, not device
// throughput. Every configuration writes to a fresh path on the same
// filesystem.
//
// bytes_written is the on-disk size after the file is closed, so MB/s and the
// compression ratio are both derivable downstream.
struct io_timing {
 
    std::uint64_t io_time_ns = 0;
    std::uint64_t finalize_time_ns = 0;
    std::uint64_t bytes_written = 0;
};


// Writes the snapshots to an HDF5 file at output_path through the same
// SnapshotWriter the production driver uses, and times it.
io_timing time_snapshot_write(const std::vector<Grid2D<Real>>& snapshots, const RealVec& x, const RealVec& y,
    const RealVec& times, const std::string& output_path, int gzip_level);

// ---------------------------------------------------------------------------
// Flop models
// ---------------------------------------------------------------------------
// Flop counts are analytic, by convention, and never measured with hardware
// counters. Both the custom FFT and FFTW are charged the same 5*N*log2(N) so
// the comparison reflects which is faster rather than which avoids more
// arithmetic; the naive DFT is charged 8*N^2 for its complex
// multiply-accumulates.
//
// The resulting rate is meaningful only between implementations of the same
// algorithm: custom FFT against FFTW, row pass against column pass, solver
// before and after an optimization. The DFT does vastly more arithmetic with
// a perfectly predictable access pattern and will report a HIGHER rate than
// the FFT while being orders of magnitude slower, so the two must never
// appear on the same axes.

// Flops for one transform of the given shape. ny is 1 for 1D transforms.
// 1D FFT/FFTW: 5*n*log2(n). 1D DFT: 8*n^2. 2D separable: the ny row
// transforms of length nx plus the nx column transforms of length ny.
double transform_flops(std::size_t nx, std::size_t ny, transform transform_type);

// Flops for one full solve: one forward 2D transform, then per snapshot one
// inverse 2D transform plus the pointwise decay application at 2 flops per
// grid point. Note the constuction of the wavenumbers is not being recorded here
// the wavenumber constuction uses order of magnitudes less flops then the main numeric suite - noise 
double solver_flops(std::size_t nx, std::size_t ny, std::size_t num_snapshots);

// Bytes of memory traffic moved by the decay pass across all snapshots.
//
// The pass does two flops per sixteen-byte complex element, so it is
// bandwidth bound by construction. Reporting it as GFLOP/s produces a tiny
// number that looks like poor performance when the pass is simply memory
// bound; GB/s against measured STREAM bandwidth is the honest metric and
// answers the question the decay benchmark exists to ask, which is whether
// the exponentials or the memory traffic dominate.
std::uint64_t decay_pass_bytes(std::size_t nx, std::size_t ny, std::size_t num_snapshots);


// ---------------------------------------------------------------------------
// Memory models
// ---------------------------------------------------------------------------
// Array working set only: element size times point count times the number of
// simultaneously live grids. This is the headline memory figure and the
// x-axis of the cache-cliff plot, so the live-array count must be honest:
// assuming one array when the transform holds two moves the cliff by a
// factor of two.
//
// Element size comes from sizeof(Complex) rather than a hardcoded 16, so the
// models stay correct if Real ever changes.

// Working set of one transform, including any scratch the implementation
// holds live at once.
std::uint64_t transform_working_set_bytes(std::size_t nx, std::size_t ny, transform transform_type);

// Working set of one solve. Includes the snapshots, because solve() returns
// every snapshot at once: the solver's footprint scales with snapshot count
// as well as with grid size, so a memory sweep that varies both at once
// measures nothing.
std::uint64_t solver_working_set_bytes(std::size_t nx, std::size_t ny, std::size_t num_snapshots);


// ---------------------------------------------------------------------------
// Error metrics
// ---------------------------------------------------------------------------
// Computed once per configuration, outside every timed region, and compared
// against error_tolerance below. FFTW is the single reference: comparing the
// custom FFT and the naive DFT against an independent implementation is
// stronger evidence than comparing two of your own implementations against
// each other.

// Forward then inverse against the original input, relative L-infinity.
//
// This is what validates the 1/(nx*ny) normalization: the factor lives
// entirely in the inverse, so a forward-only comparison against FFTW never
// exercises it. Round trip is therefore not redundant with the FFTW check.
Real round_trip_error_2d(const Grid2D<Complex>& input);


// Forward then inverse against the original input, relative L-infinity.
Real round_trip_error_1d(const ComplexVec& input);

// Tolerance for the checks above.
//
// Roundoff grows with problem size and at different rates per algorithm:
// roughly eps*log2(N) for an FFT and eps*N for a naive DFT. A single
// constant threshold is therefore simultaneously too loose at small sizes,
// masking real bugs, and too strict at large ones, failing correct code: at
// n = 4096 a naive DFT's expected relative error is already near 1e-12.
//
// safety_factor is the multiplier on the model, recorded in the run metadata
// so the strictness of the gate is part of the run's provenance.
Real error_tolerance(std::size_t nx, std::size_t ny, transform transform_type, Real safety_factor);