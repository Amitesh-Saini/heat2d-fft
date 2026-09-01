#pragma once
// bench_csv.hpp
// Responsibility:
//   Declare the streaming CSV writers that turn benchmark result structs into
//   rows on disk.
//
//   One writer class per schema, each owning an open output stream for the
//   lifetime of a session. Rows are appended and flushed as they are produced
//   rather than collected in a vector and written at the end: a solver sweep
//   at the larger grid sizes runs for minutes per configuration, and a
//   session that dies at the last one should still leave every measurement
//   before it readable. This is the same reasoning that makes SnapshotWriter
//   flush after every snapshot.
//
// Schema layout:
//   Four files, one per column set, not one per benchmark. Splitting by
//   benchmark name would produce a dozen files with identical headers, and
//   every cross-benchmark figure (FFT against DFT, fraction of FFTW) would
//   have to concatenate them back together. Rows carry a benchmark column and
//   the analysis layer filters on it.
//
// Conventions:
//   - One row per (configuration, trial). Nothing is aggregated here.
//     Medians, spreads, GFLOP/s, GB/s and every normalization are derived
//     downstream in Python from the raw rows.
//   - run_id is stamped from the run_context rather than carried on the
//     result structs, so it cannot be forgotten at a construction site.
//   - An absent optional writes an EMPTY field, never a zero. Zero is a
//     plausible measurement and would silently pollute any aggregate that did
//     not special-case it; an empty field parses as NaN and propagates
//     correctly through means and plots.
//   - Floating-point values are written at max_digits10 so they round-trip
//     exactly. The stream default of six significant digits would render
//     3.6e-15 and 3.6000001e-15 identically, which would destroy the drift
//     detection the error column exists for.
//   - Construction throws if the file cannot be opened. A benchmark that runs
//     for an hour and only then discovers it cannot write is worse than one
//     that fails immediately.
//
// Not here:
//   The run metadata JSON, which holds provenance, machine description, rep
//   policy and validation results. It has a different lifetime, format and
//   dependencies, and lives in bench_metadata. Keeping performance data and
//   run metadata in separate files is the same split Caliper and Adiak make.

#include <fstream>
#include <string>

#include "benchmark_types.hpp"
#include "bench_common.hpp"


// ---------------------------------------------------------------------------
// Transform rows
// ---------------------------------------------------------------------------

// One row per timed batch of a 1D or 2D transform.
//
// Columns:
//   run_id, benchmark, transform, nx, ny, trial, reps_used, paired,
//   total_time_ns, row_time_ns, col_time_ns, error
//
// row_time_ns and col_time_ns are empty for every 1D row and for FFTW rows,
// which have no exposed pass structure. paired records whether the batch
// alternated forward and inverse, which makes the row a forward/inverse
// average rather than a forward.
class TransformCsvWriter {

public:

    // Opens path for writing and emits the header. Throws std::runtime_error
    // if the file cannot be opened.
    TransformCsvWriter(const std::string& path, const run_context& context);

    TransformCsvWriter(const TransformCsvWriter&) = delete;
    TransformCsvWriter& operator=(const TransformCsvWriter&) = delete;

    // Appends one row and flushes.
    void write(const Transform_Result& result);

private:

    std::ofstream out_;
    const run_context& context_;
};


// ---------------------------------------------------------------------------
// Solver rows
// ---------------------------------------------------------------------------

// One row per solve.
//
// Columns:
//   run_id, benchmark, backend, ic, nx, ny, trial, reps_used,
//   total_time_ns, forward_transform_time_ns, spectral_copy_time_ns,
//   decay_time_ns, inverse_transform_time_ns, io_time_ns,
//   finalize_time_ns, bytes_written, error
//
// The phase columns come from the timing registry inside the solver and read
// as zero in a build without HEAT2D_ENABLE_TIMING, which the run metadata
// records. The I/O columns are empty on rows from runs that wrote no output.
// error is present only for Fourier-mode initial conditions, the only ones
// with a closed form to compare against.
//
// The phases deliberately do not sum to total_time_ns: the wavenumber grid,
// the physical-to-complex conversion, the real-part extraction and the
// snapshot copies are all outside the annotated regions. The residual is
// visible on purpose.
class SolverCsvWriter {

public:

    SolverCsvWriter(const std::string& path, const run_context& context);

    SolverCsvWriter(const SolverCsvWriter&) = delete;
    SolverCsvWriter& operator=(const SolverCsvWriter&) = delete;

    void write(const Solver_Result& result);

private:

    std::ofstream out_;
    const run_context& context_;
};


// ---------------------------------------------------------------------------
// Memory rows
// ---------------------------------------------------------------------------

// One row per grid size.
//
// Columns:
//   run_id, benchmark, nx, ny, num_snapshots, theoretical_bytes,
//   baseline_rss_bytes, peak_rss_bytes
//
// Unlike the other two writers this one opens in APPEND mode and emits the
// header only when the file does not already exist.
//
// Peak resident set size is a whole-process high-water mark and never
// decreases, so a single process sweeping a size ladder would report the
// largest configuration's footprint on every row. The memory benchmark is
// therefore invoked once per configuration by the harness script, and each
// invocation contributes one row to a file the previous invocation created.
// Truncating on open would leave only the last row.
class MemoryCsvWriter {
 
public:
 
    // Opens path for appending. Writes the header only if the file did not
    // exist, so repeated invocations accumulate rows under one header. The
    // harness deletes the file before the first invocation of a session, so a
    // fresh run does not pile onto the previous one.
    MemoryCsvWriter(const std::string& path, const run_context& context);
 
    MemoryCsvWriter(const MemoryCsvWriter&) = delete;
    MemoryCsvWriter& operator=(const MemoryCsvWriter&) = delete;
 
    void write(const Memory_Result& result);
 
private:
 
    std::ofstream out_;
    const run_context& context_;
};