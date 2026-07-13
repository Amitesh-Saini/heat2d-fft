#pragma once
// snapshot_writer.hpp
// Responsibility:
//   Declare the high-level output writer for complete simulation runs.
//
//   The snapshot writer owns the layout of the single consolidated HDF5
//   output file. It saves everything needed to inspect, reproduce, and
//   visualize a simulation:
//
//     /x                x-coordinates, shape (nx,)
//     /y                y-coordinates, shape (ny,)
//     /times            snapshot times, shape (nt,)
//     /snapshots        fields, shape (nt, nx, ny), chunked + compressed
//     /config_json      verbatim input JSON text
//     /diagnostics/*    per-snapshot mean, l2_norm, min, max (if enabled)
//     /error/relative_l2  analytic error (Fourier validation runs only)
//     root attributes   provenance: git commit, compiler, flags, build type,
//                       timestamp, fft backend, wall time
//
//   This module keeps output organization separate from main.cpp and from
//   the numerical solver. It uses Hdf5File (hdf5_writer.hpp) as its storage
//   backend and knows nothing about HDF5 API details.
//
// Streaming design:
//   append_snapshot takes one grid at a time, in time order. For v1, main
//   loops over the vector returned by Heat2DFourierSolver::solve(); if the
//   solver later gains a per-snapshot callback overload, this interface is
//   unchanged and memory stays flat at one snapshot.

#include <string>
#include <vector>

#include "diagnostics.hpp"
#include "grid2d.hpp"
#include "hdf5_writer.hpp"
#include "run_config.hpp"
#include "types.hpp"

// Build/run provenance recorded as root attributes of the output file.
// Together with /config_json this makes any output file exactly
// reproducible from the file alone.
struct RunProvenance {
    std::string git_commit;      // from CMake-injected build_info
    std::string compiler;        // compiler id + version
    std::string compiler_flags;
    std::string build_type;      // e.g. "Release"
    std::string timestamp_utc;   // run start, ISO 8601
    std::string fft_backend;     // "custom" or "fftw"
};

// Collects provenance from the CMake-generated build_info header, the
// current wall-clock time, and the configured FFT backend.
RunProvenance make_run_provenance(const RunConfig& config);


class SnapshotWriter{
    
public:
    // Creates the output file (honoring config.output.overwrite) and
    // immediately writes /config_json and the provenance attributes, so
    // even a run that later crashes records what was attempted.
    // Throws std::runtime_error on file-creation failure.
    SnapshotWriter(const RunConfig& config, const RunProvenance& provenance);

    // Writes /x, /y, /times and creates the extensible /snapshots dataset
    // sized (0, x.size(), y.size()). Must be called exactly once, before
    // the first append_snapshot. times.size() fixes the expected snapshot
    // count checked in finalize.
    void write_grids(const RealVec& x, const RealVec& y, const RealVec& times);

    // Appends one physical-space snapshot. Snapshots must arrive in the
    // same order as /times. Flushes after each append so partial runs
    // remain readable.
    void append_snapshot(const Grid2D<Real>& snapshot);

    // Buffers per-snapshot diagnostics, written as
    // /diagnostics/{time, mean, l2_norm, min, max} during finalize.
    void record_diagnostics(const SnapshotDiagnostics& diagnostics);

    // Buffers the per-snapshot analytic relative L2 error (Fourier-mode
    // validation runs), written as /error/relative_l2 during finalize.
    void record_analytic_error(Real relative_l2_error);

    // Writes buffered diagnostics/error datasets, records the total wall
    // time as a root attribute, verifies the appended snapshot count
    // matches /times, and flushes. Must be called exactly once; the
    // destructor closes the file but does not substitute for finalize.
    void finalize(double wall_time_seconds);

    

private:
    Hdf5File file_;
    int gzip_level_ = 4;
    std::size_t expected_snapshots_ = 0;
    std::vector<SnapshotDiagnostics> diagnostics_;
    RealVec analytic_errors_;
    bool grids_written_ = false;
    bool finalized_ = false;
};