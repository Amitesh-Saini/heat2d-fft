// snapshot_writer.cpp
// Responsibility:
//   Implements SnapshotWriter, the high-level output writer that owns the
//   layout of the consolidated HDF5 run file, and make_run_provenance, which
//   assembles the build/run identity written as root attributes.
//
//   All HDF5 API details live in Hdf5File; this file only decides WHAT is
//   written and in WHAT order:
//
//     construction : /config_json + provenance attributes (first, so even a
//                    crashed run records what was attempted)
//     write_grids  : /x, /y, /times, then the extensible /snapshots dataset
//     append       : one snapshot slab per call, flushed so partial runs
//                    remain readable
//     finalize     : /diagnostics/*, /error/relative_l2, wall-time attribute,
//                    snapshot-count verification

#include "snapshot_writer.hpp"

#include "build_info.hpp"
#include "run_config.hpp"

#include <ctime>
#include <stdexcept>
#include <string>


namespace {

// Current UTC time formatted as ISO 8601, e.g. "2026-07-12T09:41:07Z".
std::string current_utc_timestamp(){

    const std::time_t now = std::time(nullptr);

    std::tm utc_time{};
    gmtime_r(&now, &utc_time);

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc_time);

    return std::string(buffer);
}

} // namespace


RunProvenance make_run_provenance(const RunConfig& config){

    RunProvenance provenance;

    provenance.git_commit = build_info::git_commit;
    provenance.compiler = build_info::compiler;
    provenance.compiler_flags = build_info::compiler_flags;
    provenance.build_type = build_info::build_type;
    provenance.timestamp_utc = current_utc_timestamp();
    provenance.fft_backend = fft_backend_to_string(config.fft_backend);

    return provenance;
}


SnapshotWriter::SnapshotWriter(const RunConfig& config, const RunProvenance& provenance)
    : file_(config.output.output_path, config.output.overwrite), gzip_level_(config.output.gzip_level){

    // file_ has no default constructor and cannot be reassigned, so it must be
    // constructed in the member initializer list above. By the time this body
    // runs, the HDF5 file already exists on disk (or Hdf5File threw).

    // Provenance goes down first, before any grids or snapshots: a run that
    // crashes mid-solve still leaves a file recording what was attempted.
    file_.write_string("/config_json", config.source_json_text);

    file_.write_root_string_attribute("git_commit", provenance.git_commit);
    file_.write_root_string_attribute("compiler", provenance.compiler);
    file_.write_root_string_attribute("compiler_flags", provenance.compiler_flags);
    file_.write_root_string_attribute("build_type", provenance.build_type);
    file_.write_root_string_attribute("timestamp_utc", provenance.timestamp_utc);
    file_.write_root_string_attribute("fft_backend", provenance.fft_backend);

    file_.flush();
}


void SnapshotWriter::write_grids(const RealVec& x, const RealVec& y, const RealVec& times){

    if(grids_written_){

        throw std::runtime_error("SnapshotWriter::write_grids: grids already written");
    }

    file_.write_real_vector("/x", x);
    file_.write_real_vector("/y", y);
    file_.write_real_vector("/times", times);

    file_.create_snapshot_dataset("/snapshots", x.size(), y.size(), gzip_level_);

    expected_snapshots_ = times.size();
    grids_written_ = true;

    file_.flush();
}


void SnapshotWriter::append_snapshot(const Grid2D<Real>& snapshot){

    if(!grids_written_){

        throw std::runtime_error(
            "SnapshotWriter::append_snapshot: write_grids must be called first");
    }

    file_.append_snapshot(snapshot);
    file_.flush();
}


void SnapshotWriter::record_diagnostics(const SnapshotDiagnostics& diagnostics){

    diagnostics_.push_back(diagnostics);
}


void SnapshotWriter::record_analytic_error(Real relative_l2_error){

    analytic_errors_.push_back(relative_l2_error);
}


void SnapshotWriter::finalize(double wall_time_seconds){

    if(finalized_){

        throw std::runtime_error("SnapshotWriter::finalize: finalize already called");
    }

    if(file_.num_snapshots_written() != expected_snapshots_){

        throw std::runtime_error(
            "SnapshotWriter::finalize: wrote " +
            std::to_string(file_.num_snapshots_written()) +
            " snapshots but /times has " +
            std::to_string(expected_snapshots_) + " entries");
    }

    // Buffered diagnostics are row-oriented (one struct per snapshot); HDF5
    // wants column-oriented 1-D datasets, so transpose field by field.
    if(!diagnostics_.empty()){

        const std::size_t n = diagnostics_.size();

        RealVec time_values(n);
        RealVec mean_values(n);
        RealVec l2_values(n);
        RealVec min_values(n);
        RealVec max_values(n);

        for(std::size_t k = 0; k < n; ++k){

            time_values[k] = diagnostics_[k].time;
            mean_values[k] = diagnostics_[k].mean;
            l2_values[k] = diagnostics_[k].l2_norm;
            min_values[k] = diagnostics_[k].min_value;
            max_values[k] = diagnostics_[k].max_value;
        }

        file_.create_group("/diagnostics");
        file_.write_real_vector("/diagnostics/time", time_values);
        file_.write_real_vector("/diagnostics/mean", mean_values);
        file_.write_real_vector("/diagnostics/l2_norm", l2_values);
        file_.write_real_vector("/diagnostics/min", min_values);
        file_.write_real_vector("/diagnostics/max", max_values);
    }

    if(!analytic_errors_.empty()){

        file_.create_group("/error");
        file_.write_real_vector("/error/relative_l2", analytic_errors_);
    }

    file_.write_root_real_attribute("wall_time_seconds", static_cast<Real>(wall_time_seconds));

    file_.flush();

    finalized_ = true;
}