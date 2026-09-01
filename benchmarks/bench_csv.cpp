// bench_csv.cpp
// Responsibility:
//   Implements the streaming CSV writers declared in bench_csv.hpp.
//
// Implementation notes:
//   - Each constructor opens its file, sets floating-point precision, and
//     emits the header immediately, so a file that exists always has a header
//     even if the session dies before the first row.
//   - Every write flushes. The cost is negligible against a measurement
//     lasting tens of milliseconds at minimum, and it is what makes a
//     partially completed sweep readable.
//   - Absent optionals write an empty field rather than a zero, so the
//     analysis layer sees NaN rather than a value that looks measured.

#include "bench_csv.hpp"

#include <iomanip>
#include <limits>
#include <stdexcept>
#include <filesystem>

#include "run_config.hpp"


// ---------------------------------------------------------------------------
// TransformCsvWriter
// ---------------------------------------------------------------------------

TransformCsvWriter::TransformCsvWriter(const std::string& path, const run_context& context)
    : out_(path), context_(context){

    if(!out_.is_open()){

        throw std::runtime_error("TransformCsvWriter: failed to open file: " + path);
    }

    // max_digits10 is the number of decimal digits needed for a double to
    // round-trip exactly. The stream default of six would collapse 3.6e-15
    // and 3.6000001e-15 onto the same text, which would hide exactly the
    // drift the error column exists to expose.
    out_ << std::setprecision(std::numeric_limits<double>::max_digits10);

    out_ << "run_id,"
         << "benchmark,"
         << "transform,"
         << "nx,"
         << "ny,"
         << "trial,"
         << "reps_used,"
         << "paired,"
         << "total_time_ns,"
         << "row_time_ns,"
         << "col_time_ns,"
         << "error,"
         << "roundtrip_error\n";;

    out_.flush();
}


void TransformCsvWriter::write(const Transform_Result& result){

    out_ << context_.run_id << ','
         << benchmark_name_to_string(result.name) << ','
         << transform_to_string(result.transform_type) << ','
         << result.nx << ','
         << result.ny << ','
         << result.trial << ','
         << result.reps_used << ','
         << (result.paired ? 1 : 0) << ','
         << result.total_time_ns << ',';

    // Empty rather than zero: these are absent for 1D rows and for FFTW rows,
    // which expose no pass structure. A zero here would be indistinguishable
    // from a pass that genuinely took no time.
    if(result.row_time_ns.has_value()){
        out_ << result.row_time_ns.value();
    }

    out_ << ',';

    if(result.col_time_ns.has_value()){
        out_ << result.col_time_ns.value();
    }

    out_ << ',' << result.error
         << ',' << result.roundtrip_error << '\n';

    // Flushed per row so a session that dies partway leaves every earlier
    // measurement readable.
    out_.flush();
}


// ---------------------------------------------------------------------------
// SolverCsvWriter
// ---------------------------------------------------------------------------

SolverCsvWriter::SolverCsvWriter(const std::string& path, const run_context& context)
    : out_(path), context_(context){

    if(!out_.is_open()){

        throw std::runtime_error("SolverCsvWriter: failed to open file: " + path);
    }

    out_ << std::setprecision(std::numeric_limits<double>::max_digits10);

    out_ << "run_id,"
         << "benchmark,"
         << "backend,"
         << "ic,"
         << "nx,"
         << "ny,"
         << "trial,"
         << "reps_used,"
         << "total_time_ns,"
         << "forward_transform_time_ns,"
         << "spectral_copy_time_ns,"
         << "decay_time_ns,"
         << "inverse_transform_time_ns,"
         << "io_time_ns,"
         << "finalize_time_ns,"
         << "bytes_written,"
         << "gzip_level,"
         << "error\n";

    out_.flush();
}


void SolverCsvWriter::write(const Solver_Result& result){

    out_ << context_.run_id << ','
         << benchmark_name_to_string(result.name) << ','
         << fft_backend_to_string(result.fft_backend) << ','
         << result.ic_name << ','
         << result.nx << ','
         << result.ny << ','
         << result.trial << ','
         << result.reps_used << ','
         << result.total_time_ns << ',';

    // The phase columns are not optional. They come from the timing registry
    // and read as zero in a build without HEAT2D_ENABLE_TIMING, which is a
    // property of the run rather than of the row: the metadata records
    // whether the build was instrumented, so a reader can tell a zero here
    // from an absent measurement.
    out_ << result.forward_transform_time_ns << ','
         << result.spectral_copy_time_ns << ','
         << result.decay_time_ns << ','
         << result.inverse_transform_time_ns << ',';

    // Empty on rows from runs that wrote no output.
    if(result.io_time_ns.has_value()){
        out_ << result.io_time_ns.value();
    }

    out_ << ',';

    if(result.finalize_time_ns.has_value()){
        out_ << result.finalize_time_ns.value();
    }

    out_ << ',';

    if(result.bytes_written.has_value()){
        out_ << result.bytes_written.value();
    }

    out_ << ',';

    if(result.gzip_level.has_value()){
        out_ << result.gzip_level.value();
    }
 
    out_ << ',';

    // Present only for Fourier-mode initial conditions, the only ones with a
    // closed form to compare against.
    if(result.error.has_value()){
        out_ << result.error.value();
    }

    out_ << '\n';

    out_.flush();
}


// ---------------------------------------------------------------------------
// MemoryCsvWriter
// ---------------------------------------------------------------------------

MemoryCsvWriter::MemoryCsvWriter(const std::string& path, const run_context& context)
    : context_(context){
 
    // Checked before opening: opening in append mode creates the file, after
    // which the existence test would always say yes and the header would
    // never be written.
    const bool existed = std::filesystem::exists(path);
 
    out_.open(path, std::ios::app);
 
    if(!out_.is_open()){
 
        throw std::runtime_error("MemoryCsvWriter: failed to open file: " + path);
    }
 
    out_ << std::setprecision(std::numeric_limits<double>::max_digits10);
 
    if(!existed){
 
        out_ << "run_id,"
             << "benchmark,"
             << "nx,"
             << "ny,"
             << "num_snapshots,"
             << "theoretical_bytes,"
             << "baseline_rss_bytes,"
             << "peak_rss_bytes\n";
 
        out_.flush();
    }
}
 
 
void MemoryCsvWriter::write(const Memory_Result& result){
 
    // No derived columns. The ratio of measured to theoretical is the point
    // of this benchmark, but it is computed downstream from the raw numbers
    // rather than stored, so the definition lives in one place.
    out_ << context_.run_id << ','
         << benchmark_name_to_string(result.name) << ','
         << result.nx << ','
         << result.ny << ','
         << result.num_snapshots << ','
         << result.theoretical_bytes << ',';
 
    if(result.baseline_rss_bytes.has_value()){
        out_ << result.baseline_rss_bytes.value();
    }
 
    out_ << ',';
 
    if(result.peak_rss_bytes.has_value()){
        out_ << result.peak_rss_bytes.value();
    }
 
    out_ << '\n';
 
    out_.flush();
}