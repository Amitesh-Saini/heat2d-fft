// bench_metadata.cpp
// Responsibility:
//   Implements the run metadata writer declared in bench_metadata.hpp.
//
// Layout:
//   Nested objects rather than a flat key list, grouped by where the values
//   come from: build provenance is injected by CMake, machine is queried at
//   runtime, policy and benchmark come from the calling executable. That
//   grouping tells a reader immediately where a missing value should have
//   originated.
//
// Formatting:
//   Written with an indent so the file is readable directly rather than only
//   through a parser. These are small files read by humans as often as by
//   scripts, and a diff between two runs is far more useful when the JSON is
//   pretty-printed.

#include "bench_metadata.hpp"

#include <ctime>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "timing_registry.hpp"

namespace {

// Current UTC time as ISO 8601, matching the format snapshot_writer uses for
// the same purpose in the HDF5 output. UTC so runs on machines in different
// zones stay orderable.
std::string current_utc_timestamp(){

    const std::time_t now = std::time(nullptr);

    std::tm utc_time{};
    gmtime_r(&now, &utc_time);

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc_time);

    return std::string(buffer);
}

} // namespace


void write_run_metadata(const std::string& path, const run_context& context,
    const std::map<std::string, std::string>& benchmark_config){

    nlohmann::json root;

    root["run_id"] = context.run_id;
    root["written_utc"] = current_utc_timestamp();
    root["output_dir"] = context.output_dir;

    // ------------------------------------------------------------------
    // Build provenance, injected by CMake at configure time.
    //
    // The git commit is captured when CMake runs, not when the binary is
    // compiled, so it goes stale if sources change without a reconfigure.
    // That is a known limitation of the current build setup and is recorded
    // in the README rather than papered over here.
    // ------------------------------------------------------------------

    root["build"]["git_commit"] = context.build_provenance.git_commit;
    root["build"]["compiler"] = context.build_provenance.compiler;
    root["build"]["compiler_flags"] = context.build_provenance.compiler_flags;
    root["build"]["build_type"] = context.build_provenance.build_type;

    // An instrumented build is not the same binary as a plain one: the solver
    // carries timing regions that compile to nothing unless
    // HEAT2D_ENABLE_TIMING is defined. Without this flag a reader cannot tell
    // a phase column of zeros from a phase that genuinely took no time.
    root["build"]["timing_instrumentation"] = timing::enabled();

    root["build"]["sizeof_real"] = context.machine.sizeof_real;

    // ------------------------------------------------------------------
    // Machine, queried at runtime.
    // ------------------------------------------------------------------

    root["machine"]["hostname"] = context.machine.hostname;
    root["machine"]["os"] = context.machine.os_name;
    root["machine"]["cpu"] = context.machine.cpu_model;

    root["machine"]["l1d_bytes"] = context.machine.l1d_bytes;
    root["machine"]["l2_bytes"] = context.machine.l2_bytes;

    // Null rather than zero where the hardware has no conventional L3, so a
    // reader can tell "this machine has none" from "the query failed". Apple
    // Silicon is the case that matters here: it has a shared System Level
    // Cache that is not exposed the same way and is not comparable to an x86
    // L3.
    if(context.machine.l3_bytes.has_value()){
        root["machine"]["l3_bytes"] = context.machine.l3_bytes.value();
    }
    else{
        root["machine"]["l3_bytes"] = nullptr;
    }

    // ------------------------------------------------------------------
    // Measurement policy.
    //
    // reps_used appears on every CSV row, but the count is chosen at runtime
    // from a probe, so it cannot be reconstructed from the configuration
    // alone. These are the parameters that produced it.
    // ------------------------------------------------------------------

    root["policy"]["single_call_ns"] = context.policy.single_call_ns;
    root["policy"]["min_timed_ns"] = context.policy.min_timed_ns;
    root["policy"]["max_reps"] = context.policy.max_reps;
    root["policy"]["warmup_reps"] = context.policy.warmup_reps;

    root["policy"]["base_seed"] = context.base_seed;

    // ------------------------------------------------------------------
    // FFTW settings.
    //
    // The planner flag and the wisdom status together determine which
    // decomposition FFTW selected. With FFTW_MEASURE and no wisdom, two
    // sessions can pick different algorithms for the same size and differ by
    // double digits for no visible reason, so knowing which case applied is
    // part of interpreting the numbers.
    // ------------------------------------------------------------------

    root["fftw"]["planner_flag"] = context.fftw_planner_flag;
    root["fftw"]["wisdom_policy"] = context.fftw_wisdom_policy;

    // ------------------------------------------------------------------
    // What this executable swept. Supplied by the caller as text, since the
    // transform and solver runs describe entirely different things.
    // ------------------------------------------------------------------

    root["benchmark"] = nlohmann::json::object();

    for(const auto& entry : benchmark_config){

        root["benchmark"][entry.first] = entry.second;
    }

    // ------------------------------------------------------------------
    // Free-form notes: ambient conditions, why a sweep was rerun, whether
    // anything else was running. The schema cannot anticipate these and they
    // are often what explains an anomaly months later.
    // ------------------------------------------------------------------

    root["notes"] = nlohmann::json::object();

    for(const auto& entry : context.notes){

        root["notes"][entry.first] = entry.second;
    }

    std::ofstream out(path);

    if(!out.is_open()){

        throw std::runtime_error("write_run_metadata: could not open " + path);
    }

    out << root.dump(2) << "\n";

    if(!out){

        throw std::runtime_error("write_run_metadata: could not write " + path);
    }
}