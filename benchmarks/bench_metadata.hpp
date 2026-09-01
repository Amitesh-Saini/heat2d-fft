#pragma once
// bench_metadata.hpp
// Responsibility:
//   Write the sidecar JSON that describes one benchmark run: what machine it
//   ran on, what binary produced it, and what was measured.
//
//   The CSVs carry one row per measurement and a run_id column. This file is
//   what that run_id points at. Splitting the two keeps the rows narrow,
//   since everything here is constant across a session and duplicating it on
//   every row would add fifteen unchanging strings to thousands of lines.
//
//   The split is also the one Caliper and Adiak make in production HPC
//   codes: performance data in one place, run metadata in another, joined by
//   an identifier. A timing number without the metadata is not a result, it
//   is just a number.
//
// What belongs here:
//   Anything constant for the whole session. Build provenance, the machine
//   description, the rep policy, the seed, the FFTW planner settings, whether
//   the build was instrumented, and the configuration lists that say what was
//   swept. Anything that varies between measurements belongs in the CSV.
//
// Configuration lists:
//   Passed as a string map rather than a typed block, because the transform
//   and solver executables sweep completely different things: size ladders
//   and a tolerance factor on one side, grid sizes and output times and
//   compression levels on the other. A typed field for each would make this
//   file depend on both benchmarks' specifics for no gain, since JSON values
//   are text either way.

#include <map>
#include <string>

#include "benchmark_types.hpp"


// Writes the metadata for one run to path, overwriting any existing file.
//
// benchmark_config carries whatever the calling executable swept: sizes,
// trials, tolerances, output times. The caller formats the values; this
// function only records them.
//
// Throws std::runtime_error if the file cannot be written. A run whose
// metadata is missing is not reproducible, so a failure here is worth
// surfacing rather than leaving the CSVs orphaned.
void write_run_metadata(const std::string& path, const run_context& context,
    const std::map<std::string, std::string>& benchmark_config);