#pragma once
// config_io.hpp
// Responsibility:
//   Declare functions for reading simulation configuration files.
//
//   The main executable is config-file driven rather than interactive:
//
//       ./heat2d configs/gaussian_demo.json
//
//   This module converts JSON configuration files into RunConfig objects.
//   Keeping JSON parsing here prevents main.cpp from being filled with file
//   I/O and parsing details, and keeps nlohmann/json out of every other
//   translation unit.
//
// Parsing policy (strict):
//   - schema_version is required and must equal current_schema_version.
//   - Required keys must be present; missing keys are errors, not defaults,
//     except where a default is explicitly documented in run_config.hpp.
//   - Unknown keys are rejected at every level (catches typos like "alhpa"
//     instead of silently falling back to a default).
//   - Wrong JSON types (e.g. string where number expected) are errors.
//   - All errors throw with a message naming the offending key.

#include <string>

#include "run_config.hpp"

// Reads an entire text file into a string.
// Throws std::runtime_error if the file cannot be opened or read.
std::string read_text_file(const std::string& path);

// Parses JSON text into a RunConfig, applying the strict parsing policy
// above, then runs validate_run_config on the result. Stores the verbatim
// input text in RunConfig::source_json_text.
//
// Separated from file loading so parsing can be unit-tested on in-memory
// strings without touching the filesystem.
//
// Throws std::invalid_argument (bad content) or exceptions derived from
// nlohmann::json::exception (malformed JSON).
RunConfig parse_run_config_json(const std::string& json_text);

// Convenience wrapper: read_text_file + parse_run_config_json.
RunConfig load_run_config_from_json(const std::string& path);

// Semantic validation beyond JSON shape. Checks include:
//   - grid/domain sanity via validate_grid_spec_2d (IC-layer rules),
//   - alpha finite and positive,
//   - time specification sanity (delegates to time_grid validation after
//     expansion; rejects t_end < t_start, num_snapshots == 0, negative or
//     unsorted explicit times),
//   - output_path nonempty, gzip_level in [0, 9],
//   - compute_analytic_error only allowed with Fourier-mode ICs,
//   - Fourier-mode wavevectors within the Nyquist range for (nx, ny).
//
// Power-of-two grid checks remain the solver's responsibility
// (Heat2DFourierSolver::validate_config), but are also performed here when
// fft_backend == FftBackend::custom, so a bad config fails at parse time
// rather than after IC construction.
//
// Throws std::invalid_argument on the first violation found.
void validate_run_config(const RunConfig& config);