#pragma once
// run_config.hpp
// Responsibility:
//   Define the high-level configuration for a complete heat-equation run.
//
//   Heat2DConfig (heat2d_fourier.hpp) stores only the numerical solver
//   settings. RunConfig stores everything main.cpp needs to describe one
//   complete, reproducible simulation run:
//     - schema version,
//     - the verbatim input JSON text (for provenance storage),
//     - solver/domain settings, later converted into Heat2DConfig,
//     - the FFT backend choice,
//     - the initial-condition selection and its parameters (std::variant),
//     - output settings (path, overwrite behavior, compression),
//     - diagnostics settings.
//
//   RunConfig is a plain data object. JSON parsing/validation lives in
//   config_io; expanding the time specification lives in time_grid.

#include <cstddef>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "grid2d.hpp"
#include "heat2d_fourier.hpp"
#include "initial_conditions.hpp"
#include "types.hpp"

// Schema version understood by this build. config_io rejects configuration
// files whose "schema_version" field does not match this value.
inline constexpr int current_schema_version = 1;


// ---------------------------------------------------------------------------
// FFT backend selection
// ---------------------------------------------------------------------------

// Which FFT implementation the solver pipeline should use.
//   custom : the from-scratch radix-2 FFT (fft2d.hpp)
//   fftw   : the FFTW backend (validation / performance reference)
enum class FftBackend {
    custom,
    fftw
};

// Converts between FftBackend and the canonical config-file strings
// "custom" and "fftw". fft_backend_from_string throws std::invalid_argument
// for unrecognized names.
std::string fft_backend_to_string(FftBackend backend);
FftBackend fft_backend_from_string(const std::string& name);


// ---------------------------------------------------------------------------
// Initial-condition selection
// ---------------------------------------------------------------------------
// One parameter struct per generator in initial_conditions.hpp. Defaults
// mirror the generator defaults, so an empty JSON parameter block selects
// the same field the C++ defaults would produce.

struct GaussianIcParams {
    Real amplitude = Real{1};
    std::optional<Real> sigma;              // default: 0.10 * min(Lx, Ly)
    std::size_t image_radius_x = 1;
    std::size_t image_radius_y = 1;
};

struct HotSquareIcParams {
    Real amplitude = Real{1};
    std::optional<Real> width_x;            // default: 0.20 * Lx
    std::optional<Real> width_y;            // default: 0.20 * Ly
    std::optional<Real> smooth_width_x;     // default: min(3*dx, 0.10*width_x)
    std::optional<Real> smooth_width_y;     // default: min(3*dy, 0.10*width_y)
};

struct ConstantIcParams {
    Real T0 = Real{0.5};
};

struct SingleFourierModeIcParams {
    std::ptrdiff_t kx = 1;
    std::ptrdiff_t ky = 1;
    Real amplitude = Real{1};
    Real phase = Real{0};
};

struct MultiFourierModeIcParams {
    // Empty vector means "use make_default_fourier_modes()".
    std::vector<FourierMode2D> modes;
};

using InitialConditionParams = std::variant<
    GaussianIcParams,
    HotSquareIcParams,
    ConstantIcParams,
    SingleFourierModeIcParams,
    MultiFourierModeIcParams>;

// Returns the canonical config-file/metadata name of the selected IC type:
// "gaussian", "hot_square", "constant", "single_fourier_mode",
// "multi_fourier_mode". Used for JSON dispatch and output metadata.
std::string initial_condition_type_name(const InitialConditionParams& ic);


// ---------------------------------------------------------------------------
// Output-time specification
// ---------------------------------------------------------------------------
// Users may specify output times either as a uniform grid or explicitly.
// time_grid expands a UniformTimeSpec into a concrete times vector.

struct UniformTimeSpec {
    Real t_start = Real{0};
    Real t_end = Real{0.5};
    std::size_t num_snapshots = 50;
};

struct ExplicitTimeSpec {
    RealVec times;
};

using TimeSpec = std::variant<UniformTimeSpec, ExplicitTimeSpec>;


// ---------------------------------------------------------------------------
// Remaining settings blocks
// ---------------------------------------------------------------------------

// Domain/grid/physics settings, prior to conversion into Heat2DConfig.
struct SolverSettings {
    std::size_t nx = 256;
    std::size_t ny = 256;
    Real Lx = Real{2};
    Real Ly = Real{2};
    Real alpha = Real{1};
};

struct OutputSettings {
    std::string output_path;     // path of the output .h5 file
    bool overwrite = false;      // refuse to clobber unless explicitly set
    int gzip_level = 4;          // 0 disables compression of /snapshots
};

struct DiagnosticsSettings {
    bool enabled = true;                 // per-snapshot mean/L2/min/max
    bool compute_analytic_error = false; // Fourier-mode ICs only
};


// ---------------------------------------------------------------------------
// RunConfig
// ---------------------------------------------------------------------------

// One complete user-requested simulation run, as parsed from a JSON file.
struct RunConfig {
    int schema_version = 0;

    // Verbatim text of the input JSON file, stored so the output writer can
    // embed it in the HDF5 file for exact reproducibility.
    std::string source_json_text;

    SolverSettings solver;
    TimeSpec time_spec = UniformTimeSpec{};
    FftBackend fft_backend = FftBackend::custom;
    InitialConditionParams initial_condition = GaussianIcParams{};
    OutputSettings output;
    DiagnosticsSettings diagnostics;
};

// Expands the time specification (via time_grid) and returns the fully
// populated Heat2DConfig for the solver. Throws std::invalid_argument if the
// expanded configuration is invalid.
Heat2DConfig make_heat2d_config(const RunConfig& run_config);

// Dispatches on the initial-condition variant and calls the matching
// generator from initial_conditions.hpp with the configured parameters.
Grid2D<Real> make_initial_condition(const RunConfig& run_config);