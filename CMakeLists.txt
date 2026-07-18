# CMakeLists.txt
# Responsibility:
#   Top-level CMake build configuration for the 2D Fourier/FFT heat-equation project.
# What to do here:
#   - Define the project name and C++ standard.
#   - Create targets for the core library/executables.
#   - Add include directories.
#   - Add optional test and benchmark targets.
#   - Keep build logic here, not inside source files.

cmake_minimum_required(VERSION 3.20)
project(heat2d_fft LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------

# Locate FFTW3 via pkg-config.
# On macOS with Homebrew, install with:
#   brew install fftw pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3)

# HDF5 C library for consolidated run-output files.
# On macOS with Homebrew, install with:
#   brew install hdf5
find_package(HDF5 REQUIRED COMPONENTS C)

# JSON library for config-file driven runs.
# First try a system/package-manager install. If unavailable, fetch it.
find_package(nlohmann_json QUIET)

if(NOT nlohmann_json_FOUND)
    include(FetchContent)

    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    FetchContent_MakeAvailable(nlohmann_json)
endif()

# ------------------------------------------------------------
# Build provenance (build_info.hpp)
# ------------------------------------------------------------
# Captures the git commit at CMake configure time and generates
# build_info.hpp in the build tree. Note: the hash is captured when CMake
# runs, so re-run CMake after committing if you need the hash to be exact.

execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE HEAT2D_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT HEAT2D_GIT_COMMIT)
    set(HEAT2D_GIT_COMMIT "unknown")
endif()

# CMake keeps optimization flags in the per-configuration variable
# (CMAKE_CXX_FLAGS_RELEASE / _DEBUG), not in the base CMAKE_CXX_FLAGS.
# Combine both so provenance records the flags a run was actually built with.
set(HEAT2D_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")

message(STATUS "HEAT2D_CXX_FLAGS = '${HEAT2D_CXX_FLAGS}'")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/build_info.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/build_info.hpp
    @ONLY
)

# ------------------------------------------------------------
# Build options
# ------------------------------------------------------------

option(HEAT2D_BUILD_TESTS "Build test executables" ON)
option(HEAT2D_BUILD_BENCHMARKS "Build benchmark executables" ON)

# ------------------------------------------------------------
# Core numerical library
# ------------------------------------------------------------

add_library(heat2d_fft_core
    # Existing core files
    src/config.cpp
    src/dft1d.cpp
    src/dft2d.cpp
    src/fft1d.cpp
    src/fft2d.cpp
    src/wavenumbers.cpp
    src/initial_conditions.cpp
    src/heat2d_fourier.cpp

    # New architecture/output/config files
    src/run_config.cpp
    src/config_io.cpp
    src/time_grid.cpp
    src/diagnostics.cpp
    src/hdf5_writer.cpp
    src/snapshot_writer.cpp
)

target_include_directories(heat2d_fft_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_link_libraries(heat2d_fft_core
    PUBLIC
        PkgConfig::FFTW
        HDF5::HDF5
        nlohmann_json::nlohmann_json
)

# ------------------------------------------------------------
# Main executable
# ------------------------------------------------------------

add_executable(heat2d_fft_main
    src/main.cpp
)

target_link_libraries(heat2d_fft_main
    PRIVATE
        heat2d_fft_core
)

# Optional: make the built executable name nicer.
# Then you can run:
#   ./heat2d configs/gaussian_demo.json
set_target_properties(heat2d_fft_main
    PROPERTIES
        OUTPUT_NAME heat2d
)

# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

if(HEAT2D_BUILD_TESTS)
    enable_testing()

    # Test executable: direct 1D DFT
    add_executable(test_dft1d
        tests/test_dft1d.cpp
        tests/1D_test_utils.cpp
    )

    target_include_directories(test_dft1d
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    target_link_libraries(test_dft1d
        PRIVATE
            heat2d_fft_core
    )

    add_test(NAME test_dft1d COMMAND test_dft1d)

    # Test executable: radix-2 1D FFT
    add_executable(test_fft1d
        tests/test_fft1d.cpp
        tests/1D_test_utils.cpp
    )

    target_include_directories(test_fft1d
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    target_link_libraries(test_fft1d
        PRIVATE
            heat2d_fft_core
    )

    add_test(NAME test_fft1d COMMAND test_fft1d)

    # Test executable: direct 2D DFT
    add_executable(test_dft2d
        tests/test_dft2d.cpp
        tests/2D_test_utils.cpp
        tests/1D_test_utils.cpp
    )

    target_include_directories(test_dft2d
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    target_link_libraries(test_dft2d
        PRIVATE
            heat2d_fft_core
    )

    add_test(NAME test_dft2d COMMAND test_dft2d)

    # Test executable: 2D FFT
    add_executable(test_fft2d
        tests/test_fft2d.cpp
        tests/2D_test_utils.cpp
        tests/1D_test_utils.cpp
    )

    target_include_directories(test_fft2d
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    target_link_libraries(test_fft2d
        PRIVATE
            heat2d_fft_core
    )

    add_test(NAME test_fft2d COMMAND test_fft2d)

    # Test executable: Fourier wavenumbers
    add_executable(test_wavenumbers
        tests/test_wavenumbers.cpp
        tests/1D_test_utils.cpp
    )

    target_include_directories(test_wavenumbers
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    target_link_libraries(test_wavenumbers
        PRIVATE
            heat2d_fft_core
    )

    add_test(NAME test_wavenumbers COMMAND test_wavenumbers)

    # Test executable: 2D Fourier heat solver
    add_executable(test_heat2d_fourier
        tests/test_heat2d_fourier.cpp
        tests/1D_test_utils.cpp
    )

    target_include_directories(test_heat2d_fourier
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    target_link_libraries(test_heat2d_fourier
        PRIVATE
            heat2d_fft_core
    )

    add_test(NAME test_heat2d_fourier COMMAND test_heat2d_fourier)


    # Test executable: diagnostics (analytic reference + solver validation)
    add_executable(test_diagnostics
        tests/test_diagnostics.cpp
    )

    target_link_libraries(test_diagnostics
        PRIVATE
            heat2d_fft_core
    )

    add_test(NAME test_diagnostics COMMAND test_diagnostics)

    # Test executable: HDF5 writer (layout convention + hyperslab offsets)
    # Writes probe.h5 into the working directory; scripts/verify_probe.py
    # reads that file back with h5py to check the cross-language contract.
    add_executable(test_hdf5_writer
        tests/test_hdf5_writer.cpp
    )

    target_link_libraries(test_hdf5_writer
        PRIVATE
            heat2d_fft_core
    )

    add_test(NAME test_hdf5_writer COMMAND test_hdf5_writer)

    # New tests can be added once implemented, following the same pattern:
    #   - tests/test_time_grid.cpp
    #   - tests/test_config_io.cpp   (valid / missing key / unknown key / bad value)
    #   - tests/test_diagnostics.cpp
    #   - tests/test_hdf5_writer.cpp (C++ writes, Python/h5py round-trip reads)
endif()

# ------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------

if(HEAT2D_BUILD_BENCHMARKS)
    add_executable(bench_fft1d
        benchmarks/bench_fft1d.cpp
    )

    target_link_libraries(bench_fft1d
        PRIVATE
            heat2d_fft_core
    )

    add_executable(bench_fft2d
        benchmarks/bench_fft2d.cpp
    )

    target_link_libraries(bench_fft2d
        PRIVATE
            heat2d_fft_core
    )

    add_executable(bench_heat2d
        benchmarks/bench_heat2d.cpp
    )

    target_link_libraries(bench_heat2d
        PRIVATE
            heat2d_fft_core
    )
endif()