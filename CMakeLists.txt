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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Core numerical library
add_library(heat2d_fft_core
    src/config.cpp
    src/dft1d.cpp
    src/fft1d.cpp
    src/fft2d.cpp
    src/wavenumbers.cpp
    src/initial_conditions.cpp
    src/heat2d_fourier.cpp
    src/norms.cpp
    src/io_csv.cpp
)

target_include_directories(heat2d_fft_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Main executable
add_executable(heat2d_fft_main
    src/main.cpp
)

target_link_libraries(heat2d_fft_main
    PRIVATE
        heat2d_fft_core
)

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