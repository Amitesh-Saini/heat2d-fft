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

add_executable(heat2d_fft_main
    src/dft1d.cpp
    src/fft1d.cpp
    src/fft2d.cpp
    src/wavenumbers.cpp
    src/initial_conditions.cpp
    src/heat2d_fourier.cpp
    src/norms.cpp
    src/io_csv.cpp
    src/main.cpp
)

target_include_directories(heat2d_fft_main PRIVATE include)
