#pragma once
// types.hpp
// Responsibility:
//   Common type aliases used across the project.
// What to do here:
//   - Define scalar and complex types.
//   - Add common vector aliases only if they improve clarity.
//   - Keep this lightweight; do not add solver logic here.

#include <complex>
#include <vector>

using Real = double;
using Complex = std::complex<double>;
using RealVec = std::vector<Real>;
using ComplexVec = std::vector<Complex>;

constexpr Real PI = 3.14159265358979323846;

