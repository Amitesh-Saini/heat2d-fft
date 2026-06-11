// config.hpp
// Responsibility:
//   Declare helper functions for constructing default solver configurations.
//
//   Configuration objects store solver-level parameters such as domain size,
//   grid resolution, thermal diffusivity, and output times. Numerical solver
//   logic belongs in heat2d_fourier.cpp, not here.

#pragma once

#include "heat2d_fourier.hpp"

// Returns a standard default configuration for the 2D periodic FFT heat solver.
// The default domain is [-1, 1) x [-1, 1), represented by Lx = Ly = 2.
Heat2DConfig make_default_heat2d_config();