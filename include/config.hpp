#pragma once
// config.hpp
// Responsibility:
//   Defines helper utilities for constructing default solver configurations.
// Notes:
//   - Keeps main.cpp thin by centralizing default parameter choices.
//   - Extend this later if you add command-line parsing or experiment presets.

#include "heat2d_fourier.hpp"

// Returns a default configuration for the 2D Fourier heat solver.
Heat2DConfig make_default_heat2d_config();