#pragma once
// config.hpp
// Responsibility:
//   Central place for run configuration and preset experiment choices.
// What to do here:
//   - Store default grid size, alpha, output times, and selected initial condition.
//   - Keep main.cpp thin by constructing configuration here.
//   - Expand later if you add command-line parsing.

#include "heat2d_fourier.hpp"

Heat2DConfig default_heat2d_config();

