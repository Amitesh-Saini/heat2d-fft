// config_io.hpp
// Responsibility:
//   Declare functions for reading and writing simulation configuration files.
//
//   The main executable is intended to be config-file driven rather than
//   interactive. A user should be able to run the program with a command like:
//
//       ./heat2d configs/gaussian_demo.json
//
//   This module converts JSON configuration files into RunConfig / Heat2DConfig
//   objects that the rest of the program can use.
//
//   Keeping JSON parsing here prevents main.cpp from being filled with file I/O
//   and parsing details.