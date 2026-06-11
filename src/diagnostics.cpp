// diagnostics.cpp
// Responsibility:
//   Implement diagnostic calculations for heat-equation snapshots.
//
//   This file should compute scalar summary quantities from Grid2D<Real>
//   snapshots, such as:
//     - mean temperature,
//     - L2 norm,
//     - minimum value,
//     - maximum value.
//
//   These diagnostics can be written to diagnostics.csv by the snapshot writer
//   and later plotted by Python scripts.
//
//   First versions can compute diagnostics directly from physical-space
//   snapshots. Later versions may compute some diagnostics spectrally using
//   Parseval's theorem or the zero Fourier mode.