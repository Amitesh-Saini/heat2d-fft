// npy_writer.hpp
// Responsibility:
//   Declare utilities for writing Grid2D<Real> data to NumPy .npy files.
//
//   Full 2D temperature snapshots should not be saved as CSV because CSV is
//   slow, large, and can lose floating-point precision if formatting is not
//   handled carefully.
//
//   The .npy format stores raw numerical arrays in a form that Python/NumPy can
//   load directly with:
//
//       np.load("snapshot_0000.npy")
//
//   This module provides the low-level field-output backend used by the
//   higher-level snapshot writer.