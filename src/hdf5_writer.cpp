// npy_writer.cpp
// Responsibility:
//   Implement writing of Grid2D<Real> arrays to NumPy .npy files.
//
//   This file should handle the details of the .npy format:
//     - writing the NumPy file header,
//     - recording the array shape,
//     - storing the data type,
//     - writing the raw grid values in a layout that Python can load.
//
//   The rest of the project should not need to know the details of the .npy
//   format. Higher-level output code should call this module when it needs to
//   save a full 2D field snapshot.