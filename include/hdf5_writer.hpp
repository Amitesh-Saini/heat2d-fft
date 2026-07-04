#pragma once
// hdf5_writer.hpp
// Responsibility:
//   Declare a small RAII wrapper over the HDF5 C API for writing Grid2D
//   and vector data to a single consolidated HDF5 output file.
//
//   Full 2D temperature snapshots should not be saved as CSV because CSV is
//   slow, large, and can lose floating-point precision. HDF5 stores raw
//   numerical arrays in a self-describing, portable form that Python can
//   load directly:
//
//       import h5py
//       with h5py.File("run.h5") as f:
//           u = f["/snapshots"][t]      # one snapshot, shape (nx, ny)
//
//   This module is the low-level field-output backend used by the
//   higher-level snapshot writer. It knows about HDF5; it does not know
//   about the run layout (/x, /y, /times, ...), which belongs to
//   snapshot_writer.
//
// Memory model:
//   /snapshots is created as a chunked, extensible dataset with chunk shape
//   (1, nx, ny) and optional gzip compression. Snapshots are appended one at
//   a time as they are computed, so memory stays bounded at a single
//   snapshot regardless of run size, and a killed run still leaves every
//   snapshot written so far on disk.
//
// Layout convention:
//   Grid2D<Real> is row-major with (i, j) -> data[i * ny + j], so a snapshot
//   is written as a C-order (nx, ny) slab. In Python, f["/snapshots"][t, i, j]
//   is exactly u(x_i, y_j). This convention is asserted by an end-to-end
//   round-trip test on an asymmetric field.

#include <hdf5.h>

#include <cstddef>
#include <string>

#include "grid2d.hpp"
#include "types.hpp"

class Hdf5File {
public:
    // Creates a new HDF5 file at path.
    // If the file already exists: throws std::runtime_error when overwrite
    // is false; truncates when overwrite is true.
    // Throws std::runtime_error if the file cannot be created.
    Hdf5File(const std::string& path, bool overwrite);

    // Closes any open dataset handle and the file. Never throws.
    ~Hdf5File();

    // One open file maps to one writer; copying handles is a bug factory.
    Hdf5File(const Hdf5File&) = delete;
    Hdf5File& operator=(const Hdf5File&) = delete;

    // Writes a fixed-size 1D dataset of Real values (e.g. /x, /y, /times).
    // Throws std::runtime_error on HDF5 failure or duplicate dataset name.
    void write_real_vector(const std::string& dataset_name, const RealVec& values);

    // Writes a UTF-8 string dataset (e.g. /config_json holding the verbatim
    // input configuration).
    void write_string(const std::string& dataset_name, const std::string& text);

    // Writes string / Real attributes on the root group. Used for
    // provenance (git commit, compiler, timestamp, ...) and run summary
    // values (wall time).
    void write_root_string_attribute(const std::string& name, const std::string& value);
    void write_root_real_attribute(const std::string& name, Real value);

    // Creates the extensible snapshot dataset:
    //   shape (0, nx, ny), max shape (unlimited, nx, ny),
    //   chunks (1, nx, ny), gzip at gzip_level (0 disables compression,
    //   otherwise 1-9; combined with the shuffle filter).
    // One snapshot dataset per file. Throws std::runtime_error if called
    // twice or on HDF5 failure.
    void create_snapshot_dataset(
        const std::string& dataset_name, std::size_t nx, std::size_t ny, int gzip_level);

    // Extends the snapshot dataset by one along axis 0 and writes the given
    // grid as the new slab. Throws std::invalid_argument if the grid shape
    // does not match the dataset, std::runtime_error if the dataset has not
    // been created or on HDF5 failure.
    void append_snapshot(const Grid2D<Real>& snapshot);

    // Number of snapshots appended so far.
    std::size_t num_snapshots_written() const;

    // Flushes HDF5 buffers to disk (called after each append so partial
    // runs remain readable).
    void flush();

private:
    hid_t file_id_ = H5I_INVALID_HID;
    hid_t snapshot_dataset_id_ = H5I_INVALID_HID;
    std::size_t snapshot_nx_ = 0;
    std::size_t snapshot_ny_ = 0;
    std::size_t snapshots_written_ = 0;
};