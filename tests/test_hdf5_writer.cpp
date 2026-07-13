// test_hdf5_writer.cpp
// Responsibility:
//   Tests for Hdf5File, the RAII wrapper over the HDF5 C API.
//
//   This test covers the half of the contract that C++ can verify on its own:
//   that the full write sequence completes without throwing, that resource
//   handling survives repeated appends, and that the snapshot counter tracks
//   correctly. It also produces probe.h5, which scripts/verify_probe.py reads
//   back with h5py to verify the half that C++ CANNOT check -- the
//   cross-language layout contract:
//
//     - /snapshots has shape (nt, nx, ny), not transposed to (nt, ny, nx)
//     - f["/snapshots"][t][i][j] equals the C++ grid(i, j)
//     - the hyperslab offset advances, so appends do not overwrite slab 0
//
//   Regenerate probe.h5 by running this binary, then:
//       python scripts/verify_probe.py
//
// Test data:
//   The grid is deliberately NON-SQUARE (4 x 6) so an axis swap changes the
//   reported shape, and filled with u(i,j) = i + 10*j so every cell's value
//   encodes its own address -- a transposed write cannot pass unnoticed. The
//   second snapshot is offset by 1000 so an unadvanced hyperslab offset
//   (which would overwrite slab 0) produces two identical snapshots.
//
// Convention:
//   main() returns 0 only if all checks pass; failures are recorded in
//   failed_tests and the process returns nonzero so CTest reports by exit code.

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "grid2d.hpp"
#include "hdf5_writer.hpp"
#include "types.hpp"


namespace {

// Builds an asymmetric grid: u(i,j) = base + i + 10*j.
Grid2D<Real> make_probe_grid(std::size_t nx, std::size_t ny, Real base) {
    Grid2D<Real> grid(nx, ny);
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            grid(i, j) = base + static_cast<Real>(i) + Real{10} * static_cast<Real>(j);
        }
    }
    return grid;
}

} // namespace


int main() {

    std::vector<std::string> failed_tests;

    std::cout << "=== Running hdf5_writer tests ===\n\n";

    const std::size_t nx = 4;
    const std::size_t ny = 6;
    const std::string probe_path = "probe.h5";


    // -----------------------------------------------------------------------
    // Test 1: the full write sequence completes without throwing, and the
    // snapshot counter tracks the number of appends.
    // -----------------------------------------------------------------------
    {
        const std::string name = "write_sequence_completes";

        try {
            Grid2D<Real> snap0 = make_probe_grid(nx, ny, Real{0});
            Grid2D<Real> snap1 = make_probe_grid(nx, ny, Real{1000});

            const RealVec x_coords = { 0.0, 1.0, 2.0, 3.0 };
            const RealVec y_coords = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
            const RealVec times    = { 0.0, 0.5 };

            Hdf5File file(probe_path, /*overwrite=*/true);

            file.write_real_vector("/x", x_coords);
            file.write_real_vector("/y", y_coords);
            file.write_real_vector("/times", times);

            file.write_string("/config_json", "{\"probe\": true}");

            file.write_root_string_attribute("git_commit", "abc123");
            file.write_root_real_attribute("wall_time_seconds", 1.25);

            file.create_snapshot_dataset("/snapshots", nx, ny, /*gzip_level=*/4);

            file.append_snapshot(snap0);
            file.flush();

            file.append_snapshot(snap1);
            file.flush();

            if (file.num_snapshots_written() != 2) {
                std::cout << "FAIL: " << name << "\n";
                std::cout << "num_snapshots_written() = " << file.num_snapshots_written()
                          << " (expected 2)\n\n";
                failed_tests.push_back(name);
            }
        }
        catch (const std::exception& e) {
            std::cout << "FAIL: " << name << "\n";
            std::cout << "threw: " << e.what() << "\n\n";
            failed_tests.push_back(name);
        }
    }


    // -----------------------------------------------------------------------
    // Test 2: appending a grid whose shape does not match the dataset is
    // rejected rather than silently writing garbage.
    // -----------------------------------------------------------------------
    {
        const std::string name = "mismatched_snapshot_shape_rejected";

        try {
            Hdf5File file("probe_shape_check.h5", /*overwrite=*/true);
            file.create_snapshot_dataset("/snapshots", nx, ny, /*gzip_level=*/0);

            Grid2D<Real> wrong_shape = make_probe_grid(ny, nx, Real{0});  // transposed dims

            bool threw = false;
            try {
                file.append_snapshot(wrong_shape);
            }
            catch (const std::invalid_argument&) {
                threw = true;
            }

            if (!threw) {
                std::cout << "FAIL: " << name << "\n";
                std::cout << "append_snapshot accepted a grid with mismatched shape\n\n";
                failed_tests.push_back(name);
            }
        }
        catch (const std::exception& e) {
            std::cout << "FAIL: " << name << "\n";
            std::cout << "threw unexpectedly during setup: " << e.what() << "\n\n";
            failed_tests.push_back(name);
        }
    }


    // -----------------------------------------------------------------------
    // Test 3: creating the snapshot dataset twice is rejected.
    // -----------------------------------------------------------------------
    {
        const std::string name = "duplicate_snapshot_dataset_rejected";

        try {
            Hdf5File file("probe_duplicate_check.h5", /*overwrite=*/true);
            file.create_snapshot_dataset("/snapshots", nx, ny, /*gzip_level=*/0);

            bool threw = false;
            try {
                file.create_snapshot_dataset("/snapshots_again", nx, ny, /*gzip_level=*/0);
            }
            catch (const std::runtime_error&) {
                threw = true;
            }

            if (!threw) {
                std::cout << "FAIL: " << name << "\n";
                std::cout << "create_snapshot_dataset allowed a second dataset\n\n";
                failed_tests.push_back(name);
            }
        }
        catch (const std::exception& e) {
            std::cout << "FAIL: " << name << "\n";
            std::cout << "threw unexpectedly during setup: " << e.what() << "\n\n";
            failed_tests.push_back(name);
        }
    }


    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    if (failed_tests.empty()) {
        std::cout << "All tests passed.\n";
        std::cout << "Wrote " << probe_path
                  << " -- verify the layout contract with: python scripts/verify_probe.py\n";
        return 0;
    }

    std::cout << failed_tests.size() << " test(s) failed:\n";
    for (const auto& test_name : failed_tests) {
        std::cout << " - " << test_name << "\n";
    }

    return 1;
}