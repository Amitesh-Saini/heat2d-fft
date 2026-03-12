#pragma once
// grid2d.hpp
// Responsibility:
//   Lightweight contiguous 2D container for real or complex grid data.
// What to do here:
//   - Store data in one flat std::vector<T>.
//   - Map (i, j) to data_[i * ny_ + j].
//   - Provide clean indexing and raw access.
//   - Keep this generic and reusable for physical-space and Fourier-space fields.

#include <cstddef>
#include <vector>

template <typename T>
class Grid2D {
public:
    Grid2D() = default;
    Grid2D(std::size_t nx, std::size_t ny) : nx_(nx), ny_(ny), data_(nx * ny) {}

    void resize(std::size_t nx, std::size_t ny) {
        nx_ = nx;
        ny_ = ny;
        data_.assign(nx * ny, T{});
    }

    T& operator()(std::size_t i, std::size_t j) { return data_[i * ny_ + j]; }
    const T& operator()(std::size_t i, std::size_t j) const { return data_[i * ny_ + j]; }

    std::size_t nx() const { return nx_; }
    std::size_t ny() const { return ny_; }

    std::vector<T>& raw() { return data_; }
    const std::vector<T>& raw() const { return data_; }

private:
    std::size_t nx_ = 0;
    std::size_t ny_ = 0;
    std::vector<T> data_;
};

