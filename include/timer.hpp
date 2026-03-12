#pragma once
// timer.hpp
// Responsibility:
//   Tiny timing utility for benchmarks and runtime summaries.
// What to do here:
//   - Use std::chrono to measure elapsed wall-clock time.
//   - Keep benchmark timing simple and reusable.

#include <chrono>

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}

    double elapsed_seconds() const {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

