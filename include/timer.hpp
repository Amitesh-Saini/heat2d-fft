#pragma once
// timer.hpp
// Responsibility:
//   Tiny timing utility for benchmarks and runtime summaries.
// What to do here:
//   - Use std::chrono to measure elapsed wall-clock time.
//   - Keep benchmark timing simple and reusable.
//
// Notes:
//   - elapsed_seconds() is for human-facing run summaries.
//   - elapsed_ns() is the benchmark path: an integer nanosecond count, so a
//     short timed region is not first squeezed through a double in seconds
//     and then scaled back up.
//   - reset() restarts the measurement without constructing a new Timer, so
//     one instance can be reused across repeated timed regions.

#include <chrono>
#include <cstdint>

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}

    // A monotonic clock is a hard requirement here: a wall clock that can be
    // adjusted mid-run would produce negative or inflated intervals.
    static_assert(std::chrono::steady_clock::is_steady,
                  "Timer requires a monotonic steady_clock");

    // Restarts the measurement from now.
    void reset() {
        start_ = std::chrono::steady_clock::now();
    }

    double elapsed_seconds() const {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }

    std::uint64_t elapsed_ns() const {
        const auto now = std::chrono::steady_clock::now();
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_).count());
    }

private:
    std::chrono::steady_clock::time_point start_;
};