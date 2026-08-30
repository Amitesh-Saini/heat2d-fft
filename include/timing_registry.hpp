#pragma once
// timing_registry.hpp
// Responsibility:
//   A named-region timing registry for instrumenting the solver in place.
//
//   Regions are annotated inside the real code rather than reimplemented in a
//   benchmark harness. A copy of a loop can drift from the original when one
//   is optimized and the other is forgotten, and the benchmark then reports
//   the performance of code nobody runs. Annotations cannot drift, because
//   there is only one loop.
//
//   This is a small version of what Caliper, TAU, and Score-P do in
//   production HPC codes: a scoped object opens a named region, accumulates
//   its duration on destruction, and a registry holds the running totals for
//   the caller to read afterwards.
//
// Use:
//   Wrap a region with the macro, which names it and scopes it to the
//   enclosing braces:
//
//       {
//           HEAT2D_TIME_REGION("decay");
//           ... work ...
//       }
//
//   Then, in the benchmark:
//
//       timing::reset();
//       solver.solve();
//       const std::uint64_t decay_ns = timing::elapsed_ns("decay");
//
// Cost and build:
//   The macro expands to nothing unless HEAT2D_ENABLE_TIMING is defined, so
//   an ordinary build carries no instrumentation at all. When it is defined,
//   each region costs two clock reads and a map lookup; against a solver
//   snapshot measured in milliseconds that is under a thousandth, but the
//   flag exists so the choice is explicit rather than assumed.
//
//   An instrumented build is not the same binary as a release build, so
//   timing::enabled() is recorded in the run metadata.
//
// Threading:
//   Single-threaded only. The registry has no synchronization; adding
//   threads would need per-thread accumulation, as Caliper does.

#include <cstdint>
#include <map>
#include <string>

namespace timing {

// Accumulated total and call count for one named region.
struct RegionRecord {

    std::uint64_t total_ns = 0;
    std::uint64_t calls = 0;
};

// True when the build was configured with HEAT2D_ENABLE_TIMING. Recorded in
// the run metadata so an instrumented run is distinguishable from a plain
// one.
bool enabled();

// Clears every accumulated region. Called before each measured solve so the
// totals describe that solve alone.
void reset();

// Accumulates a duration against a region name, creating it on first use.
// ScopedRegion calls this; direct use is for cases where RAII does not fit.
void accumulate(const std::string& name, std::uint64_t duration_ns);

// Total accumulated nanoseconds for a region. Returns 0 for a name that was
// never recorded, which is also what an uninstrumented build returns for
// every name.
std::uint64_t elapsed_ns(const std::string& name);

// Number of times a region was entered. Useful for confirming a region ran
// as often as expected, for example once per snapshot.
std::uint64_t call_count(const std::string& name);

// Every region recorded since the last reset, for writing into the run
// metadata or for diagnostics.
const std::map<std::string, RegionRecord>& all();


// Times the enclosing scope and accumulates it against a region name on
// destruction. Non-copyable: each object owns one region entry.
class ScopedRegion {

public:

    explicit ScopedRegion(std::string name);
    ~ScopedRegion();

    ScopedRegion(const ScopedRegion&) = delete;
    ScopedRegion& operator=(const ScopedRegion&) = delete;

private:

    std::string name_;
    std::uint64_t start_ns_;
};

} // namespace timing


// Opens a timed region for the enclosing scope. Expands to nothing unless the
// build defines HEAT2D_ENABLE_TIMING.
//
// The generated variable name is line-derived so two regions in one scope do
// not collide.
#ifdef HEAT2D_ENABLE_TIMING

    #define HEAT2D_TIME_REGION_CONCAT_INNER(a, b) a##b
    #define HEAT2D_TIME_REGION_CONCAT(a, b) HEAT2D_TIME_REGION_CONCAT_INNER(a, b)

    #define HEAT2D_TIME_REGION(name) \
        ::timing::ScopedRegion HEAT2D_TIME_REGION_CONCAT(heat2d_timed_region_, __LINE__)(name)

#else

    #define HEAT2D_TIME_REGION(name) ((void)0)

#endif