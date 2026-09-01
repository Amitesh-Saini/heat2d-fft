#pragma once
// bench_platform.hpp
// Responsibility:
//   Declare the platform-specific queries that describe the machine a
//   benchmark ran on: identity, cache sizes, and process memory.
//
//   This is the only file in the benchmark suite that contains conditional
//   compilation. Everything else stays portable by depending on these
//   declarations rather than on the operating system, so the platform
//   handling is contained in one place instead of being scattered through the
//   code that happens to need it.
//
// Portability status:
//   macOS is the only platform these have been RUN on. The Linux and Windows
//   branches are written and compile-checked where possible, but untested.
//   That distinction is recorded here and in the README rather than glossed
//   over: an untested branch that is labelled as such is honest, while one
//   presented as supported is not.
//
// Failure handling:
//   Every query returns std::optional and yields nullopt when the underlying
//   call fails or the platform is unsupported. Returning zero instead would
//   put a value in the CSV that is indistinguishable from a real measurement
//   of zero, which is the same reason the result schemas use optionals for
//   absent fields.

#include <cstdint>
#include <optional>
#include <string>

#include "benchmark_types.hpp"


// ---------------------------------------------------------------------------
// Process memory
// ---------------------------------------------------------------------------

// Peak resident set size in bytes: the high-water mark of physical memory the
// process has held since it started.
//
// This is the figure the memory benchmark compares against its analytic
// working-set model. Because it is monotonic within a process, it is only
// meaningful when the binary is invoked once per configuration: a single
// process sweeping a size ladder would report the largest configuration's
// footprint on every row.
//
// Units differ by platform and are normalized to bytes here. getrusage's
// ru_maxrss is kilobytes on Linux and bytes on macOS, which is a long-
// standing inconsistency and an easy factor of 1024 to get wrong.
std::optional<std::uint64_t> peak_process_memory_bytes();

// Current resident set size in bytes: physical memory the process holds right
// now, which can fall as well as rise.
//
// Useful for bracketing a phase and taking the difference, which peak cannot
// do because it never decreases. Not used by the memory benchmark itself,
// which wants the high-water mark.
std::optional<std::uint64_t> current_process_memory_bytes();


// ---------------------------------------------------------------------------
// Machine description
// ---------------------------------------------------------------------------

// Fills a machine_info with everything this platform can report. Fields that
// cannot be queried are left at their defaults, and l3_bytes is left absent
// where the hardware has no conventional L3 at all: Apple Silicon has a
// shared System Level Cache that is not exposed the same way, so an absent
// value there is correct rather than a failed query.
//
// The cache sizes are the annotation for the cache-cliff plot, which marks
// where a transform's working set crosses each level of the hierarchy.
machine_info query_machine_info();


// The individual queries behind query_machine_info, exposed for callers that
// want one field without the rest.

std::optional<std::string> query_hostname();

// Operating system name and version, formatted for a metadata field rather
// than for parsing: "macOS 15.3", "Linux 6.8.0-45-generic", "Windows 10.0".
std::optional<std::string> query_os_name();

// CPU model string as the vendor reports it.
std::optional<std::string> query_cpu_model();

// Cache sizes in bytes. L1d is per core; L2 and L3 may be per core, per
// cluster, or shared depending on the architecture, and the platform APIs do
// not consistently say which. The values are recorded as reported and the
// README notes the ambiguity rather than pretending it away.
std::optional<std::uint64_t> query_l1d_cache_bytes();
std::optional<std::uint64_t> query_l2_cache_bytes();
std::optional<std::uint64_t> query_l3_cache_bytes();