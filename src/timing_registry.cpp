// timing_registry.cpp
// Responsibility:
//   Implements the named-region timing registry declared in
//   timing_registry.hpp.
//
//   The registry is a single file-local map, so it is process-wide state.
//   That is deliberate: a scoped region deep inside the solver has no way to
//   reach a registry passed down by argument without threading a parameter
//   through every function between, which is exactly the invasiveness the
//   annotation approach is meant to avoid.

#include "timing_registry.hpp"

#include <chrono>

namespace timing {

namespace {

// Process-wide accumulation. Constructed on first use, which avoids any
// static initialization order question with translation units that record
// regions during their own static construction.
std::map<std::string, RegionRecord>& registry(){

    static std::map<std::string, RegionRecord> records;
    return records;
}

// Nanoseconds since an arbitrary fixed point. Only differences are used, so
// the origin does not matter; steady_clock is required because a wall clock
// can be adjusted mid-run and produce negative intervals.
std::uint64_t now_ns(){

    static_assert(std::chrono::steady_clock::is_steady,
                  "timing registry requires a monotonic steady_clock");

    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

} // namespace


bool enabled(){

#ifdef HEAT2D_ENABLE_TIMING
    return true;
#else
    return false;
#endif
}


void reset(){

    registry().clear();
}


void accumulate(const std::string& name, std::uint64_t duration_ns){

    RegionRecord& record = registry()[name];

    record.total_ns += duration_ns;
    record.calls += 1;
}


std::uint64_t elapsed_ns(const std::string& name){

    // A name that was never recorded reads as zero rather than throwing: an
    // uninstrumented build records nothing at all, and a caller asking for a
    // region that did not run should see zero, not an error.
    const auto entry = registry().find(name);

    if(entry == registry().end()){
        return 0;
    }

    return entry->second.total_ns;
}


std::uint64_t call_count(const std::string& name){

    const auto entry = registry().find(name);

    if(entry == registry().end()){
        return 0;
    }

    return entry->second.calls;
}


const std::map<std::string, RegionRecord>& all(){

    return registry();
}


ScopedRegion::ScopedRegion(std::string name)
    : name_(std::move(name)), start_ns_(now_ns()){
}


ScopedRegion::~ScopedRegion(){

    // Destructors must not throw. accumulate() can allocate when the region
    // name is new, so the first entry to a region is the only place this
    // could fail; swallowing it loses one measurement rather than terminating
    // a benchmark run.
    try{

        accumulate(name_, now_ns() - start_ns_);
    }
    catch(...){
    }
}

} // namespace timing