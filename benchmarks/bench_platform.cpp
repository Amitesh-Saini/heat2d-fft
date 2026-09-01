// bench_platform.cpp
// Responsibility:
//   Implements the platform queries declared in bench_platform.hpp.
//
//   Every branch is guarded by the same three-way #if, in the same order, so
//   the file reads as three parallel implementations of one interface rather
//   than as scattered conditionals. Nothing outside this file needs to know
//   which platform it is running on.
//
// Portability status:
//   Only the macOS branch has been run. Linux and Windows are written from
//   their documented interfaces and are untested.

#include "bench_platform.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#if defined(__APPLE__)

    #include <mach/mach.h>
    #include <sys/resource.h>
    #include <sys/sysctl.h>
    #include <sys/utsname.h>
    #include <unistd.h>

#elif defined(__linux__)

    #include <sys/resource.h>
    #include <sys/utsname.h>
    #include <unistd.h>

#elif defined(_WIN32)

    #ifndef NOMINMAX
        #define NOMINMAX
    #endif

    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif

    #include <windows.h>
    #include <psapi.h>

#endif


// ---------------------------------------------------------------------------
// Process memory
// ---------------------------------------------------------------------------

std::optional<std::uint64_t> peak_process_memory_bytes(){

#if defined(__APPLE__)

    rusage usage;

    if(getrusage(RUSAGE_SELF, &usage) != 0){
        return std::nullopt;
    }

    // ru_maxrss is BYTES on macOS and KILOBYTES on Linux. The two branches
    // therefore differ by a factor of 1024 despite calling the same function,
    // which is a long-standing inconsistency in the interface rather than a
    // mistake here.
    return static_cast<std::uint64_t>(usage.ru_maxrss);

#elif defined(__linux__)

    rusage usage;

    if(getrusage(RUSAGE_SELF, &usage) != 0){
        return std::nullopt;
    }

    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ull;

#elif defined(_WIN32)

    PROCESS_MEMORY_COUNTERS counters;

    if(!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))){
        return std::nullopt;
    }

    return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);

#else

    return std::nullopt;

#endif
}


std::optional<std::uint64_t> current_process_memory_bytes(){

#if defined(__APPLE__)

    // phys_footprint rather than resident_size: it is the figure Activity
    // Monitor reports and the one macOS uses for memory limits, and it
    // accounts for compressed pages that resident_size misses.
    task_vm_info_data_t info;

    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;

    const kern_return_t result = task_info(
        mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count);

    if(result != KERN_SUCCESS){
        return std::nullopt;
    }

    return static_cast<std::uint64_t>(info.phys_footprint);

#elif defined(__linux__)

    // The second field of /proc/self/statm is the resident set size in pages.
    std::ifstream statm("/proc/self/statm");

    if(!statm.is_open()){
        return std::nullopt;
    }

    long total_pages = 0;
    long resident_pages = 0;

    statm >> total_pages >> resident_pages;

    if(!statm || resident_pages < 0){
        return std::nullopt;
    }

    const long page_size = sysconf(_SC_PAGESIZE);

    if(page_size <= 0){
        return std::nullopt;
    }

    return static_cast<std::uint64_t>(resident_pages) *
           static_cast<std::uint64_t>(page_size);

#elif defined(_WIN32)

    PROCESS_MEMORY_COUNTERS counters;

    if(!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))){
        return std::nullopt;
    }

    return static_cast<std::uint64_t>(counters.WorkingSetSize);

#else

    return std::nullopt;

#endif
}


// ---------------------------------------------------------------------------
// Machine identity
// ---------------------------------------------------------------------------

std::optional<std::string> query_hostname(){

#if defined(__APPLE__) || defined(__linux__)

    char buffer[256] = {};

    if(gethostname(buffer, sizeof(buffer) - 1) != 0){
        return std::nullopt;
    }

    return std::string(buffer);

#elif defined(_WIN32)

    char buffer[MAX_COMPUTERNAME_LENGTH + 1] = {};

    DWORD size = sizeof(buffer);

    if(!GetComputerNameA(buffer, &size)){
        return std::nullopt;
    }

    return std::string(buffer, size);

#else

    return std::nullopt;

#endif
}


std::optional<std::string> query_os_name(){

#if defined(__APPLE__)

    // uname reports the Darwin kernel version, which is not the macOS version
    // a reader would recognize. kern.osproductversion gives the marketing
    // version, so both are recorded: the kernel version is what actually
    // determines behaviour, the product version is what identifies the
    // release.
    char product[64] = {};
    std::size_t product_size = sizeof(product);

    const bool have_product =
        sysctlbyname("kern.osproductversion", product, &product_size, nullptr, 0) == 0;

    utsname info;

    if(uname(&info) != 0){
        return have_product ? std::optional<std::string>("macOS " + std::string(product))
                            : std::nullopt;
    }

    if(have_product){
        return "macOS " + std::string(product) + " (Darwin " + info.release + ")";
    }

    return "Darwin " + std::string(info.release);

#elif defined(__linux__)

    utsname info;

    if(uname(&info) != 0){
        return std::nullopt;
    }

    return std::string(info.sysname) + " " + info.release;

#elif defined(_WIN32)

    // GetVersionEx is deprecated and lies about the version unless the binary
    // carries a compatibility manifest, so the build number is read from the
    // registry instead.
    HKEY key;

    const char* path = "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";

    if(RegOpenKeyExA(HKEY_LOCAL_MACHINE, path, 0, KEY_READ, &key) != ERROR_SUCCESS){
        return std::nullopt;
    }

    char build[64] = {};
    DWORD size = sizeof(build);

    const LSTATUS status =
        RegQueryValueExA(key, "CurrentBuild", nullptr, nullptr,
                         reinterpret_cast<LPBYTE>(build), &size);

    RegCloseKey(key);

    if(status != ERROR_SUCCESS){
        return std::nullopt;
    }

    return "Windows build " + std::string(build);

#else

    return std::nullopt;

#endif
}


std::optional<std::string> query_cpu_model(){

#if defined(__APPLE__)

    char model[256] = {};
    std::size_t size = sizeof(model);

    if(sysctlbyname("machdep.cpu.brand_string", model, &size, nullptr, 0) != 0){
        return std::nullopt;
    }

    return std::string(model);

#elif defined(__linux__)

    // /proc/cpuinfo lists every logical core; the model line is identical
    // across them on a homogeneous machine, so the first is enough.
    std::ifstream cpuinfo("/proc/cpuinfo");

    if(!cpuinfo.is_open()){
        return std::nullopt;
    }

    std::string line;

    while(std::getline(cpuinfo, line)){

        // x86 uses "model name"; ARM uses "Hardware" or "CPU implementer".
        const std::string keys[] = {"model name", "Hardware", "Processor"};

        for(const std::string& key : keys){

            if(line.rfind(key, 0) == 0){

                const std::size_t colon = line.find(':');

                if(colon != std::string::npos && colon + 2 <= line.size()){

                    return line.substr(colon + 2);
                }
            }
        }
    }

    return std::nullopt;

#elif defined(_WIN32)

    HKEY key;

    const char* path = "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0";

    if(RegOpenKeyExA(HKEY_LOCAL_MACHINE, path, 0, KEY_READ, &key) != ERROR_SUCCESS){
        return std::nullopt;
    }

    char model[256] = {};
    DWORD size = sizeof(model);

    const LSTATUS status =
        RegQueryValueExA(key, "ProcessorNameString", nullptr, nullptr,
                         reinterpret_cast<LPBYTE>(model), &size);

    RegCloseKey(key);

    if(status != ERROR_SUCCESS){
        return std::nullopt;
    }

    return std::string(model);

#else

    return std::nullopt;

#endif
}


// ---------------------------------------------------------------------------
// Cache sizes
// ---------------------------------------------------------------------------
// These annotate the cache-cliff plot, which marks where a transform's
// working set crosses each level of the hierarchy.
//
// The APIs do not consistently say whether a level is per core, per cluster,
// or shared, so the values are recorded as reported. On Apple Silicon there
// is no conventional L3 at all: an absent value there is correct rather than
// a failed query.

#if defined(__linux__)

namespace {

// Reads a cache size from sysfs, where it is written as a human-readable
// string such as "32K" or "8192K" rather than a plain byte count.
std::optional<std::uint64_t> read_sysfs_cache_size(int index){

    const std::string path =
        "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index) + "/size";

    std::ifstream file(path);

    if(!file.is_open()){
        return std::nullopt;
    }

    std::string text;

    if(!(file >> text) || text.empty()){
        return std::nullopt;
    }

    const char suffix = text.back();

    std::uint64_t multiplier = 1;

    if(suffix == 'K' || suffix == 'k'){
        multiplier = 1024ull;
        text.pop_back();
    }
    else if(suffix == 'M' || suffix == 'm'){
        multiplier = 1024ull * 1024ull;
        text.pop_back();
    }

    try{
        return static_cast<std::uint64_t>(std::stoull(text)) * multiplier;
    }
    catch(...){
        return std::nullopt;
    }
}

// Reads the level and type of a sysfs cache index, so the caller can find the
// data cache at a given level rather than assuming a fixed index layout.
std::optional<std::string> read_sysfs_cache_field(int index, const std::string& field){

    const std::string path =
        "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index) + "/" + field;

    std::ifstream file(path);

    if(!file.is_open()){
        return std::nullopt;
    }

    std::string text;

    if(!(file >> text)){
        return std::nullopt;
    }

    return text;
}

// Finds the size of the data or unified cache at the given level. The index
// numbering is not fixed across architectures, so every index is inspected
// rather than assuming index1 is L1d.
std::optional<std::uint64_t> find_sysfs_cache(int wanted_level, bool want_data_only){

    for(int index = 0; index < 10; ++index){

        const std::optional<std::string> level = read_sysfs_cache_field(index, "level");
        const std::optional<std::string> type = read_sysfs_cache_field(index, "type");

        if(!level.has_value() || !type.has_value()){
            continue;
        }

        if(*level != std::to_string(wanted_level)){
            continue;
        }

        const bool is_data = (*type == "Data");
        const bool is_unified = (*type == "Unified");

        if(want_data_only ? is_data : (is_data || is_unified)){

            return read_sysfs_cache_size(index);
        }
    }

    return std::nullopt;
}

} // namespace

#endif


#if defined(_WIN32)

namespace {

// Walks the processor information buffer looking for a cache at the given
// level. GetLogicalProcessorInformation returns a variable-length array, so
// the buffer is sized by a first call that is expected to fail.
std::optional<std::uint64_t> find_windows_cache(BYTE wanted_level, bool want_data_only){

    DWORD length = 0;

    GetLogicalProcessorInformation(nullptr, &length);

    if(GetLastError() != ERROR_INSUFFICIENT_BUFFER || length == 0){
        return std::nullopt;
    }

    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
        length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));

    if(!GetLogicalProcessorInformation(buffer.data(), &length)){
        return std::nullopt;
    }

    for(const SYSTEM_LOGICAL_PROCESSOR_INFORMATION& entry : buffer){

        if(entry.Relationship != RelationCache){
            continue;
        }

        const CACHE_DESCRIPTOR& cache = entry.Cache;

        if(cache.Level != wanted_level){
            continue;
        }

        const bool is_data = (cache.Type == CacheData);
        const bool is_unified = (cache.Type == CacheUnified);

        if(want_data_only ? is_data : (is_data || is_unified)){

            return static_cast<std::uint64_t>(cache.Size);
        }
    }

    return std::nullopt;
}

} // namespace

#endif


std::optional<std::uint64_t> query_l1d_cache_bytes(){

#if defined(__APPLE__)

    std::uint64_t value = 0;
    std::size_t size = sizeof(value);

    if(sysctlbyname("hw.l1dcachesize", &value, &size, nullptr, 0) != 0){
        return std::nullopt;
    }

    return value;

#elif defined(__linux__)

    return find_sysfs_cache(1, true);

#elif defined(_WIN32)

    return find_windows_cache(1, true);

#else

    return std::nullopt;

#endif
}


std::optional<std::uint64_t> query_l2_cache_bytes(){

#if defined(__APPLE__)

    std::uint64_t value = 0;
    std::size_t size = sizeof(value);

    if(sysctlbyname("hw.l2cachesize", &value, &size, nullptr, 0) != 0){
        return std::nullopt;
    }

    return value;

#elif defined(__linux__)

    return find_sysfs_cache(2, false);

#elif defined(_WIN32)

    return find_windows_cache(2, false);

#else

    return std::nullopt;

#endif
}


std::optional<std::uint64_t> query_l3_cache_bytes(){

#if defined(__APPLE__)

    // Apple Silicon has no conventional L3: there is a System Level Cache
    // shared across the SoC, but it is not exposed through hw.l3cachesize and
    // is not directly comparable to an x86 L3. Intel Macs do report it, so
    // the query is attempted and a missing value is left absent rather than
    // treated as an error.
    std::uint64_t value = 0;
    std::size_t size = sizeof(value);

    if(sysctlbyname("hw.l3cachesize", &value, &size, nullptr, 0) != 0 || value == 0){
        return std::nullopt;
    }

    return value;

#elif defined(__linux__)

    return find_sysfs_cache(3, false);

#elif defined(_WIN32)

    return find_windows_cache(3, false);

#else

    return std::nullopt;

#endif
}


// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

machine_info query_machine_info(){

    machine_info info;

    if(const std::optional<std::string> hostname = query_hostname()){
        info.hostname = *hostname;
    }

    if(const std::optional<std::string> os_name = query_os_name()){
        info.os_name = *os_name;
    }

    if(const std::optional<std::string> cpu_model = query_cpu_model()){
        info.cpu_model = *cpu_model;
    }

    if(const std::optional<std::uint64_t> l1d = query_l1d_cache_bytes()){
        info.l1d_bytes = *l1d;
    }

    if(const std::optional<std::uint64_t> l2 = query_l2_cache_bytes()){
        info.l2_bytes = *l2;
    }

    // Left absent rather than zeroed where the hardware has no L3, so a
    // reader can tell "this machine has none" from "the query failed".
    info.l3_bytes = query_l3_cache_bytes();

    info.sizeof_real = sizeof(Real);

    return info;
}