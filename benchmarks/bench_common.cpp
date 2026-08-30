#include "bench_common.hpp"

#include "dft1d.hpp"
#include "dft2d.hpp"
#include "fft1d.hpp"
#include "fft2d.hpp"
#include "grid2d.hpp"
#include "heat2d_fourier.hpp"
#include "run_config.hpp"
#include "snapshot_writer.hpp"
#include "timer.hpp"
#include "types.hpp"
#include "wavenumbers.hpp"
#include "timing_registry.hpp"

#include "1D_test_utils.hpp"
#include "2D_test_utils.hpp"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>


std::string transform_to_string(transform transform_type){

    switch (transform_type)
    {
    case transform::DFT:
        return "DFT";

    case transform::FFT:
        return "FFT";

    case transform::FFT_2d:
        return "FFT_2d";

    case transform::FFTW:
        return "FFTW";
    
    case transform::FFTW_2d:
        return "FFTW_2d";
    }
}


std::string benchmark_name_to_string(Benchmark_name name){


    switch (name)
    {
    case Benchmark_name::DFT_1d_time:
        return "DFT_1D_time";
    
    case Benchmark_name::FFT_1d_time:
        return "FFT_1D_time";
    
    case Benchmark_name::FFT_2d_aspect:
        return "FFT_2d_aspect";

    case Benchmark_name::FFT_2d_time:
        return "FFT_2d_time";

    case Benchmark_name::FFTW_1d_time:
        return "FFTW_1d_time";

    case Benchmark_name::FFTW_2d_time:
        return "FFTW_2d_time";

    case Benchmark_name::Full_solver:
        return "Full_solver";

    case Benchmark_name::Memory_footprint:
        return "Memory_footprint";
    
    case Benchmark_name::Numeric_profile:
        return "Numeric_profile";

    case Benchmark_name::Numeric_solver:
        return "Numeric_solver";

    case Benchmark_name::Solver_ic_compare:
        return "Solver_ic_compare";
    }
}


std::uint32_t derive_seed(std::uint32_t base_seed, std::size_t nx, std::size_t ny){

    constexpr std::uint64_t golden = 0x9E3779B97F4A7C15ull;

    std::uint64_t z = static_cast<std::uint64_t>(base_seed);

    z = z * golden + static_cast<std::uint64_t>(nx);
    z = z * golden + static_cast<std::uint64_t>(ny);

    // Splitmax64 finalizer code 

    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = (z ^ (z >> 31));

    return static_cast<std::uint32_t>(z >> 32);
}


ComplexVec make_random_input_1d(std::size_t n, std::uint32_t seed){


    std::mt19937 gen(seed);

    ComplexVec input_vec(n);

    const Real max_inv =  Real{1} / static_cast<Real>(gen.max());

    for(std::size_t i = 0; i < n; ++i){

        const Real re = Real{2} * (static_cast<Real>(gen()) * max_inv) - Real{1};
        const Real im = Real{2} * (static_cast<Real>(gen()) * max_inv) - Real{1};

        input_vec[i] = {re, im};
    }

    return input_vec;

}


Grid2D<Complex> make_random_input_2d(std::size_t nx, std::size_t ny, std::uint32_t seed){


    std::mt19937 gen(seed);

    Grid2D<Complex> input_grid(nx, ny);

    const Real max_inv = Real{1} / static_cast<Real>(gen.max());

    auto& raw_grid = input_grid.raw();
    const std::size_t count = raw_grid.size();

    for(std::size_t i = 0; i < count; ++i){

        const Real re = Real{2} * (static_cast<Real>(gen()) * max_inv) - Real{1};
        const Real im = Real{2} * (static_cast<Real>(gen()) * max_inv) - Real{1};

        raw_grid[i] = {re, im};
    }

    return input_grid;
}



double transform_flops(std::size_t nx, std::size_t ny, transform transform_type){

    switch (transform_type)
    {

    case transform::DFT:
        return (Real{8} * static_cast<Real>(nx) * static_cast<Real>(nx)); 

    case transform::FFT:
        return (Real{5} * static_cast<Real>(nx) * std::log2(static_cast<Real>(nx)));

    case transform::FFTW:
        return (Real{5} * static_cast<Real>(nx) * std::log2(static_cast<Real>(nx)));
    
    case transform::FFT_2d:
        return (Real{5} * static_cast<Real>(nx) * static_cast<Real>(ny) * std::log2(static_cast<Real>(nx) * static_cast<Real>(ny)));
    
    case transform::FFTW_2d:
        return (Real{5} * static_cast<Real>(nx) * static_cast<Real>(ny) * std::log2(static_cast<Real>(nx) * static_cast<Real>(ny)));
    }
}


double solver_flops(std::size_t nx, std::size_t ny, std::size_t num_snapshots){


    double transform_flops = (static_cast<Real>(num_snapshots) + Real{1}) * (Real{5} * static_cast<Real>(nx) * static_cast<Real>(ny) * std::log2(static_cast<Real>(nx) * static_cast<Real>(ny)));
    double decay_flops = static_cast<Real>(num_snapshots) * Real{2} * static_cast<Real>(nx) * static_cast<Real>(ny);

    return (transform_flops + decay_flops);
}


std::uint64_t decay_pass_bytes(std::size_t nx, std::size_t ny, std::size_t num_snapshots){

    return (std::uint64_t{2} * sizeof(Complex) * num_snapshots * nx * ny);
}


std::uint64_t transform_working_set_bytes(std::size_t nx, std::size_t ny, transform transform_type){

    switch (transform_type)
    {
        
    case transform::DFT:
        return (static_cast<std::uint64_t>(nx) * 2 * sizeof(Complex));
    
    case transform::FFT:
        return (static_cast<std::uint64_t>(nx) * sizeof(Complex));

    case transform::FFT_2d:
        return ((static_cast<std::uint64_t>(nx) * ny + nx + ny) * sizeof(Complex));

    case transform::FFTW:
        return (static_cast<std::uint64_t>(nx) * sizeof(fftw_complex)); // inplace implementation
    
    case transform::FFTW_2d:
        return (static_cast<std::uint64_t>(nx) * ny * sizeof(fftw_complex));

    }
}


std::uint64_t solver_working_set_bytes(std::size_t nx, std::size_t ny, std::size_t num_snapshots){

    std::uint64_t points = static_cast<std::uint64_t>(nx) * ny;

    std::uint64_t pre_snap = points * (2 * sizeof(Real)  + sizeof(Complex)) + (static_cast<std::uint64_t>(nx) + ny) * (sizeof(Real) + sizeof(Complex)); // includes ifft

    std::uint64_t snap = points * (sizeof(Real) * (num_snapshots + 1) + sizeof(Complex));

    return (pre_snap + snap);
}


Real relative_linf_error(const ComplexVec& computed, const ComplexVec& reference){

    if(computed.size() != reference.size()){
        throw std::invalid_argument("relative_linf_error: vector sizes differ");
    }

    Real max_error = 0.0;
    Real max_ref = 0.0;

    for(std::size_t k = 0; k < reference.size(); k++){


        max_error = std::max(max_error, std::abs(reference[k] - computed[k]));
        max_ref = std::max(max_ref, std::abs(reference[k]));
    }

     if(max_ref == Real{0}){
        throw std::runtime_error("relative_linf_error: reference norm is zero");
    }

    return max_error / max_ref;

}


Real relative_linf_error(const Grid2D<Complex>& computed, const Grid2D<Complex>& reference){

    if(computed.nx() != reference.nx() || computed.ny() != reference.ny()){
        throw std::invalid_argument("relative_linf_error: grid shapes differ");
    }

    const std::vector<Complex>& c = computed.raw();
    const std::vector<Complex>& r = reference.raw();

    Real max_diff = Real{0};
    Real max_ref  = Real{0};

    for(std::size_t k = 0; k < r.size(); ++k){

        max_diff = std::max(max_diff, std::abs(c[k] - r[k]));
        max_ref  = std::max(max_ref,  std::abs(r[k]));
    }

    if(max_ref == Real{0}){
        throw std::runtime_error("relative_linf_error: reference norm is zero");
    }

    return max_diff / max_ref;
}



Real round_trip_error_2d(const Grid2D<Complex>& input){

    Grid2D<Complex> copied_input = input;

    fft_2d_inplace(copied_input);
    ifft_2d_inplace(copied_input);

    return relative_linf_error(copied_input, input);
}


Real round_trip_error_1d(const ComplexVec& input){
 
    ComplexVec work = input;
 
    fft_1d_inplace(work);
    ifft_1d_inplace(work);
 
    return relative_linf_error(work, input);
}


Real error_tolerance(std::size_t nx, std::size_t ny, transform transform_type, Real safety_factor){

    switch (transform_type)
    {
    case transform::DFT:
        return (safety_factor * std::numeric_limits<Real>::epsilon() * static_cast<Real>(nx));
    
    case transform::FFT:
        return (safety_factor * std::numeric_limits<Real>::epsilon() * std::max(std::log2(nx), Real{1}));

    case transform::FFT_2d:
        return (safety_factor * std::numeric_limits<Real>::epsilon() * std::max(std::log2(nx * ny), Real{1}));
    
    case transform::FFTW:
        return (safety_factor * std::numeric_limits<Real>::epsilon() * std::max(std::log2(nx), Real{1}));

    case transform::FFTW_2d:
        return (safety_factor * std::numeric_limits<Real>::epsilon() * std::max(std::log2(nx * ny), Real{1}));

    }
}


// ---------------------------------------------------------------------------
// Fftw1dPlan
// ---------------------------------------------------------------------------

Fftw1dPlan::Fftw1dPlan(std::size_t n, unsigned planner_flag)
    : n_(n), buffer_(nullptr), forward_(nullptr), inverse_(nullptr){

    if(n == 0){
        throw std::invalid_argument("Fftw1dPlan: n must be nonzero");
    }

    // FFTW's basic interface takes int dimensions, so a size that does not
    // fit in an int cannot be planned. Unreachable at benchmark sizes, but
    // the cast below would otherwise be silently wrong.
    if(n > static_cast<std::size_t>(std::numeric_limits<int>::max())){
        throw std::invalid_argument("Fftw1dPlan: n exceeds FFTW int dimension limit");
    }

    buffer_ = fftw_alloc_complex(n);

    if(buffer_ == nullptr){
        throw std::runtime_error("Fftw1dPlan: FFTW buffer allocation failed");
    }

    // Both plans are in place: the same buffer serves as input and output,
    // matching fft_2d_kernel's semantics and footprint.
    //
    // With FFTW_MEASURE these two calls run and time candidate algorithms,
    // overwriting the buffer in the process. That is why load() must be
    // called after construction and never before.
    forward_ = fftw_plan_dft_1d(static_cast<int>(n), buffer_, buffer_, FFTW_FORWARD, planner_flag);

    if(forward_ == nullptr){

        fftw_free(buffer_);
        buffer_ = nullptr;

        throw std::runtime_error("Fftw1dPlan: FFTW forward plan creation failed");
    }

    inverse_ = fftw_plan_dft_1d(static_cast<int>(n), buffer_, buffer_, FFTW_BACKWARD, planner_flag);

    if(inverse_ == nullptr){

        fftw_destroy_plan(forward_);
        fftw_free(buffer_);

        forward_ = nullptr;
        buffer_ = nullptr;

        throw std::runtime_error("Fftw1dPlan: FFTW inverse plan creation failed");
    }
}


Fftw1dPlan::~Fftw1dPlan(){

    // Plans reference the buffer, so they are destroyed first. Each member is
    // null-checked because the constructor can throw partway through, and a
    // failed construction still runs the destructors of any base or member
    // objects already built.
    if(inverse_ != nullptr){
        fftw_destroy_plan(inverse_);
    }

    if(forward_ != nullptr){
        fftw_destroy_plan(forward_);
    }

    if(buffer_ != nullptr){
        fftw_free(buffer_);
    }
}


void Fftw1dPlan::load(const ComplexVec& input){

    if(input.size() != n_){
        throw std::invalid_argument("Fftw1dPlan::load: input size does not match plan size");
    }

    // fftw_complex is a double[2] with the real part first, the same layout
    // std::complex<double> is guaranteed to have. The copy is explicit rather
    // than a memcpy so the layout assumption stays visible.
    for(std::size_t k = 0; k < n_; ++k){

        buffer_[k][0] = input[k].real();
        buffer_[k][1] = input[k].imag();
    }
}


ComplexVec Fftw1dPlan::store() const{

    ComplexVec output(n_);

    for(std::size_t k = 0; k < n_; ++k){

        output[k] = {buffer_[k][0], buffer_[k][1]};
    }

    return output;
}


void Fftw1dPlan::execute_forward(){

    fftw_execute(forward_);
}


void Fftw1dPlan::execute_inverse(){

    fftw_execute(inverse_);
}


std::size_t Fftw1dPlan::size() const{

    return n_;
}

void Fftw1dPlan::normalize_inverse(){
 
    const Real inv_n = Real{1} / static_cast<Real>(n_);
 
    for(std::size_t k = 0; k < n_; ++k){
 
        buffer_[k][0] *= inv_n;
        buffer_[k][1] *= inv_n;
    }
}


// ---------------------------------------------------------------------------
// Fftw2dPlan
// ---------------------------------------------------------------------------

Fftw2dPlan::Fftw2dPlan(std::size_t nx, std::size_t ny, unsigned planner_flag)
    : nx_(nx), ny_(ny), buffer_(nullptr), forward_(nullptr), inverse_(nullptr){

    if(nx == 0 || ny == 0){
        throw std::invalid_argument("Fftw2dPlan: nx and ny must be nonzero");
    }

    // Each dimension is passed to FFTW as an int, so each must fit
    // individually; the element count itself stays a size_t. Unreachable at
    // benchmark sizes, but the casts below would otherwise be silently wrong.
    const std::size_t int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());

    if(nx > int_max || ny > int_max){
        throw std::invalid_argument("Fftw2dPlan: nx or ny exceeds FFTW int dimension limit");
    }

    buffer_ = fftw_alloc_complex(nx * ny);

    if(buffer_ == nullptr){
        throw std::runtime_error("Fftw2dPlan: FFTW buffer allocation failed");
    }

    // Both plans are in place: the same buffer serves as input and output,
    // matching fft_2d_kernel's semantics and footprint.
    //
    // With FFTW_MEASURE these two calls run and time candidate algorithms,
    // overwriting the buffer in the process. That is why load() must be
    // called after construction and never before.
    //
    // nx is the outer dimension and ny the contiguous one, matching Grid2D's
    // (i,j) -> i*ny + j mapping.
    forward_ = fftw_plan_dft_2d(
        static_cast<int>(nx), static_cast<int>(ny), buffer_, buffer_, FFTW_FORWARD, planner_flag);

    if(forward_ == nullptr){

        fftw_free(buffer_);
        buffer_ = nullptr;

        throw std::runtime_error("Fftw2dPlan: FFTW forward plan creation failed");
    }

    inverse_ = fftw_plan_dft_2d(
        static_cast<int>(nx), static_cast<int>(ny), buffer_, buffer_, FFTW_BACKWARD, planner_flag);

    if(inverse_ == nullptr){

        fftw_destroy_plan(forward_);
        fftw_free(buffer_);

        forward_ = nullptr;
        buffer_ = nullptr;

        throw std::runtime_error("Fftw2dPlan: FFTW inverse plan creation failed");
    }
}


Fftw2dPlan::~Fftw2dPlan(){

    // Plans reference the buffer, so they are destroyed first. Each member is
    // null-checked because the constructor can throw partway through.
    if(inverse_ != nullptr){
        fftw_destroy_plan(inverse_);
    }

    if(forward_ != nullptr){
        fftw_destroy_plan(forward_);
    }

    if(buffer_ != nullptr){
        fftw_free(buffer_);
    }
}


void Fftw2dPlan::load(const Grid2D<Complex>& input){

    if(input.nx() != nx_ || input.ny() != ny_){
        throw std::invalid_argument("Fftw2dPlan::load: input grid shape does not match plan shape");
    }

    // Flat copy: Grid2D stores (i,j) at i*ny + j and FFTW expects the same
    // row-major order with ny contiguous, so element k maps to element k with
    // no index arithmetic.
    const std::vector<Complex>& raw_input = input.raw();

    for(std::size_t k = 0; k < raw_input.size(); ++k){

        buffer_[k][0] = raw_input[k].real();
        buffer_[k][1] = raw_input[k].imag();
    }
}


Grid2D<Complex> Fftw2dPlan::store() const{

    Grid2D<Complex> output(nx_, ny_);

    std::vector<Complex>& raw_output = output.raw();

    for(std::size_t k = 0; k < raw_output.size(); ++k){

        raw_output[k] = {buffer_[k][0], buffer_[k][1]};
    }

    return output;
}


void Fftw2dPlan::execute_forward(){

    fftw_execute(forward_);
}


void Fftw2dPlan::execute_inverse(){

    fftw_execute(inverse_);
}


std::size_t Fftw2dPlan::nx() const{

    return nx_;
}


std::size_t Fftw2dPlan::ny() const{

    return ny_;
}


Complex* Fftw2dPlan::data(){

    return reinterpret_cast<Complex*>(buffer_);
}


const Complex* Fftw2dPlan::data() const{

    return reinterpret_cast<const Complex*>(buffer_);
}

void Fftw2dPlan::normalize_inverse(){
 
    const std::size_t count = nx_ * ny_;
 
    const Real inv_count = Real{1} / static_cast<Real>(count);
 
    for(std::size_t k = 0; k < count; ++k){
 
        buffer_[k][0] *= inv_count;
        buffer_[k][1] *= inv_count;
    }
}


// ---------------------------------------------------------------------------
// FFTW wisdom
// ---------------------------------------------------------------------------

bool import_fftw_wisdom(const std::string& path){

    // fftw_import_wisdom_from_filename returns nonzero on success. A missing
    // or unreadable file is the ordinary first-run case, not an error, so
    // this reports it as false and lets the caller record which way the
    // session went.
    //
    // Wisdom is accumulated in FFTW's global state, so this affects every
    // plan created afterwards in this process. Import before constructing
    // any Fftw1dPlan or Fftw2dPlan.
    return fftw_import_wisdom_from_filename(path.c_str()) != 0;
}


void export_fftw_wisdom(const std::string& path){

    // Exports everything FFTW has learned in this process, so this belongs
    // at the end of a session, after every plan has been created. Exporting
    // partway through writes an incomplete file that later runs would
    // silently only partly benefit from.
    //
    // A failed write is worth surfacing: the caller asked to pin the plan
    // selection for future sessions, and silently not doing so would make a
    // later unexplained timing shift impossible to account for.
    if(fftw_export_wisdom_to_filename(path.c_str()) == 0){

        throw std::runtime_error("export_fftw_wisdom: failed to write wisdom file: " + path);
    }
}



std::size_t choose_reps(std::uint64_t single_transform_ns, const rep_policy& policy){

    if(single_transform_ns >= policy.single_call_ns) return 1;
    
    if(single_transform_ns == 0){

        return policy.max_reps;
    }

    const std::uint64_t reps = (policy.min_timed_ns + single_transform_ns - 1) / single_transform_ns; // a + b - 1 can overflow if a is near the type's maximum.

    const std::uint64_t capped = std::min(reps, static_cast<std::uint64_t>(policy.max_reps));

    return static_cast<std::size_t>(capped);
}


timed_batch time_transform_1d(const ComplexVec& input, transform transform_type, std::size_t reps){

    if(reps == 0){
        throw std::invalid_argument("time_transform_1d: reps must be nonzero");
    }

    timed_batch batch;

    batch.reps_used = reps;
    batch.paired = false;

    Timer timer;

    switch(transform_type){

    case transform::DFT:{

        ComplexVec output;

        timer.reset();

        for(std::size_t rep = 0; rep < reps; ++rep){

            output = dft_1d(input);
        }

        batch.total_time_ns = timer.elapsed_ns();

        // Nothing downstream consumes the transform's output, so without a
        // read the compiler is entitled to delete the call and leave a timed
        // region that measures an empty loop. The volatile forces the store.
        if(!output.empty()){

            volatile Real guard = output[0].real();
            (void)guard;
        }

        return batch;
    }

    case transform::FFT:{

        ComplexVec work = input;

        if(reps == 1){

            timer.reset();

            fft_1d_inplace(work);

            batch.total_time_ns = timer.elapsed_ns();

            if(!work.empty()){

                volatile Real guard = work[0].real();
                (void)guard;
            }

            return batch;
        }

        // reps > 1: alternate forward and inverse. Because the 1/n
        // normalization lives entirely in the inverse, each forward/inverse
        // pair returns the buffer to within roundoff of its original values,
        // so no restoring copy is needed inside the timed region and the
        // magnitudes stay stable rather than growing or decaying toward
        // denormals.
        //
        // Roundoff does accumulate across pairs, on the order of
        // reps * eps * log2(n). That does not affect the measurement: the
        // transform performs the same butterflies regardless of the values,
        // with no data-dependent branching, and the error metric for this
        // configuration is computed separately on a fresh input.
        //
        // The cost is that the row is a forward/inverse average rather than a
        // forward, which is what paired records.
        batch.paired = true;

        Real initial_norm = Real{0};

        for(std::size_t k = 0; k < work.size(); ++k){

            initial_norm = std::max(initial_norm, std::abs(work[k]));
        }

        timer.reset();

        for(std::size_t rep = 0; rep < reps; ++rep){

            if(rep % 2 == 0){
                fft_1d_inplace(work);
            }
            else{
                ifft_1d_inplace(work);
            }
        }

        batch.total_time_ns = timer.elapsed_ns();

        if(!work.empty()){

            volatile Real guard = work[0].real();
            (void)guard;
        }

        // Outside the timed region: confirm the pairing actually held the
        // buffer near its original magnitude. If this ever fires, the
        // measurement was running on values that had drifted far enough to
        // change the arithmetic's cost, and the timing cannot be trusted.
        // An odd rep count leaves one unpaired forward, which scales by n, so
        // the check only applies when the count is even.
        if(reps % 2 == 0){

            Real final_norm = Real{0};

            for(std::size_t k = 0; k < work.size(); ++k){

                final_norm = std::max(final_norm, std::abs(work[k]));
            }

            if(final_norm > Real{2} * initial_norm || final_norm < initial_norm / Real{2}){

                throw std::runtime_error(
                    "time_transform_1d: paired batch drifted in magnitude; timing is unreliable");
            }
        }

        return batch;
    }

    case transform::FFTW:

        throw std::invalid_argument(
            "time_transform_1d: FFTW is timed through time_fftw_1d, which takes a prebuilt plan");

    case transform::FFT_2d:

        throw std::invalid_argument(
            "time_transform_1d: FFT_2d is timed through time_transform_2d");

    case transform::FFTW_2d:

        throw std::invalid_argument(
            "time_transform_1d: FFTW_2d is timed through time_fftw_2d");
    }

    throw std::invalid_argument("time_transform_1d: unrecognized transform");
}

// ---------------------------------------------------------------------------
// time_fftw_1d
// ---------------------------------------------------------------------------
 
timed_batch time_fftw_1d(Fftw1dPlan& plan, std::size_t reps){
 
    if(reps == 0){
        throw std::invalid_argument("time_fftw_1d: reps must be nonzero");
    }
 
    timed_batch batch;
 
    batch.reps_used = reps;
    batch.paired = false;
 
    Timer timer;
 
    if(reps == 1){
 
        timer.reset();
 
        plan.execute_forward();
 
        batch.total_time_ns = timer.elapsed_ns();
 
        return batch;
    }
 
    // reps > 1: the transform is in place and FFTW's forward is unnormalized,
    // so repeated forwards would multiply the data by n on every rep and
    // reach infinity within a few dozen reps at the larger sizes. Alternating
    // forward and inverse is not sufficient on its own either, because FFTW's
    // backward transform is also unnormalized: a pair scales by n rather than
    // returning the buffer.
    //
    // normalize_inverse() supplies the missing 1/n. That is not a handicap:
    // the custom ifft_1d_inplace performs the same scaling pass internally as
    // part of its inverse, so including it here makes the two inverses do
    // equivalent work. It is inside the timed region for exactly that reason.
    //
    // With the normalization in place each pair returns the buffer to within
    // roundoff, magnitudes stay stable, and the row becomes a forward/inverse
    // average rather than a forward, which is what paired records.
    batch.paired = true;
 
    const ComplexVec initial = plan.store();
 
    Real initial_norm = Real{0};
 
    for(std::size_t k = 0; k < initial.size(); ++k){
 
        initial_norm = std::max(initial_norm, std::abs(initial[k]));
    }
 
    timer.reset();
 
    for(std::size_t rep = 0; rep < reps; ++rep){
 
        if(rep % 2 == 0){
 
            plan.execute_forward();
        }
        else{
 
            plan.execute_inverse();
            plan.normalize_inverse();
        }
    }
 
    batch.total_time_ns = timer.elapsed_ns();
 
    // Outside the timed region: confirm the pairing actually held the buffer
    // near its original magnitude. If this fires, the measurement was running
    // on values that had drifted or blown up, and the timing cannot be
    // trusted. An odd rep count leaves one unpaired forward, which scales by
    // n, so the check only applies when the count is even.
    if(reps % 2 == 0){
 
        const ComplexVec final_values = plan.store();
 
        Real final_norm = Real{0};
 
        for(std::size_t k = 0; k < final_values.size(); ++k){
 
            final_norm = std::max(final_norm, std::abs(final_values[k]));
        }
 
        if(final_norm > Real{2} * initial_norm || final_norm < initial_norm / Real{2}){
 
            throw std::runtime_error(
                "time_fftw_1d: paired batch drifted in magnitude; timing is unreliable");
        }
    }
 
    return batch;
}


timed_batch_2d time_transform_2d(const Grid2D<Complex>& grid, transform transform_type, std::size_t reps){

    if(reps == 0){
        throw std::invalid_argument("time_transform_2d: reps must be nonzero");
    }

    timed_batch_2d batch;

    batch.reps_used = reps;
    batch.paired = false;

    Timer timer;

    switch(transform_type){

    case transform::FFT_2d:{

        // The total and the row/column split are measured in two separate
        // loops rather than by nesting timers inside one.
        //
        // Nesting would charge the total for two extra clock reads per rep,
        // and it would measure the split invocation rather than the composed
        // one: fft_2d_inplace allocates its row and column buffers once,
        // while calling the two passes separately allocates in each. The
        // total is the number benchmark 3 plots as its scaling curve, so it
        // is measured on the production path.
        //
        // The cost is that this function does the work twice. In exchange,
        // total minus row minus column is a real residual covering the
        // difference between composed and separate invocation, rather than an
        // artifact of clock overhead.

        Grid2D<Complex> work = grid;

        if(reps == 1){

            timer.reset();

            fft_2d_inplace(work);

            batch.total_time_ns = timer.elapsed_ns();

            if(!work.raw().empty()){

                volatile Real guard = work.raw()[0].real();
                (void)guard;
            }

            Grid2D<Complex> split_work = grid;

            timer.reset();

            fft_2d_row_inplace(split_work);

            batch.row_time_ns = timer.elapsed_ns();

            timer.reset();

            fft_2d_col_inplace(split_work);

            batch.col_time_ns = timer.elapsed_ns();

            if(!split_work.raw().empty()){

                volatile Real guard = split_work.raw()[0].real();
                (void)guard;
            }

            return batch;
        }

        // reps > 1: alternate forward and inverse. The 1/(nx*ny)
        // normalization lives entirely in the inverse, so each pair returns
        // the buffer to within roundoff of its original values. No restoring
        // copy is needed inside the timed region and the magnitudes stay
        // stable rather than growing toward overflow.
        //
        // Roundoff accumulates across pairs, on the order of
        // reps * eps * log2(nx*ny). That does not affect the measurement:
        // the transform performs the same butterflies regardless of the
        // values, and the error metric for this configuration is computed
        // separately on a fresh input.
        //
        // The row and column passes are alternated the same way in the second
        // loop, so their accumulated times cover forward and inverse passes
        // in the same proportion as the total does.
        batch.paired = true;

        Real initial_norm = Real{0};

        for(std::size_t k = 0; k < work.raw().size(); ++k){

            initial_norm = std::max(initial_norm, std::abs(work.raw()[k]));
        }

        timer.reset();

        for(std::size_t rep = 0; rep < reps; ++rep){

            if(rep % 2 == 0){
                fft_2d_inplace(work);
            }
            else{
                ifft_2d_inplace(work);
            }
        }

        batch.total_time_ns = timer.elapsed_ns();

        if(!work.raw().empty()){

            volatile Real guard = work.raw()[0].real();
            (void)guard;
        }

        // Outside the timed region: confirm the pairing held the buffer near
        // its original magnitude. If this fires, the measurement ran on
        // values that had drifted far enough to change the arithmetic's cost.
        // An odd rep count leaves one unpaired forward, which scales by
        // nx*ny, so the check only applies when the count is even.
        if(reps % 2 == 0){

            Real final_norm = Real{0};

            for(std::size_t k = 0; k < work.raw().size(); ++k){

                final_norm = std::max(final_norm, std::abs(work.raw()[k]));
            }

            if(final_norm > Real{2} * initial_norm || final_norm < initial_norm / Real{2}){

                throw std::runtime_error(
                    "time_transform_2d: paired batch drifted in magnitude; timing is unreliable");
            }
        }

        // Second loop: the same sequence of transforms, but invoked as
        // separate row and column passes so each can be timed on its own.
        Grid2D<Complex> split_work = grid;

        std::uint64_t row_total_ns = 0;
        std::uint64_t col_total_ns = 0;

        for(std::size_t rep = 0; rep < reps; ++rep){

            if(rep % 2 == 0){

                timer.reset();
                fft_2d_row_inplace(split_work);
                row_total_ns += timer.elapsed_ns();

                timer.reset();
                fft_2d_col_inplace(split_work);
                col_total_ns += timer.elapsed_ns();
            }
            else{

                timer.reset();
                ifft_2d_row_inplace(split_work);
                row_total_ns += timer.elapsed_ns();

                timer.reset();
                ifft_2d_col_inplace(split_work);
                col_total_ns += timer.elapsed_ns();
            }
        }

        batch.row_time_ns = row_total_ns;
        batch.col_time_ns = col_total_ns;

        if(!split_work.raw().empty()){

            volatile Real guard = split_work.raw()[0].real();
            (void)guard;
        }

        return batch;
    }

    case transform::DFT:

        throw std::invalid_argument(
            "time_transform_2d: DFT is one dimensional and is timed through time_transform_1d");

    case transform::FFT:

        throw std::invalid_argument(
            "time_transform_2d: FFT is one dimensional and is timed through time_transform_1d");

    case transform::FFTW:

        throw std::invalid_argument(
            "time_transform_2d: FFTW is one dimensional and is timed through time_fftw_1d");

    case transform::FFTW_2d:

        throw std::invalid_argument(
            "time_transform_2d: FFTW_2d is timed through time_fftw_2d, which takes a prebuilt plan");
    }

    throw std::invalid_argument("time_transform_2d: unrecognized transform");
}

 
// ---------------------------------------------------------------------------
// time_fftw_2d
// ---------------------------------------------------------------------------
 
timed_batch time_fftw_2d(Fftw2dPlan& plan, std::size_t reps){
 
    if(reps == 0){
        throw std::invalid_argument("time_fftw_2d: reps must be nonzero");
    }
 
    timed_batch batch;
 
    batch.reps_used = reps;
    batch.paired = false;
 
    Timer timer;
 
    // No dead-code guard is needed here. fftw_execute is an opaque call into
    // a separately compiled library, so the compiler cannot prove the work is
    // unobserved and cannot delete it.
    //
    // The return type is timed_batch rather than timed_batch_2d: FFTW does
    // not expose its internal pass structure, so there are no row and column
    // times to report for these rows.
 
    if(reps == 1){
 
        timer.reset();
 
        plan.execute_forward();
 
        batch.total_time_ns = timer.elapsed_ns();
 
        return batch;
    }
 
    // reps > 1: the transform is in place and FFTW's forward is unnormalized,
    // so repeated forwards would multiply the data by nx*ny on every rep and
    // overflow within a handful of reps at the larger sizes. Alternating
    // forward and inverse is not sufficient on its own either, because FFTW's
    // backward transform is also unnormalized: a pair scales by nx*ny rather
    // than returning the buffer.
    //
    // normalize_inverse() supplies the missing 1/(nx*ny). That is not a
    // handicap: the custom ifft_2d_inplace performs the same scaling pass
    // internally as part of its inverse, so including it here makes the two
    // inverses do equivalent work. It is inside the timed region for exactly
    // that reason.
    //
    // With the normalization in place each pair returns the buffer to within
    // roundoff, magnitudes stay stable, and the row becomes a forward/inverse
    // average rather than a forward, which is what paired records.
    batch.paired = true;
 
    const Grid2D<Complex> initial = plan.store();
 
    Real initial_norm = Real{0};
 
    for(std::size_t k = 0; k < initial.raw().size(); ++k){
 
        initial_norm = std::max(initial_norm, std::abs(initial.raw()[k]));
    }
 
    timer.reset();
 
    for(std::size_t rep = 0; rep < reps; ++rep){
 
        if(rep % 2 == 0){
 
            plan.execute_forward();
        }
        else{
 
            plan.execute_inverse();
            plan.normalize_inverse();
        }
    }
 
    batch.total_time_ns = timer.elapsed_ns();
 
    // Outside the timed region: confirm the pairing held the buffer near its
    // original magnitude. If this fires, the measurement ran on values that
    // had drifted or blown up, and the timing cannot be trusted. An odd rep
    // count leaves one unpaired forward, which scales by nx*ny, so the check
    // only applies when the count is even.
    if(reps % 2 == 0){
 
        const Grid2D<Complex> final_values = plan.store();
 
        Real final_norm = Real{0};
 
        for(std::size_t k = 0; k < final_values.raw().size(); ++k){
 
            final_norm = std::max(final_norm, std::abs(final_values.raw()[k]));
        }
 
        if(final_norm > Real{2} * initial_norm || final_norm < initial_norm / Real{2}){
 
            throw std::runtime_error(
                "time_fftw_2d: paired batch drifted in magnitude; timing is unreliable");
        }
    }
 
    return batch;
}
 

solver_timing time_solve(const Grid2D<Real>& initial_condition, const Heat2DConfig& solver_config,
    std::vector<Grid2D<Real>>& snapshots_out){
 
    solver_timing timing;
 
    // Construction and set_initial_condition are outside the clock: they
    // validate the configuration and copy the initial field, neither of which
    // is part of the solve.
    Heat2DFourierSolver solver(solver_config);
 
    solver.set_initial_condition(initial_condition);
 
    // The registry accumulates process-wide, so it must be cleared here or
    // the phase totals would carry over from the previous trial.
    timing::reset();
 
    Timer timer;
 
    timer.reset();
 
    snapshots_out = solver.solve();
 
    timing.solve_time_ns = timer.elapsed_ns();
 
    // Region names come from the constants heat2d_fourier.cpp declares, so
    // the strings are not typed in two places and cannot drift apart.
    //
    // These read as zero in a build without HEAT2D_ENABLE_TIMING. That is
    // expected rather than an error: the caller records timing::enabled() in
    // the run metadata so an uninstrumented run is distinguishable from one
    // where a phase genuinely took no time.
    timing.forward_transform_time_ns = timing::elapsed_ns(heat2d_regions::forward_fft);
    timing.spectral_copy_time_ns = timing::elapsed_ns(heat2d_regions::spectral_copy);
    timing.decay_time_ns = timing::elapsed_ns(heat2d_regions::decay);
    timing.inverse_transform_time_ns = timing::elapsed_ns(heat2d_regions::inverse_fft);
 
    // The decay and inverse regions run once per output time, so a count that
    // does not match is a sign the annotations and the loop have diverged.
    // Only checkable when the build is instrumented.
    if(timing::enabled()){
 
        const std::uint64_t expected = static_cast<std::uint64_t>(solver_config.output_times.size());
 
        if(timing::call_count(heat2d_regions::decay) != expected){
 
            throw std::runtime_error(
                "time_solve: decay region ran a different number of times than there are output times");
        }
    }
 
    return timing;
}
 


io_timing time_snapshot_write(const std::vector<Grid2D<Real>>& snapshots, const RealVec& x, const RealVec& y,
    const RealVec& times, const std::string& output_path, int gzip_level){
 
    if(snapshots.size() != times.size()){
 
        throw std::invalid_argument("time_snapshot_write: snapshot count does not match times");
    }
 
    if(snapshots.empty()){
 
        throw std::invalid_argument("time_snapshot_write: nothing to write");
    }
 
    io_timing timing;
 
    // SnapshotWriter takes a whole RunConfig, of which this needs only the
    // output settings. The rest is left at its defaults rather than adding a
    // narrower constructor: a second construction path used only by the
    // benchmark would mean these numbers describe code no user runs. The
    // half-empty config is the honest cost of measuring the production
    // writer.
    RunConfig config;
 
    config.schema_version = current_schema_version;
    config.output.output_path = output_path;
    config.output.overwrite = true;
    config.output.gzip_level = gzip_level;
 
    config.solver.nx = snapshots.front().nx();
    config.solver.ny = snapshots.front().ny();
 
    // /config_json would otherwise be empty, which both misreports the file's
    // size and leaves a several-hundred-megabyte file on disk with nothing
    // saying what produced it. A short identifying note costs nothing and
    // keeps the field's size realistic.
    std::ostringstream note;
 
    note << "{\"source\":\"benchmark\",\"benchmark\":\"time_snapshot_write\""
         << ",\"nx\":" << config.solver.nx
         << ",\"ny\":" << config.solver.ny
         << ",\"snapshots\":" << snapshots.size()
         << ",\"gzip_level\":" << gzip_level << "}";
 
    config.source_json_text = note.str();
 
    const RunProvenance provenance = make_run_provenance(config);
 
    Timer timer;
 
    // The clock starts before construction: creating the file and writing the
    // provenance attributes is part of what writing a run costs.
    timer.reset();
 
    {
        SnapshotWriter writer(config, provenance);
 
        writer.write_grids(x, y, times);
 
        for(std::size_t k = 0; k < snapshots.size(); ++k){
 
            writer.append_snapshot(snapshots[k]);
        }
 
        // Timed separately as well as inside the total. finalize() writes the
        // remaining datasets, records the wall-time attribute, verifies the
        // snapshot count, and flushes; how much of the I/O cost sits here
        // rather than in the appends is one of the questions this benchmark
        // exists to answer.
        //
        // The wall time handed to finalize() is the elapsed time so far. It
        // goes into the file as an attribute and does not affect the
        // measurement.
        const std::uint64_t before_finalize_ns = timer.elapsed_ns();
 
        writer.finalize(static_cast<double>(before_finalize_ns) / 1e9);
 
        timing.finalize_time_ns = timer.elapsed_ns() - before_finalize_ns;
 
        // The writer leaves scope here, closing the file. That destruction is
        // inside the timed region because an unclosed file is not a written
        // one.
    }
 
    timing.io_time_ns = timer.elapsed_ns();
 
    // Read after the file is closed, so the size is final.
    timing.bytes_written = static_cast<std::uint64_t>(std::filesystem::file_size(output_path));
 
    // Removed after measuring. A sweep that left a few hundred megabytes per
    // configuration would fill the disk partway through, and later writes
    // would then be measured on a nearly full filesystem. Writing into free
    // space is also the reproducible case: overwriting an existing large file
    // may or may not reuse its blocks depending on what ran before.
    //
    // The removal is after the timing is read, so filesystem cleanup work is
    // never inside a measurement.
    std::error_code remove_error;
 
    std::filesystem::remove(output_path, remove_error);
 
    // A failed removal is not worth throwing over: the measurement is already
    // complete and correct. It does mean the next configuration writes into a
    // fuller filesystem, so it is worth surfacing rather than swallowing
    // silently, which the caller does by checking the file is gone.
 
    return timing;
}
 