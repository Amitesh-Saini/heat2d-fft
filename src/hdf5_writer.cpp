// hdf5_writer.cpp
// Responsibility:
//   Implements Hdf5File, the RAII wrapper over the HDF5 C API declared in
//   hdf5_writer.hpp. Low-level storage backend only: files, datasets,
//   attributes, and the chunked extensible snapshot dataset. Knows nothing
//   about the run layout (/x, /y, /times, ...), which is snapshot_writer's job.
//
// HDF5 C API conventions used here:
//   - Every HDF5 object is referred to by an integer handle (hid_t). Handles
//     are resources: each one created must be released with the matching
//     H5*close call.
//   - The C API signals failure with a negative return value; it never throws.
//     Every call is therefore checked and converted into a C++ exception.
//   - Short-lived handles (dataspaces, datatypes, property lists, attributes)
//     are owned by the ScopedHid guard below, so they close automatically on
//     every exit path, including thrown exceptions. Only the file handle and
//     the snapshot dataset handle are long-lived members.

#include "hdf5_writer.hpp"

#include <stdexcept>
#include <string>

namespace {

// RAII guard wrapper
// This removes the need for manual close calls on every error path.

class ScopedHid{
public:
    using CloseFn = herr_t (*)(hid_t);

    ScopedHid(hid_t id, CloseFn close_fn) : id_(id), close_fn_(close_fn) {}

    ~ScopedHid() {
        if (id_ >= 0) close_fn_(id_);
    }

    ScopedHid(const ScopedHid&) = delete;
    ScopedHid& operator=(const ScopedHid&) = delete;

    hid_t get() const { return id_; }
    bool valid() const { return id_ >= 0; }

    // Gives up ownership: the handle will NOT be closed by this guard.
    // Used when a created handle must outlive the function (the snapshot
    // dataset handle, which becomes a class member).
    hid_t release() {
        const hid_t out = id_;
        id_ = H5I_INVALID_HID;
        return out;
    }

private:
    hid_t id_;
    CloseFn close_fn_;
};

} // namespace


Hdf5File::Hdf5File(const std::string& path, bool overwrite){

    const unsigned flags = overwrite ? H5F_ACC_TRUNC : H5F_ACC_EXCL;

    file_id_ = H5Fcreate(path.c_str(), flags, H5P_DEFAULT, H5P_DEFAULT);

    if(file_id_ < 0){

        throw std::runtime_error("Hdf5File: could not create '" + path + "'" + (overwrite ? "" : " (file already exists; set overwrite to replace it)"));
    }
}


Hdf5File::~Hdf5File(){
    
    if (snapshot_dataset_id_ >= 0) H5Dclose(snapshot_dataset_id_);
    if (file_id_ >= 0) H5Fclose(file_id_);
}



// Simple datasets


void Hdf5File::write_real_vector(const std::string& dataset_name, const RealVec& values){

    const hsize_t dims[1] = {static_cast<hsize_t>(values.size())};

    ScopedHid space(H5Screate_simple(1, dims, nullptr), H5Sclose);

    if(!space.valid()){
        
        throw std::runtime_error("Hdf5File::write_real_vector: could not create dataspace for '" + dataset_name + "'");
    }

    ScopedHid dataset(H5Dcreate2(file_id_, dataset_name.c_str(), H5T_NATIVE_DOUBLE, space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);

    if(!dataset.valid()){

        throw std::runtime_error("Hdf5File::write_real_vector: could not create dataset '" + dataset_name + "'");
    }

    
    if(H5Dwrite(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0){
        
        throw std::runtime_error("Hdf5File::write_real_vector: could not write dataset '" + dataset_name + "'");
    }
}


void Hdf5File::write_string(const std::string& dataset_name, const std::string& text){

    // HDF5 has no built-in string type: copy the C-string base type and mark
    
    ScopedHid type(H5Tcopy(H5T_C_S1), H5Tclose);

    if(!type.valid()){

        throw std::runtime_error("Hdf5File::write_string: could not create string datatype for '" + dataset_name + "'");
    }

    if(H5Tset_size(type.get(), H5T_VARIABLE) < 0){
        
        throw std::runtime_error("Hdf5File::write_string: could not set variable-length size for '" + dataset_name + "'");
    }

    // Scalar dataspace: the dataset holds exactly one value (one string).
    ScopedHid space(H5Screate(H5S_SCALAR), H5Sclose);

    if(!space.valid()){

        throw std::runtime_error("Hdf5File::write_string: could not create dataspace for '" + dataset_name + "'");
    }

    ScopedHid dataset(H5Dcreate2(file_id_, dataset_name.c_str(), type.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);

    if(!dataset.valid()){

        throw std::runtime_error("Hdf5File::write_string: could not create dataset '" + dataset_name + "'");
    }

    // Variable-length strings are written as a pointer to the char pointer.
    const char* c_text = text.c_str();

    if(H5Dwrite(dataset.get(), type.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &c_text) < 0){

        throw std::runtime_error("Hdf5File::write_string: could not write dataset '" + dataset_name + "'");
    }
}

void Hdf5File::create_group(const std::string& group_name) {

    ScopedHid group(H5Gcreate2(file_id_, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);

    if(!group.valid()){
        
        throw std::runtime_error("Hdf5File::create_group: could not create group '" + group_name + "'");
    }

    // The group is closed by the guard on scope exit. HDF5 keeps the group in
    // the file; the handle is only needed to create it. Datasets written later
    // with a "/group/name" path resolve against the file, not this handle.
}



// Root-group attributes (provenance / run summary)


void Hdf5File::write_root_string_attribute(const std::string& name, const std::string& value){

    ScopedHid type(H5Tcopy(H5T_C_S1), H5Tclose);

    if(!type.valid()){

        throw std::runtime_error("Hdf5File::write_root_string_attribute: could not create datatype for '" + name + "'");
    }

    if(H5Tset_size(type.get(), H5T_VARIABLE) < 0){

        throw std::runtime_error("Hdf5File::write_root_string_attribute: could not set size for '" + name + "'");
    }

    ScopedHid space(H5Screate(H5S_SCALAR), H5Sclose);

    if(!space.valid()){
        
        throw std::runtime_error("Hdf5File::write_root_string_attribute: could not create dataspace for '" + name + "'");
    }

    // Attaching an attribute to the file handle attaches it to the root group.
    ScopedHid attribute(H5Acreate2(file_id_, name.c_str(), type.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose);

    if(!attribute.valid()){

        throw std::runtime_error("Hdf5File::write_root_string_attribute: could not create attribute '" + name + "'");
    }

    const char* c_value = value.c_str();

    if(H5Awrite(attribute.get(), type.get(), &c_value) < 0){

        throw std::runtime_error("Hdf5File::write_root_string_attribute: could not write attribute '" + name + "'");
    }
}


void Hdf5File::write_root_real_attribute(const std::string& name, Real value){

    ScopedHid space(H5Screate(H5S_SCALAR), H5Sclose);

    if(!space.valid()){

        throw std::runtime_error("Hdf5File::write_root_real_attribute: could not create dataspace for '" + name + "'");
    }

    ScopedHid attribute(H5Acreate2(file_id_, name.c_str(), H5T_NATIVE_DOUBLE, space.get(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose);

    if(!attribute.valid()){

        throw std::runtime_error("Hdf5File::write_root_real_attribute: could not create attribute '" + name + "'");
    }

    if(H5Awrite(attribute.get(), H5T_NATIVE_DOUBLE, &value) < 0){

        throw std::runtime_error("Hdf5File::write_root_real_attribute: could not write attribute '" + name + "'");
    }
}



// Snapshot dataset: create + append


void Hdf5File::create_snapshot_dataset(const std::string& dataset_name, std::size_t nx, std::size_t ny, int gzip_level) {

    if(snapshot_dataset_id_ >= 0){

        throw std::runtime_error("Hdf5File::create_snapshot_dataset: snapshot dataset already created");
    }

    const hsize_t hnx = static_cast<hsize_t>(nx);
    const hsize_t hny = static_cast<hsize_t>(ny);

    // Current shape (0, nx, ny): no snapshots yet. Max shape unlimited along
    
    const hsize_t dims[3]     = { 0, hnx, hny };
    const hsize_t max_dims[3] = { H5S_UNLIMITED, hnx, hny };

    ScopedHid space(H5Screate_simple(3, dims, max_dims), H5Sclose);
    
    if(!space.valid()){

        throw std::runtime_error("Hdf5File::create_snapshot_dataset: could not create dataspace");
    }

    
    // One chunk per snapshot makes both appending and reading a single time slice efficient.

    ScopedHid plist(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);

    if(!plist.valid()){

        throw std::runtime_error("Hdf5File::create_snapshot_dataset: could not create property list");
    }

    const hsize_t chunk_dims[3] = { 1, hnx, hny };

    if(H5Pset_chunk(plist.get(), 3, chunk_dims) < 0){

        throw std::runtime_error("Hdf5File::create_snapshot_dataset: could not set chunking");
    }

    if(gzip_level > 0){

        // Shuffle reorders bytes so neighbouring doubles' shared exponent
        // bytes group together, dramatically improving gzip on smooth fields.
        // Filter order matters: shuffle must be added before deflate.

        if(H5Pset_shuffle(plist.get()) < 0){

            throw std::runtime_error("Hdf5File::create_snapshot_dataset: could not enable shuffle filter");
        }

        if(H5Pset_deflate(plist.get(), static_cast<unsigned>(gzip_level)) < 0){

            throw std::runtime_error("Hdf5File::create_snapshot_dataset: could not enable gzip compression");
        }
    }

    ScopedHid dataset(H5Dcreate2(file_id_, dataset_name.c_str(), H5T_NATIVE_DOUBLE, space.get(), H5P_DEFAULT, plist.get(), H5P_DEFAULT), H5Dclose);

    if(!dataset.valid()){

        throw std::runtime_error("Hdf5File::create_snapshot_dataset: could not create dataset '" + dataset_name + "'");
    }

    // The dataset handle must outlive this function (append_snapshot writes
    // into it repeatedly), so ownership transfers from the guard to the member.
    snapshot_dataset_id_ = dataset.release();
    snapshot_nx_ = nx;
    snapshot_ny_ = ny;
    snapshots_written_ = 0;
}


void Hdf5File::append_snapshot(const Grid2D<Real>& snapshot) {

    if(snapshot_dataset_id_ < 0){

        throw std::runtime_error("Hdf5File::append_snapshot: snapshot dataset has not been created");
    }

    if(snapshot.nx() != snapshot_nx_ || snapshot.ny() != snapshot_ny_){
        
        throw std::invalid_argument("Hdf5File::append_snapshot: snapshot shape does not match dataset shape");
    }

    const hsize_t hnx = static_cast<hsize_t>(snapshot_nx_);
    const hsize_t hny = static_cast<hsize_t>(snapshot_ny_);

    // Step 1: grow the dataset by one along the time axis.
    const hsize_t new_dims[3] = {static_cast<hsize_t>(snapshots_written_ + 1), hnx, hny };

    if(H5Dset_extent(snapshot_dataset_id_, new_dims) < 0){
        
        throw std::runtime_error("Hdf5File::append_snapshot: could not extend snapshot dataset");
    }

    // Step 2: select the newly added slab in the (now larger) file dataspace.
    
    ScopedHid file_space(H5Dget_space(snapshot_dataset_id_), H5Sclose);

    if(!file_space.valid()){
        
        throw std::runtime_error("Hdf5File::append_snapshot: could not get file dataspace");
    }

    const hsize_t offset[3] = { static_cast<hsize_t>(snapshots_written_), 0, 0 };
    const hsize_t count[3]  = { 1, hnx, hny };

    if(H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET, offset, nullptr, count, nullptr) < 0){
        
        throw std::runtime_error("Hdf5File::append_snapshot: could not select hyperslab");
    }

    // Step 3: describe the in-memory source (one contiguous (nx, ny) grid,
    // row-major, matching Grid2D's layout) and write it into the selection.

    const hsize_t mem_dims[2] = { hnx, hny };

    ScopedHid mem_space(H5Screate_simple(2, mem_dims, nullptr), H5Sclose);
    
    if(!mem_space.valid()){

        throw std::runtime_error("Hdf5File::append_snapshot: could not create memory dataspace");
    }

    if(H5Dwrite(snapshot_dataset_id_, H5T_NATIVE_DOUBLE, mem_space.get(), file_space.get(), H5P_DEFAULT, snapshot.raw().data()) < 0){

        throw std::runtime_error("Hdf5File::append_snapshot: could not write snapshot slab");
    }

    ++snapshots_written_;
}


// Small accessors


std::size_t Hdf5File::num_snapshots_written() const{

    return snapshots_written_;
}


void Hdf5File::flush(){

    if(H5Fflush(file_id_, H5F_SCOPE_LOCAL) < 0){
        
        throw std::runtime_error("Hdf5File::flush: could not flush file buffers");
    }
}