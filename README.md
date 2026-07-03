# heat2d-fft
From-scratch C++ 2D Fourier spectral heat-equation solver using a self-implemented radix-2 FFT


## Build and Test Instructions

### Current Status

The numerical core and test suite are implemented.

The current project includes:

- 1D DFT / inverse DFT
- 1D radix-2 FFT / inverse FFT
- 2D DFT / inverse DFT
- 2D radix-2 FFT / inverse FFT
- Fourier wavenumber construction
- Periodic 2D heat equation solver using Fourier spectral evolution

The command-line JSON/config-file solver interface is still under development. For now, the main thing to run is the test suite. I will let you know when the solver is more user friendly so you can many differnet test cases right now theres alot the user has to do to call and run the solver 

---

## Dependencies

This project uses:

- C++17
- CMake
- FFTW3
- pkg-config
- nlohmann/json

The project uses CMake to find FFTW through `pkg-config`. The JSON dependency is handled by CMake and will be fetched automatically if it is not already installed.

---

Windows setup - install fftw 

sudo apt update
sudo apt install -y pkg-config libfftw3-dev

you can check that FFTW is visible with: pkg-config --modversion fftw3 
Native Windows builds may require extra FFTW and pkg-config setup. For now, WSL/Ubuntu is the recommended Windows route

then clone and build 

git clone <repo-url>
cd <repo-name>

mkdir -p build
cd build

cmake ..
cmake --build .


test execuabtles:

./test_dft1d
./test_fft1d
./test_dft2d
./test_fft2d
./test_wavenumbers
./test_heat2d_fourier

more executables found in the cmake test 

the utils file in tests is where all the helpers reside that both the dft and fft call upon such as grid constructors error helpers etc. 

