#include "fft2d.hpp"
#include "fft1d.hpp"
#include <cmath>
#include <stdexcept>

// fft2d.cpp
// Responsibility:
//   Implementation of row-column 2D FFT/IFFT.
// What to do here:
//   - Transform rows first because they are contiguous in row-major storage.
//   - Gather each column into a temporary vector, transform it, and scatter back.
//   - Keep this file focused only on multidimensional transform orchestration.


namespace{

    template <typename Transform>

    void fft_2d_kernel(Grid2D<Complex>& field, Transform transform){

    std::size_t nx = field.nx();
    std::size_t ny = field.ny();

    if(nx == 0 || ny == 0) throw std::invalid_argument("fft2d error: nx or ny = 0");
    if(!is_power_of_two(nx) || !is_power_of_two(ny)) throw std::invalid_argument("fft2d error: nx or ny is not a power of 2");

    // row transforms


    ComplexVec row(ny);


    for(std::size_t i = 0; i < nx; ++i){

        for(std::size_t j = 0; j < ny; ++j){
            
            row[j] = field(i, j);
        }

        transform(row);

        for(std::size_t j = 0; j < ny; ++j){

            field(i, j) = row[j];
        }
    }

    // column transforms

    ComplexVec col(nx);

    for(std::size_t j = 0; j < ny; ++j){

        for(std::size_t i = 0; i < nx; ++i){

            col[i] = field(i, j);
        }

        transform(col);

        for(std::size_t i = 0; i < nx; ++i){

            field(i, j) = col[i];
        }
    }
}


}


void fft_2d_inplace(Grid2D<Complex>& field) {
    
    fft_2d_kernel(field, fft_1d_inplace);
}

void ifft_2d_inplace(Grid2D<Complex>& field) {

    fft_2d_kernel(field, ifft_1d_inplace);
}

void fft_2d_row_inplace(Grid2D<Complex>& field){

    std::size_t nx = field.nx();
    std::size_t ny = field.ny();

    if(nx == 0 || ny == 0) throw std::invalid_argument("fft2d error: nx or ny = 0");
    if(!is_power_of_two(nx) || !is_power_of_two(ny)) throw std::invalid_argument("fft2d error: nx or ny is not a power of 2");

    ComplexVec row(ny);

    for(std::size_t i = 0; i < nx; ++i){

        for(std::size_t j = 0; j < ny; ++j){
            
            row[j] = field(i, j);
        }

        fft_1d_inplace(row);

        for(std::size_t j = 0; j < ny; ++j){

            field(i, j) = row[j];
        }
    }
}

void fft_2d_col_inplace(Grid2D<Complex>& field){

    std::size_t nx = field.nx();
    std::size_t ny = field.ny();

    if(nx == 0 || ny == 0) throw std::invalid_argument("fft2d error: nx or ny = 0");
    if(!is_power_of_two(nx) || !is_power_of_two(ny)) throw std::invalid_argument("fft2d error: nx or ny is not a power of 2");

    ComplexVec col(nx);

    for(std::size_t j = 0; j < ny; ++j){

        for(std::size_t i = 0; i < nx; ++i){

            col[i] = field(i, j);
        }

        fft_1d_inplace(col);

        for(std::size_t i = 0; i < nx; ++i){

            field(i, j) = col[i];
        }
    }
}


void ifft_2d_row_inplace(Grid2D<Complex>& field){

    std::size_t nx = field.nx();
    std::size_t ny = field.ny();

    if(nx == 0 || ny == 0) throw std::invalid_argument("fft2d error: nx or ny = 0");
    if(!is_power_of_two(nx) || !is_power_of_two(ny)) throw std::invalid_argument("fft2d error: nx or ny is not a power of 2");

    ComplexVec row(ny);

    for(std::size_t i = 0; i < nx; ++i){

        for(std::size_t j = 0; j < ny; ++j){
            
            row[j] = field(i, j);
        }

        ifft_1d_inplace(row);

        for(std::size_t j = 0; j < ny; ++j){

            field(i, j) = row[j];
        }
    }
}

void ifft_2d_col_inplace(Grid2D<Complex>& field){

    std::size_t nx = field.nx();
    std::size_t ny = field.ny();

    if(nx == 0 || ny == 0) throw std::invalid_argument("fft2d error: nx or ny = 0");
    if(!is_power_of_two(nx) || !is_power_of_two(ny)) throw std::invalid_argument("fft2d error: nx or ny is not a power of 2");

    ComplexVec col(nx);

    for(std::size_t j = 0; j < ny; ++j){

        for(std::size_t i = 0; i < nx; ++i){

            col[i] = field(i, j);
        }

        ifft_1d_inplace(col);

        for(std::size_t i = 0; i < nx; ++i){

            field(i, j) = col[i];
        }
    }
}
