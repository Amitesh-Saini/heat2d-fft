#include "dft1d.hpp"
#include "dft2d.hpp"
#include <cmath>
#include <stdexcept>


// dft2d.cpp
// Responsibility:
//   Implementation of row-column 2D dFT/IdFT.
// What to do here:
//   - Transform rows first because they are contiguous in row-major storage.
//   - Gather each column into a temporary vector, transform it, and scatter back.
//   - Keep this file focused only on multidimensional transform orchestration.


namespace{

    template <typename Transform>

    Grid2D<Complex>  dft_2d_kernal(const Grid2D<Complex>& field, Transform transform){

    std::size_t nx = field.nx();
    std::size_t ny = field.ny();

    if(nx == 0 || ny == 0) throw std::invalid_argument("dft2d error: nx or ny = 0");

    Grid2D<Complex> transformed_field(nx, ny);

    // row transforms


    ComplexVec row(ny);


    for(std::size_t i = 0; i < nx; ++i){

        for(std::size_t j = 0; j < ny; ++j){
            
            row[j] = field(i, j);
        }

        row = transform(row);

        for(std::size_t j = 0; j < ny; ++j){

            transformed_field(i, j) = row[j];
        }
    }

    // column transforms

    ComplexVec col(nx);

    for(std::size_t j = 0; j < ny; ++j){

        for(std::size_t i = 0; i < nx; ++i){

            col[i] = transformed_field(i, j);
        }

        col = transform(col);

        for(std::size_t i = 0; i < nx; ++i){

            transformed_field(i, j) = col[i];
        }
    }


    return transformed_field;
    
}


}



Grid2D<Complex>  dft_2d(const Grid2D<Complex>& field){

    return dft_2d_kernal(field, dft_1d);
}



Grid2D<Complex>  idft_2d(const Grid2D<Complex>& field){

    return dft_2d_kernal(field, idft_1d);
}