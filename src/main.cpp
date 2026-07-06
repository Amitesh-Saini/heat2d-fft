#include "config.hpp"
#include "heat2d_fourier.hpp"
#include "run_config.hpp"
#include "initial_conditions.hpp"
#include "config_io.hpp"


#include <iostream>
#include <cmath>
#include <algorithm>



int main() {
    const auto cfg = make_default_heat2d_config();

    Heat2DFourierSolver solver(cfg);    

    // TODO: build/select initial condition
    // TODO: solver.set_initial_condition(u0)
    // TODO: auto snapshots = solver.solve()
    
return 0;
}