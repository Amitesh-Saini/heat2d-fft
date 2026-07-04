// config.cpp
// Responsibility:
//   Define default solver configurations for standard runs.

#include "config.hpp"

Heat2DConfig make_default_heat2d_config() {
    Heat2DConfig cfg;

    cfg.nx = 512;
    cfg.ny = 512;   

    cfg.Lx = Real{2.0};
    cfg.Ly = Real{2.0};

    cfg.alpha = Real{1.0};

    cfg.output_times = {
        Real{0.0},
        Real{0.05},
        Real{0.10},
        Real{0.50},
        Real{1.00}
    };

    return cfg;
}