#include "config.hpp"

Heat2DConfig make_default_heat2d_config() {
    Heat2DConfig cfg;
    cfg.grid_size = 256;
    cfg.alpha = 1.0;
    cfg.output_times = {0.0, 0.05, 0.1, 0.5, 1.0};
    return cfg;
}