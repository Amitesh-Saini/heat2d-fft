#pragma once

#include "types.hpp"

struct ValidationConfig {
    Real min_domain_length = Real{1e-4};
    Real min_grid_spacing  = Real{1e-8};
};