#ifndef PNL_H
#define PNL_H

#include <cstdio>
#ifdef _NVHPC_STDPAR_GPU
// These headers are not strictly necessary but used for performance
// tuning.
#include <cuda_runtime.h>
#include <helper_cuda.h>  // helper functions CUDA error checking and initialization
#endif
#include <memory>
#include <span>
#include <chrono>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <ranges>
#include <algorithm>
#include <execution>
#include <atomic>
#include <cassert>
#include <experimental/mdspan>
namespace stdex = std::experimental;

#include "BSM.hpp"
#include "greek.hpp"

std::vector<double> 
generate_paths(const double s0, const double sigma_r, const double RiskFreeRate,
               const int horizon, const double dt, const int num_paths);

void calculate_pnl_paths_sequential(stdex::mdspan<const double, stdex::dextents<size_t,2>> paths, 
                         std::span<const double>Strikes, 
                         std::span<const double>Maturities, 
                         std::span<const double>Volatilities, 
                         const double RiskFreeRate,
                         std::span<double>pnl, 
                         const double dt);

void calculate_pnl_paths_parallel(stdex::mdspan<const double, stdex::dextents<size_t,2>> paths, 
                         std::span<const double>Strikes, 
                         std::span<const double>Maturities, 
                         std::span<const double>Volatilities, 
                         const double RiskFreeRate,
                         std::span<double>pnl, 
                         const double dt);


#endif // PNL_H
