#ifndef PNL_CUDA_H
#define PNL_CUDA_H

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

__global__ void calculate_pnl_kernel(const double* paths, 
                                    const double* Strikes, 
                                    const double* Maturities,
                                    const double* Volatilities,
                                    const double RiskFreeRate,
                                    double* pnl,
                                    const double dt, const int optN, const int num_paths, const int horizon);

void calculate_pnl_paths_parallel_cuda(const double* paths, 
                                    const double* Strikes, 
                                    const double* Maturities,
                                    const double* Volatilities,
                                    const double RiskFreeRate,
                                    double* pnl,
                                    const double dt, const int optN, const int num_paths, const int horizon);

void calculate_pnl_paths_parallel_omp(const double* paths, 
                                    const double* Strikes, 
                                    const double* Maturities,
                                    const double* Volatilities,
                                    const double RiskFreeRate,
                                    double* pnl,
                                    const double dt, const int optN, const int num_paths, const int horizon);

#endif // PNL_CUDA_H
