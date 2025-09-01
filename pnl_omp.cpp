#include <cstdio>
#ifdef _NVHPC_STDPAR_GPU
// These headers are not strictly necessary but used for performance
// tuning.
#include <cuda_runtime.h>
#include <helper_cuda.h>  // helper functions CUDA error checking and initialization
#endif
#include <omp.h>
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



void calculate_pnl_paths_parallel_omp(const double* paths, 
                                    const double* Strikes, 
                                    const double* Maturities,
                                    const double* Volatilities,
                                    const double RiskFreeRate,
                                    double* pnl,
                                    const double dt, const int optN, const int num_paths=1000, const int horizon=180)
{
    // int horizon = 180;   // 180 day (6 month) simulation horizon
    // int num_paths = 1000; // 1000 simulation paths
    // int optN = int(sizeof(pnl) / sizeof(double));
    int opts = optN * num_paths;

    double* pnl_private = (double*)omp_target_alloc(num_paths * optN * sizeof(double), omp_get_default_device());

    #pragma omp target teams distribute parallel for is_device_ptr(pnl_private)
    for (int i = 0; i < num_paths * optN; ++i)
        pnl_private[i] = 0.0;

    #pragma omp target teams distribute parallel for num_teams((opts+255)/256) thread_limit(256) \
        map(to: paths[0:num_paths*horizon], Strikes[0:optN], Maturities[0:optN], Volatilities[0:optN], RiskFreeRate, dt, optN) is_device_ptr(pnl_private) // map(tofrom: pnl[0:optN]) 
    for (int idx = 0; idx < opts; ++idx) {
        int path = idx / optN;
        int opt = idx % optN;
        double value = 0.0;
        // #pragma reduction(+:value)
        for (int i = 1; i < horizon; ++i) {  // evt. optimer med smem her?
            double gamma = 0.0, theta = 0.0;
            double s = paths[path*horizon + i]; // problem med strided access...
            double s_prev = paths[path*horizon + (i-1)]; // ...sammen med denne her?
            double ds2 = s - s_prev;
            ds2 *= ds2;
            
            double time_to_maturity = Maturities[opt] - (std::max(dt * (i-1),0.0));
            gamma = black_scholes(s_prev, Strikes[opt], RiskFreeRate, time_to_maturity, Volatilities[opt], CALL, GAMMA);
            theta = black_scholes(s_prev, Strikes[opt], RiskFreeRate, time_to_maturity, Volatilities[opt], CALL, THETA);
            // BlackScholesBody(gamma,
            //             s_prev, 
            //             Strikes[opt], 
            //             time_to_maturity,
            //             RiskFreeRate,
            //             Volatilities[opt],
            //             CALL, 
            //             GAMMA);
            // BlackScholesBody(theta,
            //             s_prev, 
            //             Strikes[opt], 
            //             time_to_maturity,
            //             RiskFreeRate,
            //             Volatilities[opt],
            //             CALL, 
            //             THETA);
            // pnl[opt] += 0.5 * gamma * ds2 + (theta*dt);
            // atomicAdd(&pnl[opt], 0.5 * gamma * ds2 + (theta*dt)); // denne her summation er nok ikke så hurtig
            // }

            value += 0.5 * gamma * ds2 + (theta*dt);
        }
        pnl_private[path*optN + opt] += value;
    // #pragma omp atomic update
    // pnl[opt] += value;
    }
    #pragma omp target teams distribute parallel for map(from: pnl[0:optN]) is_device_ptr(pnl_private)
    for (int opt = 0; opt < optN; ++opt) {
        double sum = 0.0;
        for (int path = 0; path < num_paths; ++path) {
            sum += pnl_private[path * optN + opt];
        }
        pnl[opt] = sum;
    }

    omp_target_free(pnl_private, omp_get_default_device());
}
