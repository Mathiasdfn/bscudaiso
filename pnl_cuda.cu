#include <cstdio>
// #ifdef _NVHPC_STDPAR_GPU
// // These headers are not strictly necessary but used for performance
// // tuning.
// #include <cuda_runtime.h>
// #include <helper_cuda.h>  // helper functions CUDA error checking and initialization
// #endif
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



__global__ void calculate_pnl_kernel(const double* paths, 
                                    const double* Strikes, 
                                    const double* Maturities,
                                    const double* Volatilities,
                                    const double RiskFreeRate,
                                    double* pnl,
                                    const double dt, const int optN, const int num_paths=1000, const int horizon=180)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int horizon = 180;   // 180 day (6 month) simulation horizon
    // int num_paths = 1000; // 1000 simulation paths
    // int optN = int(sizeof(pnl) / sizeof(double));
    int opts = optN * num_paths;

    if (idx < opts) {
        // for (int idx = 0; idx < opts; ++idx) {
            int path = idx / optN;
            int opt = idx % optN;
            // int opt = blockIdx.x; // each block handles one option
            // int path = threadIdx.x + blockDim.x * blockIdx.y; // each thread handles one path for the option
            double value = 0.0;
            for (int i = 1; i < horizon; ++i) {  // evt. optimer med smem her?
                double gamma = 0.0, theta = 0.0;
                double s = paths[path*horizon + i]; // problem med strided access...
                double s_prev = paths[path*horizon + (i-1)]; // ...sammen med denne her?
                double ds2 = s - s_prev;
                ds2 *= ds2;
                
                double time_to_maturity = Maturities[opt] - (std::max(dt * (i-1),0.0));
                BlackScholesBody(gamma,
                            s_prev, 
                            Strikes[opt], 
                            time_to_maturity,
                            RiskFreeRate,
                            Volatilities[opt],
                            CALL, 
                            GAMMA);
                BlackScholesBody(theta,
                           s_prev, 
                           Strikes[opt], 
                           time_to_maturity,
                           RiskFreeRate,
                           Volatilities[opt],
                           CALL, 
                           THETA);
                // pnl[opt] += 0.5 * gamma * ds2 + (theta*dt);
                // atomicAdd(&pnl[opt], 0.5 * gamma * ds2 + (theta*dt)); // denne her summation er nok ikke så hurtig
                // }
                value += 0.5 * gamma * ds2 + (theta*dt);
            }
        atomicAdd(&pnl[opt], value);
        // }
    }
}

__global__ void calculate_pnl_kernel_smem(const double* paths, 
                                    const double* Strikes, 
                                    const double* Maturities,
                                    const double* Volatilities,
                                    const double RiskFreeRate,
                                    double* pnl,
                                    const double dt, const int optN, const int num_paths=1000, const int horizon=180)
{
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int horizon = 180;   // 180 day (6 month) simulation horizon
    // int num_paths = 1000; // 1000 simulation paths
    // int optN = int(sizeof(pnl) / sizeof(double));
    // int opts = optN * num_paths;
    extern __shared__ double smem[];  // equals (blockDim.x/32), which is total nr. of warps
    int opt = blockIdx.x; // each block handles one option
    int tid = threadIdx.x; // thread index within the block
    int path = tid + blockDim.x * blockIdx.y; // each thread handles one path for the option
    double value = 0.0;

    if (opt < optN && path < num_paths) {
        for (int i = 1; i < horizon; ++i) {
            double gamma = 0.0, theta = 0.0;
                double s = paths[path*horizon + i]; // problem med strided access...
                double s_prev = paths[path*horizon + (i-1)]; // ...sammen med denne her?
                double ds2 = s - s_prev;
                ds2 *= ds2;
                
                double time_to_maturity = Maturities[opt] - (std::max(dt * (i-1),0.0));
                BlackScholesBody(gamma,
                            s_prev, 
                            Strikes[opt], 
                            time_to_maturity,
                            RiskFreeRate,
                            Volatilities[opt],
                            CALL, 
                            GAMMA);
                BlackScholesBody(theta,
                           s_prev, 
                           Strikes[opt], 
                           time_to_maturity,
                           RiskFreeRate,
                           Volatilities[opt],
                           CALL, 
                           THETA);
                value += 0.5 * gamma * ds2 + (theta*dt);
        }
        smem[tid] = value; // store the value in shared memory
    } else {
        smem[tid] = 0.0; // if the thread is out of bounds, store 0
    }
    __syncthreads();

    // Perform a reduction in shared memory to sum the values for this option
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    // The first thread in the block writes the result to the global memory
    if (tid == 0) atomicAdd(&pnl[opt], smem[0]);
}

__global__ void calculate_pnl_kernel_noatom(const double* paths, 
                                    const double* Strikes, 
                                    const double* Maturities,
                                    const double* Volatilities,
                                    const double RiskFreeRate,
                                    double* pnl_private,
                                    const double dt, const int optN, const int num_paths=1000, const int horizon=180)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int horizon = 180;   // 180 day (6 month) simulation horizon
    // int num_paths = 1000; // 1000 simulation paths
    // int optN = int(sizeof(pnl) / sizeof(double));
    int opts = optN * num_paths;

    if (idx < opts) {
        // for (int idx = 0; idx < opts; ++idx) {
            int path = idx / optN;
            int opt = idx % optN;
            double value = 0.0;
            for (int i = 1; i < horizon; ++i) {  // evt. optimer med smem her?
                double gamma = 0.0, theta = 0.0;
                double s = paths[path*horizon + i]; // problem med strided access...
                double s_prev = paths[path*horizon + (i-1)]; // ...sammen med denne her?
                double ds2 = s - s_prev;
                ds2 *= ds2;
                
                double time_to_maturity = Maturities[opt] - (std::max(dt * (i-1),0.0));
                BlackScholesBody(gamma,
                            s_prev, 
                            Strikes[opt], 
                            time_to_maturity,
                            RiskFreeRate,
                            Volatilities[opt],
                            CALL, 
                            GAMMA);
                BlackScholesBody(theta,
                           s_prev, 
                           Strikes[opt], 
                           time_to_maturity,
                           RiskFreeRate,
                           Volatilities[opt],
                           CALL, 
                           THETA);
                // pnl[opt] += 0.5 * gamma * ds2 + (theta*dt);
                // atomicAdd(&pnl[opt], 0.5 * gamma * ds2 + (theta*dt)); // denne her summation er nok ikke så hurtig
                // }
                value += 0.5 * gamma * ds2 + (theta*dt);
            }
        pnl_private[path * optN + opt] = value; // store the value in private memory
        // }
    }
}

__global__ void reduce_pnl_kernel(
    const double* pnl_private, double* pnl, int num_paths, int optN)
{
    int opt = blockIdx.x * blockDim.x + threadIdx.x;
    if (opt < optN) {
        double sum = 0.0;
        for (int path = 0; path < num_paths; ++path) {
            sum += pnl_private[path * optN + opt];
        }
        pnl[opt] = sum;
    }
}

void calculate_pnl_paths_parallel_cuda(const double* paths, 
                                    const double* Strikes, 
                                    const double* Maturities,
                                    const double* Volatilities,
                                    const double RiskFreeRate,
                                    double* pnl,
                                    const double dt, const int optN, const int num_paths=1000, const int horizon=180)
{
    // int horizon = 180;   // 180 day (6 month) simulation horizon
    // int num_paths = 1000; // 1000 simulation paths
    // int optN = sizeof(pnl) / sizeof(double);
    // auto opts = optN * num_paths;
    int opts = optN * num_paths;
    // printf("optN = %d, num_paths = %d, opts = %d\n", optN, num_paths, opts);

    // Launch the kernel to calculate PnL in parallel
    // Each thread will handle one option for one path
    // The grid size is determined by the number of options and paths

    int gridSize = (opts + 255) / 256; // 256 threads per block
    int blockSize = 256; // 256 threads per block
    calculate_pnl_kernel<<<gridSize, blockSize>>>(paths, 
                                                  Strikes, 
                                                  Maturities, 
                                                  Volatilities, 
                                                  RiskFreeRate, 
                                                  pnl, 
                                                  dt,
                                                  optN,
                                                  num_paths,
                                                  horizon);
    
    // smem kernel (umiddelbart langsommere)
    // int blockSize = 256; // number of threads per block
    // dim3 block(blockSize); // each block handles one option
    // dim3 grid(optN, (num_paths + blockSize - 1) / blockSize); // grid of options, each block handles one option
    // size_t smem_size = blockSize * sizeof(double); // size of shared memory per block
    // calculate_pnl_kernel_smem<<<grid, block, smem_size>>>(paths, 
    //                                            Strikes, 
    //                                            Maturities, 
    //                                            Volatilities, 
    //                                            RiskFreeRate, 
    //                                            pnl, 
    //                                            dt,
    //                                            optN,
    //                                            num_paths,
    //                                            horizon);

    
    // double kernel uden atomics
    // double* pnl_private_d;
    // cudaMalloc(&pnl_private_d, num_paths * optN * sizeof(double));

    // // Launch main kernel
    // int total_threads = num_paths * optN;
    // int blockSize = 256;
    // int gridSize = (total_threads + blockSize - 1) / blockSize;
    // calculate_pnl_kernel_noatom<<<gridSize, blockSize>>>(
    //     paths, 
    //     Strikes, 
    //     Maturities, 
    //     Volatilities, 
    //     RiskFreeRate, 
    //     pnl_private_d, 
    //     dt,
    //     optN,
    //     num_paths,
    //     horizon);
    // // Launch reduction kernel
    // int reduceBlockSize = 256;
    // int reduceGridSize = (optN + reduceBlockSize - 1) / reduceBlockSize;
    // reduce_pnl_kernel<<<reduceGridSize, reduceBlockSize>>>(
    //     pnl_private_d, pnl, num_paths, optN);


    // calculate_pnl_kernel<<<(num_paths + 255) / 256, 256>>>(paths, Strikes, Maturities, Volatilities, RiskFreeRate, pnl, dt);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize to ensure all threads have completed
    cudaDeviceSynchronize();
}
#ifdef MAIN_FILE
int main(int argc, char ** argv) {
    // Start logs
    printf("[%s] - Starting...\n", argv[0]);

    int i;
    ////////////////////////////////////////////////////////////////////////////////
    // Data configuration
    //
    // Equally spaced grid along maturity and moneyness
    ////////////////////////////////////////////////////////////////////////////////
    const int days_in_year = 365;                   // 365 days in year
    const int num_years = 10;                       // 10 years
    const int n_t_steps = days_in_year * num_years; // number of time steps
    const double t_start = 0.5;                     // starting maturity (1/2 year) 
    const double t_step = 1./(n_t_steps);           // daily

    const int n_money_steps = 60;    // moneyness steps
    const double money_start = -0.4; // starting moneyness 40% below at the money
    const double money_step = 0.01;  // step size of 1%

    const int OPT_N = n_t_steps * n_money_steps;

    ////////////////////////////////////////////////////////////////////////////////
    // Simulation parameters
    //
    // Simulate each option in the grid for a 180 day horizon
    ////////////////////////////////////////////////////////////////////////////////
    const double s0 = 100.0,              // Initial Spot Price
                sigma_r = 0.5,           // Realized Spot Volatility Used for Simulation
                sigma_i = 0.3,           // Implied Spot Volatility Used for Pricing
                dt = 1.0 / days_in_year, // Timestep in years (1 day)
                RiskFreeRate = 0.0;               // Risk-free Rate

    const int horizon = 180;   // 180 day (6 month) simulation horizon
    const int num_paths = 1000; // 1000 simulation paths
    // if (argc >= 2) num_paths = atoi(argv[1]);
    double *path_vec_h;
    cudaMallocHost((void**)&path_vec_h, horizon * num_paths * sizeof(double));
    double *path_vec_d;
    cudaMalloc(&path_vec_d, horizon * num_paths * sizeof(double)); 
    for (int p = 0; p < num_paths; ++p) {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(100+p);
        std::normal_distribution<double> dist{0.0,1.0};

        path_vec_h[p*horizon] = s0;
        for (int k = 1; k < horizon; ++k) {
            double w = dist(gen);
            path_vec_h[p*horizon + k] = path_vec_h[p*horizon + (k-1)] * exp((RiskFreeRate - 0.5 * sigma_r * sigma_r) * dt + sigma_r * sqrt(dt) * w);
        }
    }
    cudaMemcpy(path_vec_d, path_vec_h, horizon * num_paths * sizeof(double), cudaMemcpyHostToDevice);

    double* Strikes_h;
    double* Strikes_d;
    double* Maturities_h;
    double* Maturities_d;
    double* Volatilities_h;
    double* Volatilities_d;
    cudaMallocHost((void**)&Strikes_h, OPT_N * sizeof(double));
    cudaMalloc(&Strikes_d, OPT_N * sizeof(double));
    cudaMallocHost((void**)&Maturities_h, OPT_N * sizeof(double));
    cudaMalloc(&Maturities_d, OPT_N * sizeof(double));
    cudaMallocHost((void**)&Volatilities_h, OPT_N * sizeof(double));
    cudaMalloc(&Volatilities_d, OPT_N * sizeof(double));
    double* pnl_vec_h;
    double* pnl_vec_d;
    cudaMallocHost((void**)&pnl_vec_h, OPT_N * num_paths * sizeof(double));
    for (int p = 0; p < num_paths; ++p) {
        for (int k = 0; k < OPT_N; ++k) {
            pnl_vec_h[p * OPT_N + k] = 0.0; // Initialize PnL vector to zero
        }
    }
    cudaMalloc(&pnl_vec_d, OPT_N * num_paths * sizeof(double));
    cudaMemcpy(pnl_vec_d, pnl_vec_h, OPT_N * num_paths * sizeof(double), cudaMemcpyHostToDevice);
    srand(5347);
    for (int t = 0; t < n_t_steps; ++t) {
        for (int m = 0; m < n_money_steps; ++m) {
            i = t * n_money_steps + m;
            Strikes_h[i] = s0 * (1.0 + money_start + money_step * m);
            Maturities_h[i] = t_start + t_step * t;
            Volatilities_h[i] = sigma_i;
        }
    }
    cudaMemcpy(Strikes_d, Strikes_h, OPT_N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Maturities_d, Maturities_h, OPT_N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Volatilities_d, Volatilities_h, OPT_N * sizeof(double), cudaMemcpyHostToDevice);
    
    // for (int p = 0; p < 4; ++p) {
    //     for (int k = 0; k < 4; ++k) {
    //         printf("%f ", path_vec_h[p * horizon + k]);
    //     }
    //     printf("\n");
    // }
    for (int p = 0; p < num_paths; ++p) {
        for (int k = 0; k < horizon; ++k) {
            if (path_vec_h[p * horizon + k] == 0){
                printf("Path %d, time step %d: 0.0\n", p, k);
                break;
            }
        }
    }

    calculate_pnl_paths_parallel_cuda(path_vec_d, 
                                Strikes_d, 
                                Maturities_d, 
                                Volatilities_d, 
                                RiskFreeRate, 
                                pnl_vec_d, 
                                dt,
                                OPT_N,
                                num_paths,
                                horizon);
    // Copy results back to host
    cudaMemcpy(pnl_vec_h, pnl_vec_d, OPT_N * num_paths * sizeof(double), cudaMemcpyDeviceToHost);
    // divide by number of paths to get average PnL per path
    for (int p = 0; p < num_paths; ++p) {
        for (int k = 0; k < OPT_N; ++k) {
            pnl_vec_h[p * OPT_N + k] /= num_paths;
        }
    }
    // Print sum of pnl for all options to verify calculations are made
    // double total_pnl = 0.0;
    // for (int p = 0; p < 4; ++p) {
    //     for (int k = 0; k < 4; ++k) {
    //         printf("%f\n",pnl_vec_h[p * OPT_N + k]);
    //     }
    // }
    for (int p = 0; p < 16; ++p) {
        printf("%f\n", pnl_vec_h[p]);
        // if (pnl_vec_h[p] == 0) {
        //     printf("\n\n%d\n\n", p);
        //     break;
        // }
    }
    // printf("Total PnL for all options: %f\n", total_pnl);

    // Free memory
    cudaFreeHost(path_vec_h);
    cudaFree(path_vec_d);
    cudaFreeHost(Strikes_h);
    cudaFree(Strikes_d);
    cudaFreeHost(Maturities_h);
    cudaFree(Maturities_d);
    cudaFreeHost(Volatilities_h);
    cudaFree(Volatilities_d);
    cudaFreeHost(pnl_vec_h);
    cudaFree(pnl_vec_d);

}
#endif