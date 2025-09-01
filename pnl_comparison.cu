// #include <nvToolsExt.h>
#define NVTX3_ENABLE 1
#define NVTX1_DISABLE 1
#include <nvtx3/nvtx3.hpp>

// #define NVTX_ENABLE 1

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

#include "pnl_cuda.cuh"
#include "pnl.hpp"
#include <cuda_profiler_api.h>


#ifdef MAIN_FILE
int main(int argc, char **argv) {
    using globalnvtx = nvtx3::domain::global;
    
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

    int horizon = 180;   // 180 day (6 month) simulation horizon
    int num_paths = 1000; // 1000 simulation paths
    int graph_mode = 0; // 0 for verbose, 1 for graph mode
    if (argc > 1) {
        num_paths = atoi(argv[1]);
    }
    if (argc > 2) {
        horizon = atoi(argv[2]);
    }
    if (argc > 3) {
        graph_mode = atof(argv[3]);
    }
    std::vector<double> Strikes     (OPT_N);
    std::vector<double> Maturities  (OPT_N);
    std::vector<double> Volatilities(OPT_N);

    // Used for reference implementation
    std::vector<double> pnl_vec        (OPT_N, 0.0);
    std::span pnl{pnl_vec};  // Non-owning view into pnl vector

    // Used for parallel paths implementation
    std::vector<double> pnl2_vec       (OPT_N, 0.0);
    std::span pnl2{pnl2_vec}; // Non-owning view into pnl2 vector

    srand(5347);

    for (int t = 0; t < n_t_steps; ++t) 
    {
    for (int m = 0; m < n_money_steps; ++m) 
    {
        i = t * n_money_steps + m;
        Strikes[i] = s0 * (1 + money_start + m * money_step);
        Volatilities[i] = sigma_i;
        Maturities[i] = t_start + t * t_step;
    }
    }

    // generate paths
    auto path_vec = generate_paths(s0, sigma_r, RiskFreeRate, horizon, dt, num_paths);
    // Create a 2D view into the paths array [num_paths,horizon]
    auto paths = stdex::mdspan{path_vec.data(),num_paths,horizon};

// #ifdef _NVHPC_STDPAR_GPU
//     // Optional. Prefetches data to GPU memory to avoid expensive page faults 
//     // in the first call.
//     const int OPT_SZ = OPT_N * sizeof(double);
//     checkCudaErrors(cudaMemPrefetchAsync(&Strikes[0],      OPT_SZ,0,0));
//     checkCudaErrors(cudaMemPrefetchAsync(&Maturities[0],   OPT_SZ,0,0));
//     checkCudaErrors(cudaMemPrefetchAsync(&Volatilities[0], OPT_SZ,0,0));
//     checkCudaErrors(cudaMemPrefetchAsync(&pnl_vec[0],      OPT_SZ,0,0));
//     checkCudaErrors(cudaMemPrefetchAsync(&pnl2_vec[0],     OPT_SZ,0,0));
//     checkCudaErrors(cudaDeviceSynchronize()); // Synchronize before calculation to ensure proper timing.
// #endif

    auto t1 = std::chrono::high_resolution_clock::now();

    ///////////////////////////////////////////////////////////////////////////
    // The original implementation of P&L calculation parallelizes only over
    // options within the calculate_pnl function. This limits the amount of
    // available parallelism. The iteration along paths is done sequentially on
    // the CPU, even when building for the GPU.
    ///////////////////////////////////////////////////////////////////////////
    // uint64_t seqR = nvtxRangeStartA("sequential");
    {
    nvtx3::scoped_range_in<globalnvtx> seqR{"sequential" };
    // nvtxRangePushA("sequential");
    // cudaProfilerStart();
    calculate_pnl_paths_sequential(paths, Strikes, Maturities, Volatilities, RiskFreeRate, pnl, dt);
    // cudaProfilerStop();
    // nvtxRangePop();
    }
    // nvtxRangeEnd(seqR);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_ms = (t2-t1);
    double time = time_ms.count();

    // pnl holds an accumulation of P&L for all paths, need to divide by num_paths
    std::transform(pnl.begin(),pnl.end(),pnl.begin(),[=](double p){ return p/num_paths; });
    // Find the maximum PNL value
    auto max_pnl = std::max_element(pnl.begin(),pnl.end());

    long numOpts = (long)OPT_N * (long)num_paths;
    if (graph_mode == 0) {
    printf(
        "Profit & Loss, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u "
        "options, Paths = %d\n",
        (((double)(numOpts) * 1.0E-9) / (time * 1.0E-3)), time * 1e-3,
        (numOpts), num_paths);
    }
    if (graph_mode == 1) {
        printf("%.5f ", time * 1e-3);
    }
// #ifdef _NVHPC_STDPAR_GPU
//     // Optional - prefetch path_vec to GPU as a performance optimization
//     checkCudaErrors(cudaMemPrefetchAsync(&path_vec[0], horizon*num_paths*sizeof(double),0,0));
//     checkCudaErrors(cudaDeviceSynchronize()); // Synchronize before calculation to ensure proper timing.
// #endif

    auto t1paths = std::chrono::high_resolution_clock::now();

    ///////////////////////////////////////////////////////////////////////////
    // The optimized implementation of P&L calculation parallelizes over
    // options but also over paths. This increases parallelism and also 
    // reduces the need to synchronize between the CPU and GPU when building
    // for GPU execution.
    ///////////////////////////////////////////////////////////////////////////
    // uint64_t parR = nvtxRangeStartA("parallel");
    {
    nvtx3::scoped_range_in<globalnvtx> parR{"parallel" };
    // nvtxRangePushA("parallel");
    cudaProfilerStart();
    calculate_pnl_paths_parallel(paths, Strikes, Maturities, Volatilities, RiskFreeRate, pnl2, dt);
    cudaProfilerStop();
    // nvtxRangePop();
    }
    // nvtxRangeEnd(parR);
    // PNL holds an accumulation of P&L for all paths, to calculate the average we divide by num_paths 
    // Since pnl has already been used on the device, we will run in parallel to avoid data migration

    auto t2paths = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> timepaths_ms = (t2paths-t1paths);
    double timepaths = timepaths_ms.count();
    std::transform(std::execution::par_unseq, pnl2.begin(),pnl2.end(),pnl2.begin(),[=](double summed_pnl){ return summed_pnl/num_paths; });

    if (graph_mode == 0) {
    printf(
        "\nProfit & Loss, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u "
        "options, Paths = %d\n",
        (((double)(numOpts) * 1.0E-9) / (timepaths * 1.0E-3)), timepaths * 1e-3,
        (numOpts), num_paths);
    }
    if (graph_mode == 1) {
        printf("%.5f ", timepaths * 1e-3);
    }
    // Find the maximum PNL value, returns a pointer to the value in the array
    auto max_pnl2 = std::max_element(pnl2.begin(),pnl2.end());
    // This returns the index of the maximum value in the array
    int max_idx = std::distance(pnl2.begin(),max_pnl2);
    if (graph_mode == 0) {
    printf("Max PNL is at index %d and has a value of %lf\n\n", max_idx, *max_pnl2);
    printf("Speed-up from parallelizing over paths: %lfX\n", time/timepaths);
    }
    // Calculate max absolute difference and L1 distance
    // between reference and optimized results
    double sum_diff = 0;
    double sum_ref = 0;
    double max_diff = 0;

    for (i = 0; i < OPT_N; i++) {
        double ref = pnl[i];
        double diff = fabs(pnl[i] - pnl2[i]);

        if (diff > max_diff) {
        max_diff = diff;
        }

        sum_diff += diff;
        sum_ref += fabs(ref);
    }

    // for (int p = 0; p < 16; ++p) {
    //     printf("%f\n",pnl2[p]);
    // }

    double L1norm = sum_diff / sum_ref;
    if (graph_mode == 0) {
    printf("L1 norm: %E\n", L1norm);
    printf("Max absolute error: %E\n\n", max_diff);
    }
    assert(L1norm < max_diff);

    // printf("Test passed\n");
    // exit(EXIT_SUCCESS);

    ///////////////////////////////////////////////////////////////////////////
    // CUDA implementation
    ///////////////////////////////////////////////////////////////////////////
    double *path_vec_h;
    cudaMallocHost((void**)&path_vec_h, horizon * num_paths * sizeof(double));
    double *path_vec_d;
    cudaMalloc(&path_vec_d, horizon * num_paths * sizeof(double)); 
    // for (int p = 0; p < num_paths; ++p) {
    //     std::random_device rd{};
    //     std::mt19937 gen{rd()};
    //     gen.seed(100+p);
    //     std::normal_distribution<double> dist{0.0,1.0};

    //     path_vec_h[0 * num_paths + p] = s0;
    //     for (int k = 1; k < horizon; ++k) {
    //         double w = dist(gen);
    //         path_vec_h[k * num_paths + p] = path_vec_h[(k-1) * num_paths + p] * exp((RiskFreeRate - 0.5 * sigma_r * sigma_r) * dt + sigma_r * sqrt(dt) * w);
    //     }
    // }
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

    checkCudaErrors(cudaDeviceSynchronize());
    
    auto t1pathscuda = std::chrono::high_resolution_clock::now();
    // uint64_t cudaR = nvtxRangeStartA("cuda");
    {
    nvtx3::scoped_range_in<globalnvtx> cudaR{"cuda"};
    // nvtxRangePushA("cuda");
    cudaProfilerStart();
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
    cudaProfilerStop();
    // nvtxRangePop();
    }
    // nvtxRangeEnd(cudaR);

    // Copy results back to host
    // divide by number of paths to get average PnL per path
    auto t2pathscuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> timepaths_mscuda = (t2pathscuda-t1pathscuda);
    double timepathscuda = timepaths_mscuda.count();
    cudaMemcpy(pnl_vec_h, pnl_vec_d, OPT_N * num_paths * sizeof(double), cudaMemcpyDeviceToHost);
    for (int p = 0; p < num_paths; ++p) {
        for (int k = 0; k < OPT_N; ++k) {
            pnl_vec_h[p * OPT_N + k] /= num_paths;
        }
    }
    if (graph_mode == 0) {
    printf(
        "\nProfit & Loss, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u "
        "options, Paths = %d\n",
        (((double)(numOpts) * 1.0E-9) / (timepathscuda * 1.0E-3)), timepathscuda * 1e-3,
        (numOpts), num_paths);
    }
    if (graph_mode == 1) {
        printf("%.5f ", timepathscuda * 1e-3);
    }
    // Print sum of pnl for all options to verify calculations are made
    // double total_pnl = 0.0;
    // for (int p = 0; p < 4; ++p) {
    //     for (int k = 0; k < 4; ++k) {
    //         printf("%f\n",pnl_vec_h[p * OPT_N + k]);
    //     }
    // }
    // for (int p = 0; p < 16; ++p) {
    //     printf("%f\n", pnl_vec_h[p]);
    //     // if (pnl_vec_h[p] == 0) {
    //     //     printf("\n\n%d\n\n", p);
    //     //     break;
    //     // }
    // }
    // printf("Total PnL for all options: %f\n", total_pnl);

    // Compare the results with the previous implementation
    double max_pnl_cuda = 0.0;
    int max_idx_cuda = 0;
    for (int p = 0; p < num_paths; ++p) {
        for (int k = 0; k < OPT_N; ++k) {
            if (pnl_vec_h[p * OPT_N + k] > max_pnl_cuda) {
                max_pnl_cuda = pnl_vec_h[p * OPT_N + k];
                max_idx_cuda = p * OPT_N + k;
            }
        }
    }
    if (graph_mode == 0) {
    printf("Max PNL from CUDA implementation is at index %d and has a value of %lf\n", max_idx_cuda, max_pnl_cuda);
    }
    // compare absolute difference between the two implementations
    sum_diff = 0;
    sum_ref = 0;
    max_diff = 0;

    for (i = 0; i < OPT_N; i++) {
        double ref = pnl[i];
        double diff = fabs(pnl[i] - pnl_vec_h[i]);

        if (diff > max_diff) {
        max_diff = diff;
        }

        sum_diff += diff;
        sum_ref += fabs(ref);
    }
    L1norm = sum_diff / sum_ref;
    if (graph_mode == 0) {
    printf("L1 norm: %E\n", L1norm);
    printf("Max absolute error: %E\n\n", max_diff);
    }
    assert(L1norm < max_diff);
    // printf("Test passed\n");

    ///////////////////////////////////////////////////////////////////////////
    // OMP implementation
    ///////////////////////////////////////////////////////////////////////////
    double* path_vec_h_omp;
    cudaMallocHost((void**)&path_vec_h_omp, horizon * num_paths * sizeof(double));
    for (int p = 0; p < num_paths; ++p) {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        gen.seed(100+p);
        std::normal_distribution<double> dist{0.0,1.0};

        path_vec_h_omp[p*horizon] = s0;
        for (int k = 1; k < horizon; ++k) {
            double w = dist(gen);
            path_vec_h_omp[p*horizon + k] = path_vec_h_omp[p*horizon + (k-1)] * exp((RiskFreeRate - 0.5 * sigma_r * sigma_r) * dt + sigma_r * sqrt(dt) * w);
        }
    }

    double* pnl_vec_h_omp;
    cudaMallocHost((void**)&pnl_vec_h_omp, OPT_N * num_paths * sizeof(double));
    for (int p = 0; p < num_paths; ++p) {
        for (int k = 0; k < OPT_N; ++k) {
            pnl_vec_h_omp[p * OPT_N + k] = 0.0; // Initialize PnL vector to zero
        }
    }

    double timepathsomp = 0.0;
    // #pragma omp target data map(to: paths[0:num_paths], Strikes_h[0:OPT_N], Maturities_h[0:OPT_N], Volatilities_h[0:OPT_N], RiskFreeRate, dt, OPT_N) map(tofrom: pnl_vec_h_omp[0:OPT_N], timepathsomp) 
    // {
        // eval data to make sure the data allocation is not timed
        // auto eval = paths(0,0) + Strikes_h[0] + Maturities_h[0] + Volatilities_h[0] + RiskFreeRate + dt + OPT_N + pnl_vec_h_omp[0] + timepathsomp;

        auto t1pathsomp = std::chrono::high_resolution_clock::now();
        // uint64_t ompR = nvtxRangeStartA("omp");
        {
        nvtx3::scoped_range_in<globalnvtx> ompR{"omp" };
        // nvtxRangePushA("omp");
        // cudaProfilerStart();
        calculate_pnl_paths_parallel_omp(path_vec_h_omp, 
                                    Strikes_h, 
                                    Maturities_h, 
                                    Volatilities_h, 
                                    RiskFreeRate, 
                                    pnl_vec_h_omp, 
                                    dt,
                                    OPT_N,
                                    num_paths,
                                    horizon);
        // cudaProfilerStop();
        // nvtxRangePop();
        // nvtxRangeEnd(ompR);
        }
        // Copy results back to host
        // divide by number of paths to get average PnL per path
        auto t2pathsomp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> timepaths_msomp = (t2pathsomp-t1pathsomp);
        timepathsomp = timepaths_msomp.count();
        for (int p = 0; p < num_paths; ++p) {
            for (int k = 0; k < OPT_N; ++k) {
                pnl_vec_h_omp[p * OPT_N + k] /= num_paths;
            }
        }
    // }
    if (graph_mode == 0) {
    printf(
        "\nProfit & Loss, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u "
        "options, Paths = %d\n",
        (((double)(numOpts) * 1.0E-9) / (timepathsomp * 1.0E-3)), timepathsomp * 1e-3,
        (numOpts), num_paths);
    }
    if (graph_mode == 1) {
        printf("%.5f\n", timepathsomp * 1e-3);
    }
    // Compare the results with the previous implementation
    double max_pnl_omp = 0.0;
    int max_idx_omp = 0;
    for (int p = 0; p < num_paths; ++p) {
        for (int k = 0; k < OPT_N; ++k) {
            if (pnl_vec_h_omp[p * OPT_N + k] > max_pnl_omp) {
                max_pnl_omp = pnl_vec_h_omp[p * OPT_N + k];
                max_idx_omp = p * OPT_N + k;
            }
        }
    }
    if (graph_mode == 0) {
    printf("Max PNL from OMP implementation is at index %d and has a value of %lf\n", max_idx_omp, max_pnl_omp);
    }
    // compare absolute difference between the two implementations
    sum_diff = 0;
    sum_ref = 0;
    max_diff = 0;

    for (i = 0; i < OPT_N; i++) {
        double ref = pnl[i];
        double diff = fabs(pnl[i] - pnl_vec_h_omp[i]);

        if (diff > max_diff) {
        max_diff = diff;
        }

        sum_diff += diff;
        sum_ref += fabs(ref);
    }
    L1norm = sum_diff / sum_ref;
    if (graph_mode == 0) {
    printf("L1 norm: %E\n", L1norm);
    printf("Max absolute error: %E\n\n", max_diff);
    }
    assert(L1norm < max_diff);
    // printf("Test passed\n");


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