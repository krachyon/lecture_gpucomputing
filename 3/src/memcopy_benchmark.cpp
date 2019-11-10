#include "memcopy_benchmark.h"
#include <iostream>

inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
      fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
      assert(result == cudaSuccess);
    }
#endif
    return result;
}

Timers memcpy_benchmark(bool optUsePinnedMemory, size_t optMemorySize, size_t optMemCpyIterations)
{
    ChTimer memCpyH2DTimer, memCpyD2HTimer, memCpyD2DTimer;

    float* h_memoryA = nullptr;
    float* h_memoryB = nullptr;


    if (!optUsePinnedMemory) { // Pageable

        h_memoryA = (float*)malloc( optMemorySize );
        h_memoryB = (float*)malloc( optMemorySize );
    }
    else { // Pinned

        cudaMallocHost(&h_memoryA, optMemorySize);
        cudaMallocHost(&h_memoryB, optMemorySize);
    }

//
// Device Memory
//
    float* d_memoryA = nullptr;
    float* d_memoryB = nullptr;
    cudaMalloc(&d_memoryA, optMemorySize);
    cudaMalloc(&d_memoryB, optMemorySize);

    cudaError_t cudaError = cudaGetLastError();

    if (!h_memoryA || !h_memoryB || !d_memoryA || !d_memoryB || cudaError!=cudaSuccess) {
        std::cout << "**" << std::endl
                  << "*** Error - Memory allocation failed" << std::endl
                  << "***" << std::endl;

        if (cudaError!=cudaSuccess) {
            std::cout << "***" << std::endl
                      << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
                      << std::endl
                      << "***" << std::endl;


        }
        exit(-1);
    }

//
// Copy
//


// Host To Device
    memCpyH2DTimer.start();

    for (size_t i = 0; i<optMemCpyIterations; i++) {
        cudaMemcpy(d_memoryA, h_memoryA, optMemorySize, cudaMemcpyHostToDevice);
    }
    memCpyH2DTimer.stop();

// Device To Device
    memCpyD2DTimer.start();
    for (size_t i = 0; i<optMemCpyIterations; i++) {
        cudaMemcpy(d_memoryB, d_memoryA, optMemorySize, cudaMemcpyDeviceToDevice);

    }
    memCpyD2DTimer.stop();

// Device To Host
    memCpyD2HTimer.start();
    for (size_t i = 0; i<optMemCpyIterations; i++) {
        cudaMemcpy(h_memoryB, d_memoryB, optMemorySize, cudaMemcpyDeviceToHost);
    }
    memCpyD2HTimer.stop();

    cudaFree(d_memoryA);
    cudaFree(d_memoryB);
//
// Check for Errors
//

    cudaError = cudaGetLastError();
    if (cudaError!=cudaSuccess) {
        std::cout << "***" << std::endl
                  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
                  << std::endl
                  << "***" << std::endl;

        exit(-1);
    }

    return Timers{memCpyD2DTimer,memCpyD2HTimer,memCpyH2DTimer};
}