#include "memcopy_benchmark.h"
#include <iostream>

Timers memcpy_benchmark(bool optUsePinnedMemory, size_t optMemorySize, size_t optMemCpyIterations)
{
    ChTimer memCpyH2DTimer, memCpyD2HTimer, memCpyD2DTimer;

    int* h_memoryA = nullptr;
    int* h_memoryB = nullptr;


    if (!optUsePinnedMemory) { // Pageable
        std::cout << "***" << " Using pageable memory" << std::endl;
        h_memoryA = static_cast <int*> ( malloc(static_cast <size_t> ( optMemorySize )));
        h_memoryB = static_cast <int*> ( malloc(static_cast <size_t> ( optMemorySize )));
    }
    else { // Pinned
        std::cout << "***" << " Using pinned memory" << std::endl;
        cudaMallocHost(&h_memoryA, optMemorySize);
        cudaMallocHost(&h_memoryB, optMemorySize);
    }

//
// Device Memory
//
    int* d_memoryA = nullptr;
    int* d_memoryB = nullptr;
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
        cudaMemcpy(h_memoryA, d_memoryA, optMemorySize, cudaMemcpyHostToDevice);
    }
    memCpyH2DTimer.stop();

// Device To Device
    memCpyD2DTimer.start();
    for (size_t i = 0; i<optMemCpyIterations; i++) {
        cudaMemcpy(d_memoryA, d_memoryB, optMemorySize, cudaMemcpyDeviceToDevice);

    }
    memCpyD2DTimer.stop();

// Device To Host
    memCpyD2HTimer.start();
    for (size_t i = 0; i<optMemCpyIterations; i++) {
        cudaMemcpy(d_memoryB, h_memoryB, optMemorySize, cudaMemcpyDeviceToHost);
    }
    memCpyD2HTimer.stop();

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