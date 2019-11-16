/******************************************************************************
 *
 *Computer Engineering Group, Heidelberg University - GPU Computing Exercise 04
 *
 *                  Group : TBD
 *
 *                   File : kernel.cu
 *
 *                Purpose : Memory Operations Benchmark
 *
 ******************************************************************************/

#include "kernel.cuh"

#include <cassert>
#include <chrono>
#include "errorHandling.h"

using std::chrono::nanoseconds;
using std::chrono::high_resolution_clock;

//
// Test Kernel
//

template <typename Callable>
nanoseconds dt(Callable c, size_t n_iters)
{
    auto start = high_resolution_clock::now();
    c();
    auto stop = high_resolution_clock::now();
    return (stop - start)/n_iters;
}

__global__ void globalMem2SharedMem(float * d_mem, size_t n_elements) {
    extern __shared__ float shared_mem[];
    size_t n_threads = blockDim.x;
    size_t chunk_size = n_elements / n_threads;
    size_t offset = threadIdx.x * chunk_size;


    // make sure threads behave like they would in a scenario where we actually use the memory
    __syncthreads();
    for (size_t i = offset; i != offset + chunk_size; ++i)
        shared_mem[i] = d_mem[i];
    __syncthreads();
}

nanoseconds globalMem2SharedMem_Wrapper(size_t gridSize, size_t blockSize, size_t bytes, size_t n_iter) {
    size_t n_elements = bytes / sizeof(float);
    float * d_mem = nullptr;
    checkCuda(cudaMalloc(&d_mem, bytes));

    // chunked access only works if it's cleanly divisible
    assert(n_elements % (blockSize * gridSize) == 0);

    auto start = high_resolution_clock::now();
    for (size_t i = 0; i != n_iter; ++i)
        globalMem2SharedMem <<< gridSize, blockSize, bytes >>> (d_mem, n_elements);
        cudaDeviceSynchronize(); //synchronize after every kernel launch to ensure work get's done
    auto stop = high_resolution_clock::now();
    quitOnCudaError();

    checkCuda(cudaFree(d_mem));
    return (stop - start) / n_iter;
}

__global__ void SharedMem2globalMem(float * d_mem, size_t n_elements) {
    extern __shared__ float shared_mem[];
    size_t n_threads = blockDim.x;
    size_t chunk_size = n_elements / n_threads;
    size_t offset = threadIdx.x * chunk_size;

    __syncthreads();
    for (size_t i = offset; i != offset + chunk_size; ++i)
        shared_mem[i] = d_mem[i];

    __syncthreads();
}

nanoseconds sharedMem2globalMem_Wrapper(size_t gridSize, size_t blockSize, size_t bytes, size_t n_iter) {
    size_t n_elements = bytes / sizeof(float);
    float * d_mem = nullptr;
    checkCuda(cudaMalloc(&d_mem, n_elements * sizeof(float)));

    // chunked access only works if it's cleanly divisible
    assert(n_elements % (blockSize * gridSize) == 0);

    auto start = high_resolution_clock::now();
    for (size_t i = 0; i != n_iter; ++i) {
        SharedMem2globalMem << < gridSize, blockSize, bytes >> > (d_mem, n_elements);
        cudaDeviceSynchronize();
    }
    auto stop = high_resolution_clock::now();
    quitOnCudaError();

    checkCuda(cudaFree(d_mem));
    return (stop - start) / n_iter;
}

//according to the cuda documentation we're guaranteed that ammount of registers per thread.
constexpr size_t n_registers = 63;

__global__ void SharedMem2Registers(size_t n_elements, size_t chunk_size, float& dummy)
{
    extern __shared__ float shared_mem[];
    // need static size to guarantee compiler puts this into registers
    float registers[n_registers];

    size_t offset = threadIdx.x * chunk_size;

    __syncthreads();
    for(size_t i = 0; i!= chunk_size; ++i)
        registers[i] = shared_mem[offset+i];
    __syncthreads();
    // does not do anything but is supposed to confuse the compiler enough to not optimize away access to registers
    // We'll never have that many threads...
    if(threadIdx.x == 3000) {
        dummy = registers[threadIdx.y];
    }
};

nanoseconds sharedMem2Registers_Wrapper(size_t gridSize, size_t blockSize, size_t n_elements, size_t n_iters)
{
    size_t bytes = n_elements * sizeof(float);
    size_t chunk_size = n_elements / blockSize;
    assert(chunk_size < n_registers); // writing outside of the statically allocated register would be no bueno

    float dummy;
    // create a lambda function to pass to the timer; capture everything by value except the dummy parameter
    // it needs to be a reference
    auto time = dt([=, &dummy]() {
        SharedMem2Registers <<< gridSize, blockSize, bytes >>> (n_elements, chunk_size, dummy);
    },n_iters);

    return time;
}

__global__ void
Registers2SharedMem
//(/*TODO Parameters*/)
        () {
    /*TODO Kernel Code*/
}

void Registers2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
//	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}

__global__ void
bankConflictsRead
//(/*TODO Parameters*/)
        () {
    /*TODO Kernel Code*/
}

void bankConflictsRead_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
//	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}
