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
#include <iostream>

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



__global__ void SharedMem2Registers(size_t chunk_size, float volatile* volatile dummy)
{
    extern __shared__ float shared_mem[];
    // ideally need static size<63 to ensure these are registers, but then this would have to be a template
    // and then the we'd have to summon the TMP Cuthulhu to iterate...
    float registers[n_registers];

    size_t offset = threadIdx.x * chunk_size;

    __syncthreads();
    for(size_t i = 0; i!= chunk_size; ++i) {
        registers[i] = shared_mem[offset + i];
    }
    __syncthreads();
    // does not do anything but is supposed to confuse the compiler enough to not optimize away access to registers
    // We'll never have that many threads...
    // This is quite the challenge to get right, according to compilerExplorer the code does not vanish but I'm not sure
    // Why all the kernels take the same time then
    if(threadIdx.x == 1025) {
        printf("We should never have hit this");
        *dummy = registers[chunk_size/2];
    }
};

nanoseconds sharedMem2Registers_Wrapper(size_t gridSize, size_t blockSize, size_t bytes, size_t n_iters)
{
    size_t n_elements =  bytes / sizeof(float);
    size_t chunk_size = n_elements / blockSize;
    assert(chunk_size < n_registers); // writing outside of the statically allocated register would be no bueno

    //Wow so you can't pass a float& as an output variable to the kernel because it converts it to a device pointer
    //which you then can't dereference. That's some next level bullshit that it let's you do it but creates
    //wrong code...
    float* dummy = nullptr;
    cudaMalloc(&dummy, sizeof(float));

    // create a lambda function to pass to the timer; capture everything by value except the dummy parameter,
    // it needs to be a reference
    auto time = dt([=, &dummy]() {
        SharedMem2Registers <<< gridSize, blockSize, bytes >>> (chunk_size, dummy);
        cudaDeviceSynchronize();
    },n_iters);
    quitOnCudaError();

    cudaFree(dummy);
    return time;
}

__global__ void Registers2SharedMem(size_t chunk_size, float* dummy)
{
    extern __shared__ float shared_mem[];
    // ideally need static size<63 to ensure these are registers, but then this would have to be a template
    // and then the we'd have to summon the TMP Cuthulhu to iterate...
    float registers[n_registers];

    size_t offset = threadIdx.x * chunk_size;

    __syncthreads();
    for(size_t i = 0; i!= chunk_size; ++i)
        shared_mem[offset+i] = registers[i];
    __syncthreads();

    if(threadIdx.x == 1025) {
        printf("We should never have hit this");
        *dummy = shared_mem[chunk_size/2];
    }
}

nanoseconds Registers2SharedMem_Wrapper(size_t gridSize, size_t blockSize, size_t bytes, size_t n_iters)
{
    size_t n_elements =  bytes / sizeof(float);
    size_t chunk_size = n_elements / blockSize;
    assert(chunk_size < n_registers); // writing outside of the statically allocated register would be no bueno

    float* dummy = nullptr;
    cudaMalloc(&dummy, sizeof(float));
    
    // create a lambda function to pass to the timer; capture everything by value except the dummy parameter,
    // it needs to be a reference
    auto time = dt([=, &dummy]() {
        Registers2SharedMem <<< gridSize, blockSize, bytes >>> (chunk_size, dummy);
        cudaDeviceSynchronize();
    },n_iters);
    quitOnCudaError();

    cudaFree(dummy);
    return time;
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
