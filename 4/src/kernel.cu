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
#include <device_launch_parameters.h>


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

__global__ void Registers2SharedMem(size_t chunk_size, float volatile* volatile dummy)
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

constexpr c64_t max_clock = std::numeric_limits<c64_t>::max();

__global__ void bankConflictsRead(size_t n_iters, size_t stride, double* results)
{
    extern __shared__ float shared_mem[];
    size_t const chunk_size = 64;
    float volatile registers[chunk_size];

    size_t offset = threadIdx.x * chunk_size;

    auto start = clock64();
    for(size_t _=0; _!=n_iters; ++_)
    {
        __syncthreads();
        for(size_t idx = offset; idx!= chunk_size; ++idx)
        {
            registers[idx] = shared_mem[offset+idx*stride];
        }
    }
    auto stop = clock64();
    if(threadIdx.x == 3000)
    {
        printf("not supposed to happen, just to force compiler to write to registers");
        results[0] = registers[0]+registers[63];
    }

    c64_t result = 0;
    if(start>stop)
    {
        printf("I really don't think this should ever happen...");
        result = max_clock-start+stop;
    }
    else
    {
        result = stop-start;
    }

    results[blockIdx.x*blockDim.x+threadIdx.x] = double(result)/n_iters;
}

std::vector<double> bankConflictsRead_Wrapper(size_t gridSize, size_t blockSize, size_t stride, size_t bytes)
{
    size_t const n_iters = 1000;

    assert(gridSize*blockSize <= bytes/sizeof(float)/64/stride ); //if every thread reads 64 elements, that's all we can do;

    double* results_d = nullptr;
    size_t result_bytes = gridSize*blockSize * sizeof(double);
    cudaMalloc(&results_d, result_bytes);

    bankConflictsRead<<< gridSize, blockSize, bytes>>>(n_iters, stride,results_d);
    cudaDeviceSynchronize();
    quitOnCudaError();

    std::vector<double> ret(result_bytes/sizeof(double));

    cudaMemcpy(ret.data(),results_d,result_bytes, cudaMemcpyDeviceToHost);
    cudaFree(results_d);

    return ret;
}
