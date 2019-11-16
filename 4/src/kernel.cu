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

__global__ void globalMem2SharedMem(float* d_mem, size_t n_elements)
{
    extern __shared__ float shared_mem[];
    size_t n_threads = blockDim.x;
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    size_t chunk_size =  n_elements/n_threads;

    __syncthreads();
    for(size_t i =offset; i!=offset+chunk_size; ++i)
        shared_mem[i] = d_mem[i];

    __syncthreads();
}

nanoseconds globalMem2SharedMem_Wrapper(size_t gridSize, size_t blockSize, size_t bytes)
{

    size_t n_elements = bytes*sizeof(float);
    float* d_mem = nullptr;
    checkCuda(cudaMalloc(&d_mem, bytes));

    // chunked access only works if it's cleanly divisible
    assert(n_elements % (blockSize*gridSize) == 0);

    auto start = high_resolution_clock::now();
	globalMem2SharedMem<<< gridSize, blockSize, n_elements*sizeof(float) >>>(d_mem, n_elements);
	auto stop = high_resolution_clock::now();

    checkCuda(cudaFree(d_mem));
    return stop-start;
}

__global__ void SharedMem2globalMem (float* d_mem, size_t n_elements)
{
    extern __shared__ float shared_mem[];
    size_t n_threads = blockDim.x;
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    size_t chunk_size =  n_elements/n_threads;

    __syncthreads();
    for(size_t i =offset; i!=offset+chunk_size; ++i)
        shared_mem[i] = d_mem[i];

    __syncthreads();
}
nanoseconds SharedMem2globalMem_Wrapper(size_t gridSize, size_t blockSize, size_t bytes)
{
    size_t n_elements = bytes*sizeof(float);
    float* d_mem = nullptr;
    checkCuda(cudaMalloc(&d_mem, n_elements* sizeof(float)));

    // chunked access only works if it's cleanly divisible
    assert(n_elements % (blockSize*gridSize) == 0);

    auto start = high_resolution_clock::now();
    SharedMem2globalMem<<< gridSize, blockSize, n_elements*sizeof(float) >>>(d_mem, n_elements);
    auto stop = high_resolution_clock::now();

    checkCuda(cudaFree(d_mem));
    return stop-start;
}

__global__ void 
SharedMem2Registers
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}
void SharedMem2Registers_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
//	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}

__global__ void 
Registers2SharedMem
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}
void Registers2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
//	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}

__global__ void 
bankConflictsRead
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}

void bankConflictsRead_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
//	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}
