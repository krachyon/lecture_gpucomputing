#pragma once
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cassert>

//wrapper to check error codes of cuda functions
inline void checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
      fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    }
    //todo just throw exception here?
}
inline void quitOnCudaError() {
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(cudaError));
        exit(-1);
    }
}