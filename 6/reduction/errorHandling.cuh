#pragma once
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cassert>
#include <stdexcept>
#include <sstream>

//wrapper to check error codes of cuda functions
inline void checkCudaNonFatal(cudaError_t result)
{
    if (result != cudaSuccess) {
      fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    }
}
//same but with thrown exception
inline void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::stringstream message{""};
        message << "cuda Error: " << err << ": " << cudaGetErrorString(err);
        throw (std::runtime_error(message.str()));
    }
}
//Quit if any previous operation errored
inline void quitOnCudaError() {
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(cudaError));
        exit(-1);
    }
}
