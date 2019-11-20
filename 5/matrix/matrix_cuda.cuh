#pragma once
#include <cuda_runtime.h>

template<typename T>
__global__ void mmul_naive_kernel(T* mem_a, T* mem_b, T* mem_out);


template <typename T>
void mmul_naive_wrapper(T* mem_a, T* mem_b, T* mem_out, dim3 blocks, dim3 threads);



