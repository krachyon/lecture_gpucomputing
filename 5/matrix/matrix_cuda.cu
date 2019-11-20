#include "matrix_cuda.cuh"

template<typename T>
__global__ void mmul_naive_kernel(T* mem_a, T* mem_b, T* mem_out)
{

}

template <typename T>
void mmul_naive_wrapper(T* mem_a, T* mem_b, T* mem_out, dim3 blocks, dim3 threads)
{
    return mmul_naive_kernel<<<blocks,threads>>>(mem_a, mem_b, mem_out);
}


//TODO
template void mmul_naive_wrapper<float>(float* mem_a, float* mem_b, float* mem_out, dim3 blocks, dim3 threads);
template void mmul_naive_wrapper<double>(double* mem_a, double* mem_b, double* mem_out, dim3 blocks, dim3 threads);
template void mmul_naive_wrapper<int>(int* mem_a, int* mem_b, int* mem_out, dim3 blocks, dim3 threads);


