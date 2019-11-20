#include "matrix_cuda.h"
#include "memoryWrapper.cuh"
#include <cuda_runtime.h>

template<typename T>
__global__ void mmul_naive_kernel(T* mem_a, T* mem_b, T* mem_out)
{
    //TODO need to know about the matrix dimensions here
}

// generic implementation
template <typename T>
Matrix<T> mmul_cuda_naive (Matrix<T> const& left, Matrix<T> const& right)
{
    size_t rrows = left.M;
    size_t rcols = right.N;
    Matrix<T> ret(rrows,rcols);

    //initialize and copy
    DeviceMemory<T> left_mem(left.data(), left.size());
    DeviceMemory<T> right_mem(right.data(), right.size());
    //just initialize
    DeviceMemory<T> out_mem(ret.memsize());

    //TODO create a sensible heuristic for these
    dim3 blocks{1,0,0};
    dim3 threads{1,0,0};
    mmul_naive_kernel<T><<<blocks,threads>>>(left_mem.mem(), right_mem.mem(), out_mem.mem());

    cudaMemcpy(ret.data(), out_mem);
    return ret;
}

// fill out overloads
Matrix<float> mmul_cuda_naive_float (Matrix<float> const& left, Matrix<float> const& right)
{return mmul_cuda_naive<float>(left,right);}
Matrix<double> mmul_cuda_naive_double (Matrix<double> const& left, Matrix<double> const& right)
{return mmul_cuda_naive<double>(left,right);}


////hand instantiate templates
//template void mmul_naive_wrapper<float>(float* mem_a, float* mem_b, float* mem_out, dim3 blocks, dim3 threads);
//template void mmul_naive_wrapper<double>(double* mem_a, double* mem_b, double* mem_out, dim3 blocks, dim3 threads);
//template void mmul_naive_wrapper<int>(int* mem_a, int* mem_b, int* mem_out, dim3 blocks, dim3 threads);


