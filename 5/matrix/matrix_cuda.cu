#include "matrix_cuda.h"
#include "memoryWrapper.cuh"
#include <cuda_runtime.h>


template<typename T>
__global__ void mmul_naive_kernel(T* mem_left, T* mem_right, T* mem_out, dim3 sizes)
{
    //TODO smarter way of handling threads and better naming/error checks for matrix dimensions
    uint32_t row = threadIdx.x;
    uint32_t col = threadIdx.y;

    uint32_t stride_left = sizes.x;
    uint32_t product_size = sizes.y;
    uint32_t stride_right = sizes.z;

    //product_size is the size of the scalar product, the amount of columns in left and the amount of rows in right

    T elem = 0;
    for (size_t i=0; i<product_size; ++i) {
        //elem += left(row, i)*right(i, col) -> _mem[N*row+col];
        elem += mem_left[stride_left*row+i] * mem_right[stride_right*i+col];
    }

    // result has the same amount of columns == stride as left
    mem_out[stride_left*row+col] = elem;
}

// generic implementation
template <typename T>
Matrix<T> mmul_cuda_naive (Matrix<T> const& left, Matrix<T> const& right)
{
    uint32_t rrows = left.M;
    uint32_t rcols = right.N;
    Matrix<T> ret(rrows,rcols);

    //initialize and copy
    DeviceMemory<T> left_mem(left.data(), left.size());
    DeviceMemory<T> right_mem(right.data(), right.size());
    //just initialize
    DeviceMemory<T> out_mem(ret.size());

    //TODO create a sensible heuristic for these
    //ATTENTION putting 0 in any dimension is a bad idea...
    //dim3 blocks{1,1,1};
    dim3 threads{rrows,rcols,1};
    dim3 sizes = {uint32_t(left.M), uint32_t(left.N), uint32_t(right.M)};
    mmul_naive_kernel<T><<<1,threads,0>>>(left_mem.mem(), right_mem.mem(), out_mem.mem(), sizes);
    cudaDeviceSynchronize();
    quitOnCudaError();

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


