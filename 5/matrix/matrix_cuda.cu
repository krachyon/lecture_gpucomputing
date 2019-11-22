#include "matrix_cuda.h"
#include "memoryWrapper.cuh"
#include <cuda_runtime.h>

//#include <type_traits>
//template<typename T>
//inline typename std::enable_if<std::is_unsigned<T>::value, T>::type  ceildiv (T x, T y)
//{
//    return x / y + (x % y != 0);
//}

inline __host__ __device__ uint32_t ceildiv(uint32_t x, uint32_t y) {
    // division instruction gives you a free modulo. So add one if not cleanly divisible. not that should matter...
    return x / y + (x % y != 0);
}


template<typename T>
__global__ void mmul_naive_kernel(T * mem_left, T * mem_right, T * mem_out, dim3 sizes) {
    //I put gridDim here instead of blockDim and that was a really weird bug, cause most of the tests still passed.
    //TODO Maybe see if e.g.  TEST(mmul_cuda, simple_equality) is just garbage because it still worked...
    uint32_t row = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t col = threadIdx.y + blockIdx.y * blockDim.y;
    //product_size is the size of the scalar product, the amount of columns in left and the amount of rows in right
    uint32_t stride_left = sizes.x;
    uint32_t product_size = sizes.y;
    uint32_t stride_right = sizes.z;

    //If the matrix size is not divisible, just ignore too large indices
    if (row >= stride_left || col >= stride_right) {
        //printf("skipped %i %i\n", row,col);
        return;
    }

    T elem = 0;

    //Todo what about splitting this loop over many threads with either atomic write or some sort of aggregation step?
    for (size_t i = 0; i < product_size; ++i) {
        //elem += left(row, i)*right(i, col) -> _mem[N*row+col];
        elem += mem_left[stride_left * row + i] * mem_right[stride_right * i + col];
    }

    // result has the same amount of rows == stride as left and columns as right
    mem_out[stride_left * row + col] = elem;
}

// generic implementation
template<typename T>
Matrix<T> mmul_cuda_naive(Matrix<T> const& left, Matrix<T> const& right) {
    uint32_t rrows = left.M;
    uint32_t rcols = right.N;
    Matrix<T> ret(rrows, rcols);

    //initialize and copy
    DeviceMemory<T> left_mem(left.data(), left.size());
    DeviceMemory<T> right_mem(right.data(), right.size());
    //just initialize
    DeviceMemory<T> out_mem(ret.size());

    dim3 sizes = {uint32_t(left.M), uint32_t(left.N), uint32_t(right.M)};

    //TODO check heuristic for these
    //ATTENTION putting 0 in any dimension is invalid and does not signify "nonexistent"
    //let's try using thread blocks of 8x8=2 warps. This sucks a bit for very small matrices but then wtf use cuda...

    dim3 blocks{ceildiv(rrows, 8), ceildiv(rcols, 8), 1};
    dim3 threads{8, 8, 1};

    assert(blocks.x * blocks.y * threads.x * threads.y >= ret.size());
    // there should be at most one nearly empty set of blocks
    assert(blocks.x * blocks.y * threads.x * threads.y < (blocks.x + 1) * (blocks.y + 1) * threads.x * threads.y);

    mmul_naive_kernel<T> << < blocks, threads, 0 >> > (left_mem.mem(), right_mem.mem(), out_mem.mem(), sizes);
    //cudaDeviceSynchronize(); // todo needed?
    quitOnCudaError();

    cudaMemcpy(ret.data(), out_mem);
    return ret;
}



template<typename T>
__global__ void mmul_shared_kernel(T * mem_left, T * mem_right, T * mem_out, dim3 sizes) {
    //product_size is the size of the scalar product, the amount of columns in left and the amount of rows in right
    uint32_t stride_left = sizes.x;
    uint32_t product_size = sizes.y;
    uint32_t stride_right = sizes.z;
    uint32_t row = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t col = threadIdx.y + blockIdx.y * blockDim.y;

    //If the matrix size is not divisible, just ignore too large indices
    if (row >= stride_left || col >= stride_right) {
        //printf("skipped %i %i\n", row,col);
        return;
    }

    //fill shared mem
    extern __shared__ float smem[];
    size_t offset_to_right_mem = 0; //TODO
    uint32_t row_max = 8 + blockIdx.x * blockDim.x;
    uint32_t row_min = 0 + blockIdx.x * blockDim.x;
    uint32_t col_max = 8 + blockIdx.y * blockDim.y;
    uint32_t col_min = 0 + blockIdx.y * blockDim.y;

    // every row in the thread block should fetch a row from left matrix and divide it so that the columns do equal work
    {
        uint32_t left_row = row;
        uint32_t n_left_cols = sizes.x;
        uint32_t step_size = ceildiv(n_left_cols, blockDim.y);
        uint32_t start_col = threadIdx.y * step_size;
        uint32_t end_col = threadIdx.y * step_size + step_size;
        for (size_t col_to_copy = start_col; col_to_copy != end_col && col_to_copy != n_left_cols; ++col_to_copy){
            smem[row*blockDim.y+col_to_copy] = mem_left[stride_left*left_row+col_to_copy];
        }
    }



    //wee need

    //do the accessing
}


// fill out overloads
Matrix<float> mmul_cuda_naive(Matrix<float> const& left, Matrix<float> const& right) {
    return mmul_cuda_naive<float>(left, right);
}

Matrix<double> mmul_cuda_naive(Matrix<double> const& left, Matrix<double> const& right) {
    return mmul_cuda_naive<double>(left, right);
}

Matrix<int16_t> mmul_cuda_naive(Matrix<int16_t> const& left, Matrix<int16_t> const& right) {
    return mmul_cuda_naive<int16_t>(left, right);
}



//// or hand instantiate templates
//template void mmul_naive_wrapper<float>(float* mem_a, float* mem_b, float* mem_out, dim3 blocks, dim3 threads);
//template void mmul_naive_wrapper<double>(double* mem_a, double* mem_b, double* mem_out, dim3 blocks, dim3 threads);
//template void mmul_naive_wrapper<int>(int* mem_a, int* mem_b, int* mem_out, dim3 blocks, dim3 threads);


