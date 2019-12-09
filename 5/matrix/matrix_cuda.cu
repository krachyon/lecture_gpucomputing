#include "matrix_cuda.h"
#include "memoryWrapper.cuh"
#include <cuda_runtime.h>


//#include <type_traits>
//template<typename T>
//inline typename std::enable_if<std::is_unsigned<T>::value, T>::type  ceildiv (T x, T y)
//{
//    return x / y + (x % y != 0);
//}

inline __device__ __host__ uint32_t ceildiv(uint32_t x, uint32_t y) {
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
    uint32_t stride_left = sizes.y;
    uint32_t product_size = sizes.y;
    uint32_t stride_right = sizes.z;
    uint32_t stride_out = stride_right;

    //If the matrix size is not divisible, just ignore too large indices
    if (row >= sizes.x || col >= sizes.z) {
        //printf("skipped %i %i\n", row,col);
        return;
    }

    T elem = 0;

    //Todo what about splitting this loop over many threads with either atomic write or some sort of aggregation step?
    for (size_t i = 0; i < product_size; ++i) {
        //elem += left(row, i)*right(i, col) -> _mem[N*row+col];
        elem += mem_left[stride_left * row + i] * mem_right[stride_right * i + col];

    // result has the same amount of colunmns == stride as right
    mem_out[stride_out * row + col] = elem;
}

// generic implementation
template<typename T>
Matrix<T> mmul_cuda_naive(Matrix<T> const& left, Matrix<T> const& right, uint32_t n_threads) {
    uint32_t rrows = left.M;
    uint32_t rcols = right.N;
    Matrix<T> ret(rrows, rcols);

    auto mem_start = std::chrono::high_resolution_clock::now();
    //initialize and copy
    DeviceMemory<T> left_mem(left.data(), left.size());
    DeviceMemory<T> right_mem(right.data(), right.size());
    //just initialize
    DeviceMemory<T> out_mem(ret.size());
    auto mem_stop = std::chrono::high_resolution_clock::now();
    auto mem_duration = mem_stop - mem_start;

    dim3 sizes = {uint32_t(left.M), uint32_t(left.N), uint32_t(right.N)};

    //TODO check heuristic for these
    //ATTENTION putting 0 in any dimension is invalid and does not signify "nonexistent"
    //let's try using thread blocks of 8x8=2 warps. This sucks a bit for very small matrices but then wtf use cuda...

    dim3 blocks{ceildiv(rrows, n_threads), ceildiv(rcols, n_threads), 1};
    dim3 threads{n_threads, n_threads, 1};

    assert(blocks.x * blocks.y * threads.x * threads.y >= ret.size());
    // there should be at most one nearly empty set of blocks
    assert(blocks.x * blocks.y * threads.x * threads.y < (blocks.x + 1) * (blocks.y + 1) * threads.x * threads.y);

    mmul_naive_kernel<T> << < blocks, threads, 0 >> > (left_mem.mem(), right_mem.mem(), out_mem.mem(), sizes);
    cudaDeviceSynchronize(); // todo needed?
    quitOnCudaError();

    mem_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(ret.data(), out_mem);
    mem_stop = std::chrono::high_resolution_clock::now();

    mem_duration += mem_stop - mem_start;
    lastMemoryOp = mem_duration;
    return ret;
}



// TODO this turned out to be much more of a problem than if it where just square matrices.
// Not sure how efficient all the index mess is, but this should be started with as many threads as possible so that
// a block uses the maximum possible amount of shared memory as to make the copying worth it.
template<typename T>
__global__ void mmul_shared_kernel(T* mem_left, T* mem_right, T* mem_out, dim3 sizes) {
    //product_size is the size of the scalar product, the amount of columns in left and the amount of rows in right
    uint32_t stride_left = sizes.y;
    uint32_t product_size = sizes.y;
    uint32_t stride_right = sizes.z;
    uint32_t row = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t col = threadIdx.y + blockIdx.y * blockDim.y;

    //range of rows in the left matrix and cols in the right matrix that we need in this block
    extern __shared__ float smem[];
    uint32_t row_left_max  = blockDim.x + blockIdx.x * blockDim.x;
    uint32_t row_left_min  = 0          + blockIdx.x * blockDim.x;
    uint32_t col_right_max = blockDim.y + blockIdx.y * blockDim.y;
    uint32_t col_right_min = 0          + blockIdx.y * blockDim.y;

    // Row,Col coordinates to memory location
    auto left_idx = [=](size_t row, size_t col)
    {
        return stride_left*row + col;
    };
    auto right_idx = [=](size_t row, size_t col)
    {
        return stride_right*row + col;
    };

    // Helpers to convert row,col coordinates on original matrix to shared memory location
    auto left2shared = [=] (size_t row, size_t col) -> size_t
    {
        assert(row >= row_left_min && row <= row_left_max && col <= col_right_max && col >= col_right_max);
        auto memory_location = (row - row_left_min) * row_left_max + (col - col_right_min);
        return memory_location;
    };

    auto right2shared = [=] (size_t row, size_t col) -> size_t
    {
        auto offset_to_right_mem = sizes.x * sizes.y * sizeof(T); //amount of elements in left matrix
        return offset_to_right_mem + left2shared(row, col);
    };

    // every row in the thread block should fetch the corresponding row
    // from left matrix and divide it so that the threadblock columns do equal work
    {
        uint32_t n_left_cols = sizes.x;
        uint32_t step_size = ceildiv(n_left_cols, blockDim.y);

        uint32_t start_col = threadIdx.y * step_size;
        uint32_t end_col = threadIdx.y * step_size + step_size;

        for (size_t col_to_copy = start_col;
            (col_to_copy != end_col) && (col_to_copy != n_left_cols);
             ++col_to_copy)
        {
            smem[left2shared(row,col_to_copy)] = mem_left[left_idx(row, col_to_copy)];
        }
    }

    // every row in the thread block should fetch the corresponding column
    // from right matrix and divide it so that the threadblock columns do equal work
    {
        uint32_t n_right_rows = sizes.z;
        uint32_t step_size = ceildiv(n_right_rows, blockDim.y);

        uint32_t start_row = threadIdx.y * step_size;
        uint32_t end_row = threadIdx.y * step_size + step_size;

        // The ceildiv operation could overshoot. In that case just end loop if we reach n_right_rows
        // TODO right now the same element can be copied multiple times. include check to avoid that?
        // TODO Should not be a functional problem, as it's writing the same thing and does not care about the original
        // memory value. Also adapt in above block
        for (size_t row_to_copy = start_row;
             (row_to_copy != end_row) && (row_to_copy != n_right_rows);
             ++row_to_copy)
        {
            smem[right2shared(row_to_copy,col)] = mem_right[right_idx(row_to_copy, col)];
        }
    }
    //make sure all the shared memory is available
    __syncthreads();

    //If the matrix size is not cleanly tileable, just ignore too large indices
    if (row >= stride_left || col >= stride_right) {
        return;
    }

    T elem;
    for (size_t i = 0; i < product_size; ++i) {
        //elem += left(row, i)*right(i, col) -> _mem[N*row+col];
        elem += smem[left2shared(row,i)] * smem[right2shared(i,col)];
    }

    mem_out[stride_left * row + col] = elem;

}


//easier version that only understands NxN matrices
template<typename T>
__global__ void mmul_shared_kernel_NN(T* mem_left, T* mem_right, T* mem_out, uint32_t N) {
    uint32_t row = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t col = threadIdx.y + blockIdx.y * blockDim.y;
    //If the matrix size is not divisible, just ignore too large indices
    if (row >= N || col >= N) {
        //printf("skipped %i %i\n", row,col);
        return;
    }

    uint32_t min_row = 0          + blockIdx.x * blockDim.x;
//    uint32_t max_row = blockDim.x + blockIdx.x * blockDim.x;
    uint32_t min_col = 0          + blockIdx.y * blockDim.y;
//    uint32_t max_col = blockDim.x + blockIdx.x * blockDim.x;

    uint32_t elements_in_block = blockDim.x * blockDim.y;

    // works for both matrices, both are NxN and N=stride
    auto matrix_idx = [=](uint32_t row, uint32_t col) -> uint32_t {
        return N * row + col;
    };
    __shared__ extern float smem[];

    //for a given iteration (each iteration moves the block window by one step) compute indices in original matrices
    auto sliding_idx_left = [=](uint32_t i){
        return matrix_idx(row, i*blockDim.y+threadIdx.y);
    };

    auto sliding_idx_right = [=](uint32_t i){
        return matrix_idx(i*blockDim.x+threadIdx.x, col);
    };

    //TODO this should be different between rows and cols of scalar product
    auto scalar_prod_index = [=](uint32_t j)
    {
        return threadIdx.x * blockDim.x + j;
    };


    //always write to these locations
    uint32_t left_element = (row-min_row) * blockDim.y + (col-min_col);
    uint32_t right_element = elements_in_block + left_element;

    T output_elem = 0;

    for(size_t i=0;i<ceildiv(N,blockDim.y);++i) {
        __syncthreads();
        smem[left_element] = mem_left[sliding_idx_left(i)];
        smem[right_element] = mem_right[sliding_idx_right(i)];
        __syncthreads();
        for(size_t j=0;j!=blockDim.y;++j) {
            output_elem += smem[scalar_prod_index(j)] * smem[scalar_prod_index(j)+elements_in_block];
        }
    }

    mem_out[matrix_idx(row,col)] = output_elem;
    //mem_out[matrix_idx(row,col)] = threadIdx.x*threadIdx.y;
}


template<typename T>
Matrix<T> mmul_cuda_shared(Matrix<T> const& left, Matrix<T> const& right, uint32_t n_threads) {
    assert(left.M == right.M && left.N == right.N);


    uint32_t N = left.M;

    Matrix<T> ret(N, N);

    auto mem_start = std::chrono::high_resolution_clock::now();
    //initialize and copy
    DeviceMemory<T> left_mem(left.data(), left.size());
    DeviceMemory<T> right_mem(right.data(), right.size());
    //just initialize
    DeviceMemory<T> out_mem(ret.size());
    auto mem_stop = std::chrono::high_resolution_clock::now();
    auto mem_duration = mem_stop - mem_start;

    dim3 blocks{ceildiv(N, n_threads), ceildiv(N, n_threads), 1};
    dim3 threads{n_threads, n_threads, 1};

    assert(blocks.x * blocks.y * threads.x * threads.y >= ret.size());
    // there should be at most one nearly empty set of blocks
    assert(blocks.x * blocks.y * threads.x * threads.y < (blocks.x + 1) * (blocks.y + 1) * threads.x * threads.y);

    size_t shared_mem_size = sizeof(T) * 2 * 8 * 8;
    mmul_shared_kernel_NN<T> << < blocks, threads, shared_mem_size>> > (left_mem.mem(), right_mem.mem(), out_mem.mem(), N);
    cudaDeviceSynchronize(); // todo needed?
    quitOnCudaError();
    mem_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(ret.data(), out_mem);
    mem_stop = std::chrono::high_resolution_clock::now();

    mem_duration += mem_stop - mem_start;
    lastMemoryOp = mem_duration;
    return ret;

}


// fill out overloads
Matrix<float> mmul_cuda_naive(Matrix<float> const& left, Matrix<float> const& right, uint32_t n_threads) {
    return mmul_cuda_naive<float>(left, right,n_threads);
}

Matrix<double> mmul_cuda_naive(Matrix<double> const& left, Matrix<double> const& right, uint32_t n_threads) {
    return mmul_cuda_naive<double>(left, right,n_threads);
}

Matrix<int16_t> mmul_cuda_naive(Matrix<int16_t> const& left, Matrix<int16_t> const& right, uint32_t n_threads) {
    return mmul_cuda_naive<int16_t>(left, right,n_threads);
}


Matrix<float> mmul_cuda_shared(Matrix<float> const& left, Matrix<float> const& right, uint32_t n_threads) {
    return mmul_cuda_shared<float>(left, right,n_threads);
}

Matrix<double> mmul_cuda_shared(Matrix<double> const& left, Matrix<double> const& right, uint32_t n_threads) {
    return mmul_cuda_shared<double>(left, right,n_threads);
}


//// or hand instantiate templates
//template void mmul_naive_wrapper<float>(float* mem_a, float* mem_b, float* mem_out, dim3 blocks, dim3 threads);
//template void mmul_naive_wrapper<double>(double* mem_a, double* mem_b, double* mem_out, dim3 blocks, dim3 threads);
//template void mmul_naive_wrapper<int>(int* mem_a, int* mem_b, int* mem_out, dim3 blocks, dim3 threads);


