#include "reduction.cuh"

#include "memoryWrapper.cuh"

#include <device_launch_parameters.h>
#include <cassert>

#include "tracing.h"


template<typename T>
__global__ void reduce_kernel_naive(T* __restrict volatile in, T* __restrict out)
//bytes(out) == gridDim.x
{
    auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto const n_total_threads = blockDim.x * gridDim.x;

    uint32_t threads_alive = blockDim.x;

    // first computation: Fetch elements on right side of memory
    in[tid] += in[tid+n_total_threads];
    threads_alive >>= 1;
    //now all we care about is the memory block that corresponds to our block size

    while(threads_alive!=0 && threadIdx.x < threads_alive)
    {
        __syncthreads();
        in[tid] += in[tid+threads_alive];
        threads_alive >>= 1;
    }

    __syncthreads();

    if (threadIdx.x == 0)
        out[blockIdx.x] = in[tid];
}

template<typename T>
T reduce_cuda_naive(std::vector<T>& in, uint32_t const n_blocks, size_t iters) {
    Trace::set(tracepoint::start);
    //zero pad end of vector if it doesn't fit
    if (!is_power_of_2(in.size())) {
        uint32_t next_power = ceil(log2((double) in.size()));
        uint32_t new_size = pow(2, next_power);
        in.resize(new_size, 0);
    }
    uint32_t const threads_total = in.size() / 2;
    uint32_t const threads_per_block = threads_total / n_blocks;

    assert(threads_per_block <= 1024);
    assert(is_power_of_2(threads_per_block));
    assert(is_power_of_2(n_blocks));

    Trace::set(tracepoint::copy_start);
    DeviceMemory<T> d_in(in.data(), in.size());
    Trace::set(tracepoint::copy_end);
    DeviceMemory<T> d_out(n_blocks);
    DeviceMemory<T> d_final_out(1);

    //output is a single value per block
    for (auto i = 0; i != iters; ++i) {
        reduce_kernel_naive<T> << < n_blocks, threads_per_block >> > (d_in.mem(), d_out.mem());
        cudaDeviceSynchronize();
    }
    throwOnCudaError();

    std::vector<T> result(0);

    //use a single block with n_blocks threads to do final summation
    if(n_blocks>1) {
        reduce_kernel_naive<T> << < 1, n_blocks / 2 >> > (d_out.mem(), d_final_out.mem());
        Trace::set(tracepoint::backcopy_start);
        result = d_final_out.to_vector();
        Trace::set(tracepoint::backcopy_end);
    }
    //unless we're already done;
    else{
        result.push_back(d_out.to_vector()[0]);
    }

    assert(result.size() == 1);
    Trace::set(tracepoint::end);
    return result[0];
}

//relevant implementations of reduction.h

float reduce_cuda_naive(std::vector<float>& in, uint32_t const n_blocks, size_t iters)
{
    return reduce_cuda_naive<float>(in, n_blocks,iters);
}
double reduce_cuda_naive(std::vector<double>& in, uint32_t const n_blocks, size_t iters)
{
    return reduce_cuda_naive<double>(in, n_blocks,iters);
}
uint32_t reduce_cuda_naive(std::vector<uint32_t>& in, uint32_t const n_blocks, size_t iters)
{
    return reduce_cuda_naive<uint32_t>(in, n_blocks,iters);
}
int32_t reduce_cuda_naive(std::vector<int32_t>& in, uint32_t const n_blocks, size_t iters)
{
    return reduce_cuda_naive<int32_t>(in, n_blocks,iters);
}
int16_t reduce_cuda_naive(std::vector<int16_t>& in, uint32_t const n_blocks, size_t iters)
{
    return reduce_cuda_naive<int16_t>(in, n_blocks,iters);
}
uint16_t reduce_cuda_naive(std::vector<uint16_t>& in, uint32_t const n_blocks, size_t iters)
{
    return reduce_cuda_naive<uint16_t>(in, n_blocks,iters);
}

