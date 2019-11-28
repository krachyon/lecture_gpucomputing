#include "reduction.cuh"
#include "tracing.h"
#include "memoryWrapper.cuh"
#include <device_launch_parameters.h>
#include <cassert>


template<typename T>
__global__ void reduce_kernel_shared(T* __restrict volatile in, T* __restrict out)
{
    extern __shared__ float smem[];
    auto const tid_glob = threadIdx.x + blockIdx.x * blockDim.x;
    auto const tid_loc = threadIdx.x;
    auto const n_total_threads = blockDim.x * gridDim.x;

    uint32_t threads_alive = blockDim.x;

    smem[tid_loc] = in[tid_glob+n_total_threads];
    smem[tid_loc] += in[tid_glob];

    threads_alive >>= 1;
    //now all we care about is the memory block that corresponds to our block size

    while(threads_alive!=0 && threadIdx.x < threads_alive)
    {
        __syncthreads();
        smem[tid_loc] += smem[tid_loc+threads_alive];
        threads_alive >>= 1;
    }
    __syncthreads();

    if (threadIdx.x == 0)
        out[blockIdx.x] = smem[tid_loc];

}


template<typename T>
T reduce_cuda_shared(std::vector<T>& in, uint32_t const n_blocks)
{
    Trace::set("cuda_shared_start");
    //zero pad end of vector if it doesn't fit
    if(!is_power_of_2(in.size()))
    {
        uint32_t next_power = ceil(log2((double)in.size()));
        uint32_t new_size = pow(2,next_power);
        in.resize(new_size, 0);
    }
    uint32_t const threads_total = in.size()/2;
    uint32_t const threads_per_block = threads_total / n_blocks;
    uint32_t const shared_size  = threads_per_block * sizeof(T);

    assert(threads_per_block <= 1024);
    assert(is_power_of_2(threads_per_block));
    assert(is_power_of_2(n_blocks));

    Trace::set("cuda_shared_copy_in");
    DeviceMemory<T> d_in(in.data(), in.size());
    Trace::set("cuda_shared_copy_in_done");
    DeviceMemory<T> d_out(n_blocks);
    DeviceMemory<T> d_final_out(1);

    //output is a single value per block
    reduce_kernel_shared<T><<<n_blocks,threads_per_block, shared_size>> > (d_in.mem(), d_out.mem());
    cudaDeviceSynchronize();

    std::vector<T> result(0);

    //use a single block with n_blocks threads to do final summation
    if(n_blocks>1) {
        reduce_kernel_shared<T> << < 1, n_blocks / 2 >> > (d_out.mem(), d_final_out.mem());
        Trace::set("cuda_shared_copy_out");
        result = d_final_out.to_vector();
        Trace::set("cuda_shared_copy_out_done");
    }
        //unless we're already done;
    else{
        result.push_back(d_out.to_vector()[0]);
    }

    assert(result.size() == 1);
    Trace::set("cuda_shared_end");
    return result[0];
}

#include <thrust/device_vector.h>

float thrust_reduce(std::vector<float>const& in)
{
    thrust::device_vector<float>(in.begin(),in.end());
    float sum = thrust::reduce(in.begin(), in.end());
    return sum;
}

