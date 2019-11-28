#include "reduction.cuh"

#include "memoryWrapper.cuh"

#include <device_launch_parameters.h>
#include <cassert>
//#include <cmath>

//TODO replace log2/ceil with integer math

template<typename T>
__device__ __host__ bool is_power_of_2(T in)
{
    return log2((double)in) == floor(log2((double)in));
};

template<typename T>
__global__ void reduce_kernel_naive(T* __restrict volatile in, T* __restrict out)
//bytes(out) == gridDim.x
{
    auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto const n_total_threads = blockDim.x * gridDim.x;

    //ensure that threadblock has a power of 2 as bytes

    if(tid==0)
    {
        for(auto i =0;i!=n_total_threads*2;++i)
        printf("%f ", in[i]);
        printf("\n");
    }
    printf("thread %i before loop\n",threadIdx.x);

    uint32_t threads_alive = blockDim.x;

    // first computation: Fetch elements on right side of memory
    in[tid] += in[tid+n_total_threads];
    threads_alive >>= 1;
    //now all we care about is the memory block that corresponds to our block size

    __syncthreads();
    printf("[%i] += [%i]\n",tid, tid+n_total_threads);

    while(threads_alive!=0 && threadIdx.x < threads_alive)
    {
        __syncthreads();
        printf("threads alive %i\n", threads_alive);
        printf("%f [%i] += %f [%i]\n", in[tid], tid, in[tid+threads_alive], tid+threads_alive);
        __syncthreads();

        in[tid] += in[tid+threads_alive];
        threads_alive >>= 1;

        __syncthreads();
    }
    printf("thread %i after loop\n",threadIdx.x);

    if(threadIdx.x != 0)
        return;
    else if (threadIdx.x == 0)
    {
        __syncthreads();
        printf("result %f\n",in[tid]);
        // We are the last thread in the block; "report" the result
        out[blockIdx.x] = in[tid];
        return;
    }
}

template<typename T>
T reduce_cuda_naive(std::vector<T>& in, uint32_t const n_blocks)
{
    //zero pad end of vector if it doesn't fit
    if(!is_power_of_2(in.size()))
    {
        uint32_t next_power = ceil(log2((double)in.size()));
        uint32_t new_size = pow(2,next_power);
        in.resize(new_size, 0);
    }
    uint32_t threads_total = in.size()/2;
    uint32_t threads_per_block = threads_total / n_blocks;

    assert(threads_per_block <= 1024);
    assert(is_power_of_2(threads_per_block));
    assert(is_power_of_2(n_blocks));

    DeviceMemory<T> d_in(in.data(), in.size());
    DeviceMemory<T> d_out(n_blocks);
    DeviceMemory<T> d_final_out(1);

    //output is a single value per block
    reduce_kernel_naive<T><<<n_blocks,threads_per_block>> > (d_in.mem(), d_out.mem());
    cudaDeviceSynchronize();
    //use a single block with n_blocks threads to do final summation
    //TODO this does not currently work with a single block in the first iteration as the kernel will do a first summation
    // regardless. See if this can be adapted
    reduce_kernel_naive<T><< < 1, n_blocks/2 >> > (d_out.mem(),d_final_out.mem());

    auto result = d_final_out.to_vector();
    assert(result.size() == 1);
    return result[0];
}

//relevant implementations of reduction.h

float reduce_cuda_naive(std::vector<float>& in, uint32_t const n_blocks)
{
    return reduce_cuda_naive<float>(in, n_blocks);
}
