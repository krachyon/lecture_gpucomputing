/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 03
 *
 *                           Group : TBD
 *
 *                            File : main.cu
 *
 *                         Purpose : Memory Operations Benchmark
 *
 *************************************************************************************************/
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

//
// Kernels
//

__device__ __host__ size_t flat(dim3 v)
{
    return v.x*v.y*v.z;
}

__global__ void globalMemCoalescedKernel(float* in, float* out, size_t n_elements)
{
    //attention: n_elements needs to be cleanly divisible by n_threads
    size_t n_threads = flat(blockDim);
    size_t id = flat(threadIdx);
    size_t chunk = n_elements/n_threads;

    for (size_t i = chunk*id; i!=chunk*id+chunk; ++i) {
        out[i] = in[i];
    }
}

void globalMemCoalescedKernel_Wrapper(dim3 gridDim, dim3 blockDim, float* in, float* out, size_t n_elements)
{
    globalMemCoalescedKernel <<<gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(in, out, n_elements);
}

__global__ void globalMemStrideKernel(float* in, float* out, size_t stride)
{
    out[threadIdx.x] = in[stride*threadIdx.x];
}

void globalMemStrideKernel_Wrapper(dim3 gridDim, dim3 blockDim, float* in, float* out, size_t stride)
{
    globalMemStrideKernel <<<gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(in, out, stride);
}

__global__ void globalMemOffsetKernel(float* in, float* out, size_t offset)
{
    out[threadIdx.x] = in[threadIdx.x+offset];
}

void globalMemOffsetKernel_Wrapper(dim3 gridDim, dim3 blockDim, float* in, float* out, size_t offset)
{
    globalMemOffsetKernel <<<gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(in, out, offset);
}

