#include "nbody.h"
#include "utility.cuh"
#include <device_launch_parameters.h>
#include <random>
#include "errorHandling.cuh"

__global__ void velocity_halfstep(float3* positions, float3* velocities, float const* masses, size_t N)
{
    //index computation etc.
    extern __shared__ float smem[]; // size == N * sizeof(float3) + N * sizeof(float)
    uint32_t tid = threadIdx.x + gridDim.x * blockIdx.x;
    if(tid>=N)
        return;

    float3* spos = (float3*)(smem);
    float* sm = (float*)&spos[N];

    uint32_t tiles = ceildiv(N, blockDim.x);
    uint32_t last_tile_size = N % blockDim.x;

    float3 accel{0.f,0.f,0.f};
    float3 ref_pos = positions[tid];

    // loops to calculate acceleration on particle, using a tile the width of blockDim
    // last, potentially ill-fitting tile get's special treatment
    for (uint32_t tile = 0; tile != tiles - 1; ++tile) {
        size_t to_fetch = blockDim.x * tile + threadIdx.x;

        spos[threadIdx.x] = positions[to_fetch];
        sm[threadIdx.x] = masses[to_fetch];
        __syncthreads();
        for (uint32_t tile_idx = 0; tile_idx != blockDim.x; ++tile_idx) {
            auto diff = spos[tile_idx] - ref_pos;
            accel += G * sm[tile_idx] * diff / (norm_pow3(diff) + eps);
        }
    }
    //now tile == tiles-1
    size_t to_fetch = blockDim.x * (tiles - 1) + threadIdx.x;
    // we only have work to do if our part of the tile is overlapping the data
    if (to_fetch < N) {
        __syncthreads();
        spos[threadIdx.x] = positions[to_fetch];
        sm[threadIdx.x] = masses[to_fetch];
        __syncthreads();
    }
    for (uint32_t tile_idx = 0; tile_idx != last_tile_size; ++tile_idx) {
        auto diff = spos[tile_idx] - ref_pos;
        accel += G * sm[tile_idx] * diff / (norm_pow3(diff) + eps);
    }

    //finally, update main storage
    velocities[tid] = accel;
}

__global__ void position_step(float3* positions, float3* velocities, float const* masses, size_t N)
{
    extern __shared__ float smem[]; // size == N * 2 * sizeof(float3)
    uint32_t tid = threadIdx.x + gridDim.x * blockIdx.x;
    if(tid>=N)
        return;

    float3* spos = (float3*)(smem);
    float3* sv = (float3*)&spos[N];

    uint32_t tiles = ceildiv(N, blockDim.x);
    uint32_t last_tile_size = N % blockDim.x;
    float3 dv{0,0,0};

    for (uint32_t tile = 0; tile != tiles - 1; ++tile) {
        size_t to_fetch = blockDim.x * tile + threadIdx.x;

        spos[threadIdx.x] = positions[to_fetch];
        sv[threadIdx.x] = velocities[to_fetch];
        __syncthreads();
        for (uint32_t tile_idx = 0; tile_idx != blockDim.x; ++tile_idx) {
            dv += sv[tile_idx];
        }
    }
    size_t to_fetch = blockDim.x * (tiles - 1) + threadIdx.x;
    // we only have work to do if our part of the tile is overlapping the data
    if (to_fetch < N) {
        __syncthreads();
        spos[threadIdx.x] = positions[to_fetch];
        sv[threadIdx.x] = velocities[to_fetch];
        __syncthreads();
    }
    for (uint32_t tile_idx = 0; tile_idx != last_tile_size; ++tile_idx) {
        dv += sv[tile_idx];
    }
    positions[tid] += dv * dt;
}


timed<thrust::host_vector<float3>> run_leapfrog_soa(size_t N, size_t threads_per_block, size_t iters)
{
    thrust::device_vector<float3> positions;
    thrust::device_vector<float3> velocities;
    thrust::device_vector<float> masses;

    std::default_random_engine generator(0xdeadbeef);
    std::uniform_real_distribution<float> position_distribution(-1,1);
    auto x = std::bind(position_distribution, generator);
    std::uniform_real_distribution<float> mass_distribution(0.1f,5.f);
    auto m = std::bind(mass_distribution, generator);

    for(auto _=0;_!=N;++_) {
        positions.push_back({x(),x(),x()});
        masses.push_back(m());
        velocities.push_back({0,0,0});
    }
    size_t n_blocks = ceildiv(N,threads_per_block);

    auto start = std::chrono::high_resolution_clock::now();

    for(auto _=0; _!=iters; ++_) {
        velocity_halfstep<<<n_blocks,threads_per_block,N * sizeof(float3) + N * sizeof(float)>>>
        (thrust::raw_pointer_cast(positions.data()),thrust::raw_pointer_cast(velocities.data()),thrust::raw_pointer_cast(masses.data()),N);
        cudaDeviceSynchronize();
        quitOnCudaError();

        position_step<<<n_blocks,threads_per_block,N *2*sizeof(float3)>>>
        (thrust::raw_pointer_cast(positions.data()),thrust::raw_pointer_cast(velocities.data()),thrust::raw_pointer_cast(masses.data()),N);
        cudaDeviceSynchronize();
        quitOnCudaError();

        velocity_halfstep<<<n_blocks,threads_per_block,N * sizeof(float3) + N * sizeof(float)>>>
        (thrust::raw_pointer_cast(positions.data()),thrust::raw_pointer_cast(velocities.data()),thrust::raw_pointer_cast(masses.data()),N);
        cudaDeviceSynchronize();
        quitOnCudaError();
    }

    auto end = std::chrono::high_resolution_clock::now();

    return {std::chrono::duration_cast<seconds>(end-start), thrust::host_vector<float3>(positions)};
}