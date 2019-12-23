#include "nbody.h"
#include "utility.cuh"
#include "errorHandling.cuh"

#include <cstdint>
#include <cstddef>
#include <device_launch_parameters.h>
#include <random>

template<typename Body_t>
__device__ float3 accel(Body_t* bodies, size_t idx, size_t N)
{
    float3 dv{0,0,0};
    Body_t const& body = bodies[idx];

    for(size_t i=0; i!=N; ++i)
    {
        if(i==idx)
            continue;
        Body_t const& current_body = bodies[i];

        auto diff = current_body.pos - body.pos;
        dv += G * current_body.m * diff / (norm_pow3(diff) + eps);
    }

    return dv;
}

template<typename Body_t>
__global__ void leapfrog_aos(Body_t* bodies, size_t N, size_t iters)
{
    uint32_t tid = threadIdx.x + gridDim.x * blockIdx.x;
    if(tid>=N)
        return;

    float3 vel_half = bodies[tid].vel + accel(bodies,tid,N) * dt/2;
    bodies[tid].pos += vel_half * dt;
    vel_half += accel(bodies,tid,N) * dt/2;
    bodies[tid].vel = vel_half;
}

template<typename Body_t>
timed<thrust::host_vector<Body_t>>  run_leapfrog_aos(size_t N, size_t threads_per_block, size_t iters)
{
    thrust::device_vector<Body_t> bodies = make_random_bodies<Body_t>(N);
    size_t n_blocks = ceildiv(bodies.size(),threads_per_block);

    auto start = std::chrono::high_resolution_clock::now();
    for(auto _=0; _!=iters; ++_) {
        leapfrog_aos <<< n_blocks, threads_per_block, 0 >>> (thrust::raw_pointer_cast(bodies.data()), bodies.size(), iters);
        cudaDeviceSynchronize();
        quitOnCudaError();
    }
    auto end = std::chrono::high_resolution_clock::now();

    thrust::host_vector<Body_t> res(bodies);
    auto elapsed_seconds = std::chrono::duration_cast<seconds>(end - start);
    return {elapsed_seconds, res};
}

template<typename Body_t>
thrust::device_vector<Body_t> make_random_bodies(size_t N)
{
    thrust::device_vector<Body_t> ret;

    std::default_random_engine generator(0xdeadbeef);
    std::uniform_real_distribution<float> position_distribution(-1,1);
    std::uniform_real_distribution<float> mass_distribution(0.1f,5.f);
    auto x = std::bind(position_distribution, generator);
    auto m = std::bind(mass_distribution, generator);

    for(auto _=0;_!=N;++_) {
        ret.push_back(Body_t{{x(), x(), x()}, m(), {0.f, 0.f, 0.f}});
    }

    return ret;
}

timed<thrust::host_vector<Body>> run_leapfrog_aos(size_t N, size_t threads_per_block, size_t iters)
{
    return run_leapfrog_aos<Body>(N,threads_per_block,iters);
}

timed<thrust::host_vector<UnalignedBody>> run_leapfrog_aos_unaligned(size_t N, size_t threads_per_block, size_t iters)
{
    return run_leapfrog_aos<UnalignedBody>(N,threads_per_block,iters);
}