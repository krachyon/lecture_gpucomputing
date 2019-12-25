#pragma once
#include <thrust/device_vector.h>
#include <chrono>
#include <utility>

const static float dt =      1e-6; // s
const static float G    = 6.673e-11; // (Nm^2)/(kg^2)
const static float eps = 1.f; //todo
const static size_t iters = 10;

struct __align__(32) Body
{
    float3 pos;
    float m;
    float3 vel;
};

struct UnalignedBody
{
    float3 pos;
    float m;
    float3 vel;
};

using seconds = std::chrono::duration<double>;
template<typename T>
using timed = std::pair<seconds,T>;

timed<thrust::host_vector<Body>> run_leapfrog_aos(size_t N, size_t threads_per_block, size_t iters);
timed<thrust::host_vector<UnalignedBody>> run_leapfrog_aos_unaligned(size_t N, size_t threads_per_block, size_t iters);
timed<thrust::host_vector<float3>> run_leapfrog_soa(size_t N, size_t threads_per_block, size_t iters);

template<typename Body_t>
thrust::device_vector<Body_t> make_random_bodies(size_t N);