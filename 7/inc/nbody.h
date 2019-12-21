#pragma once
#include <thrust/device_vector.h>
#include <chrono>

const static float dt =      1e-6; // s
const static float G    = 6.673e-11; // (Nm^2)/(kg^2)
const static float eps = 1.f; //todo
const static size_t iters = 3000;

// todo: check performance impact of alignment
struct __align__(32) Body
{
    float3 pos;
    float m;
    float3 vel;
};

using seconds = std::chrono::duration<double>;
seconds run_leapfrog_aos(size_t N, size_t threads_per_block, size_t iters);

thrust::device_vector<Body> make_random_bodies(size_t N);