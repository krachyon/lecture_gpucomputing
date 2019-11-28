#pragma once
#include <vector>
#include <cstdint>
#include "reduction.h"

template<typename T>
T reduce_cuda_naive(std::vector<T>const& in, uint32_t const n_blocks);

//TODO maybe replace log2/ceil with integer math
template<typename T>
__device__ __host__ bool is_power_of_2(T in)
{
    return log2((double)in) == floor(log2((double)in));
};

