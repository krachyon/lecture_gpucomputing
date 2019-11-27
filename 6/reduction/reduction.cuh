#include <vector>
#include <cstdint>
#include "reduction.h"

template<typename T>
T reduce_cuda_naive(std::vector<T>const& in, uint32_t const n_blocks);

