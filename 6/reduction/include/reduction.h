#include <vector>

float reduce_cpu(std::vector<float>const& in);
float reduce_cuda_naive(std::vector<float>& in, uint32_t const n_blocks);