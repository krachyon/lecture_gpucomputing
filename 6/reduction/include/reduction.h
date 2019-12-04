#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>


float reduce_cpu(std::vector<float>const& in);

float reduce_cuda_naive(std::vector<float>& in, uint32_t const n_blocks, size_t iter=1);
double reduce_cuda_naive(std::vector<double>& in, uint32_t const n_blocks, size_t iter=1);
uint32_t reduce_cuda_naive(std::vector<uint32_t>& in, uint32_t const n_blocks, size_t iter=1);
int32_t reduce_cuda_naive(std::vector<int32_t>& in, uint32_t const n_blocks, size_t iter=1);
int16_t reduce_cuda_naive(std::vector<int16_t>& in, uint32_t const n_blocks, size_t iter=1);
uint16_t reduce_cuda_naive(std::vector<uint16_t>& in, uint32_t const n_blocks, size_t iter=1);

float reduce_cuda_shared(std::vector<float>& in, uint32_t const n_blocks, size_t iter=1);
double reduce_cuda_shared(std::vector<double>& in, uint32_t const n_blocks, size_t iter=1);
uint32_t reduce_cuda_shared(std::vector<uint32_t>& in, uint32_t const n_blocks, size_t iter=1);
int32_t reduce_cuda_shared(std::vector<int32_t>& in, uint32_t const n_blocks, size_t iter=1);
int16_t reduce_cuda_shared(std::vector<int16_t>& in, uint32_t const n_blocks, size_t iter=1);
uint16_t reduce_cuda_shared(std::vector<uint16_t>& in, uint32_t const n_blocks, size_t iter=1);

float thrust_reduce(std::vector<float>const& in, size_t iter=1);
