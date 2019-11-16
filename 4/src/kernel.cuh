#include <chrono>
#include <cuda_runtime.h>


__global__ void globalMem2SharedMem(float* d_mem, size_t n_elements);
std::chrono::nanoseconds globalMem2SharedMem_Wrapper(size_t gridSize, size_t blockSize, size_t bytes, size_t n_iter);
__global__ void SharedMem2globalMem (float* d_mem, size_t n_elements);
std::chrono::nanoseconds sharedMem2globalMem_Wrapper(size_t gridSize, size_t blockSize, size_t bytes, size_t n_iter);
__global__ void SharedMem2Registers(size_t n_elements, size_t chunk_size, float& dummy);
std::chrono::nanoseconds sharedMem2Registers_Wrapper(size_t gridSize, size_t blockSize, size_t bytes, size_t n_iters);



