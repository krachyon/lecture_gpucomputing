#include <chrono>
#include <cuda_runtime.h>


__global__ void globalMem2SharedMem(float* d_mem, size_t n_elements);
std::chrono::nanoseconds globalMem2SharedMem_Wrapper(size_t gridSize, size_t blockSize, size_t bytes, size_t n_iter);
__global__ void SharedMem2globalMem (float* d_mem, size_t n_elements);
std::chrono::nanoseconds sharedMem2globalMem_Wrapper(size_t gridSize, size_t blockSize, size_t bytes, size_t n_iter);
__global__ void SharedMem2Registers(size_t n_elements, size_t chunk_size, float volatile* volatile dummy);
std::chrono::nanoseconds sharedMem2Registers_Wrapper(size_t gridSize, size_t blockSize, size_t bytes, size_t n_iters);
__global__ void Registers2SharedMem(size_t chunk_size, float volatile* volatile dummy);
std::chrono::nanoseconds Registers2SharedMem_Wrapper(size_t gridSize, size_t blockSize, size_t bytes, size_t n_iters);

// No way this fits into registers if we actually use it.
// TODO Will this only spill out if we use too much or just be global mem in the first place?
constexpr size_t n_registers = 1024;


