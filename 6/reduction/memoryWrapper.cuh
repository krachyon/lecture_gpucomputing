#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "errorHandling.cuh"
#include <vector>

enum class memKind {
    device,
    host,
    pinned
};


//forward declarations
class Memory;

template<typename T>
class DeviceMemory;

//overloads to directly use the above wrappers
void cudaMemcpy(Memory const& dest, Memory const& src);

//TODO could be also for all kinds of memory...
template<typename T>
void cudaMemcpy(T * dest, DeviceMemory<T> const& src) {
    checkCuda(cudaMemcpy(dest, src._mem, src.bytes, cudaMemcpyDeviceToHost));
}

// Quick and dirty RAII wrappers to avoid memleaks and boilerplate
struct Memory {
    Memory() = default;

    Memory(Memory const&) = delete;

    Memory(Memory&&) = delete;

    Memory& operator=(Memory const&) = delete;

    Memory& operator=(Memory&&) = delete;

    virtual ~Memory() {};

    void* _mem;
    size_t bytes;
    size_t n;
    memKind kind;
};

template<typename T>
struct DeviceMemory : public Memory {
    DeviceMemory(size_t count) {
        kind = memKind::device;
        n = count;
        bytes = count * sizeof(T);
        checkCuda(cudaMalloc(&_mem, bytes));
        // TODO in cuda-gdb memset does not do anything...
        // memset wants bytes in bytes. Set memory to some recognizable pattern for debugging help
        checkCuda(cudaMemset(_mem, 0x0f, bytes / 8));
    }

    DeviceMemory(T const * data, size_t elem_count) {
        kind = memKind::device;
        bytes = elem_count * sizeof(T);
        checkCuda(cudaMalloc(&_mem, bytes));
        checkCuda(cudaMemcpy(_mem, const_cast<T *>(data), bytes, cudaMemcpyHostToDevice));
    }

    T * mem() {
        return static_cast<T *>(_mem);
    }

    std::vector<T> to_vector()
    {
        std::vector<T> ret(n);
        cudaMemcpy(ret.data(),*this);
        return ret;
    }

    virtual ~DeviceMemory() { cudaFree(_mem); }
};

template<typename T>
struct HostMemory : public Memory {
    HostMemory(size_t count) {
        kind = memKind::host;
        n = count;
        bytes = count * sizeof(T);
        _mem = malloc(bytes);
        memset(_mem, 0, count);
    }

    virtual ~HostMemory() { free(_mem); }
};

template<typename T>
struct PinnedMemory : public Memory {
    PinnedMemory(size_t count) {
        kind = memKind::pinned;
        n = count;
        bytes = count * sizeof(T);
        checkCuda(cudaMallocHost(&_mem, bytes));
        checkCuda(cudaMemset(_mem, 0, count));
    }

    virtual ~PinnedMemory() { cudaFreeHost(_mem); }
};

