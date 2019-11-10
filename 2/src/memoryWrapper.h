#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstring>

enum class memKind
{
device,
host,
pinned
};

// Quick and dirty RAII wrappers to avoid memleaks
struct Memory
{
    Memory() = default;
    Memory(Memory const&) = delete;
    Memory(Memory &&) = delete;
    Memory& operator=(Memory const&) = delete;
    Memory& operator=(Memory &&) = delete;
    virtual ~Memory(){};

    void* _mem;
    size_t size;
    memKind kind;
};

struct DeviceMemory: public Memory
{
    DeviceMemory(size_t count)
    {
        kind = memKind::device;
        size = count*sizeof(uint8_t);
        cudaMalloc(&_mem, size);
        cudaMemset(_mem, 0, count);

    }
    virtual ~DeviceMemory(){cudaFree(_mem);}
};

struct HostMemory: public Memory
{
    HostMemory(size_t count)
    {
        kind = memKind::host;
        size = count*sizeof(uint8_t);
        _mem = malloc(size);
        memset(_mem, 0, count);
    }
    virtual ~HostMemory(){free(_mem);}
};

struct PinnedMemory: public Memory
{
    PinnedMemory(size_t count)
    {
        kind = memKind::pinned;
        size = count*sizeof(uint8_t);
        cudaMallocHost(&_mem, size);
        cudaMemset(_mem, 0, count);
    }
    virtual ~PinnedMemory(){cudaFreeHost(_mem);}
};


void cudaMemcpy(Memory const& dest, Memory const& src);