#include <stdint.h>
#include <chrono>
#include <iostream>
#include <vector>
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;


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
};

struct DeviceMemory: public Memory
{
    DeviceMemory(size_t count)
    {
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
        size = count*sizeof(uint8_t);
        cudaMallocHost(&_mem, size);
        cudaMemset(_mem, 0, count);
    }
    virtual ~PinnedMemory(){cudaFreeHost(_mem);}
};

const size_t nloop = 10;

template <typename HostMem>
size_t timeDeviceToHost(size_t count)
{
    DeviceMemory dev(count);
    HostMem host(count);

    auto start = high_resolution_clock::now();
    for(size_t _=0;_!=nloop;++_)
        cudaMemcpy(dev._mem, host._mem, host.size, cudaMemcpyDeviceToHost);
    auto end = high_resolution_clock::now();
    return std::chrono::duration_cast<nanoseconds>((end-start)/nloop).count();
}

template <typename HostMem>
size_t timeHostToDevice(size_t count)
{
    DeviceMemory dev(count);
    HostMem host(count);

    auto start = high_resolution_clock::now();
    for(size_t _=0;_!=nloop;++_)
        cudaMemcpy(host._mem, dev._mem, dev.size, cudaMemcpyHostToDevice);
    auto end = high_resolution_clock::now();
    return std::chrono::duration_cast<nanoseconds>((end-start)/nloop).count();
}

int main()
{
    size_t kb = 1024;
    size_t GB = kb*kb*kb;
    std::vector<size_t> sizes;
    for(size_t current = kb; current <= GB; current*=2)
        sizes.push_back(current);

    std::cout << "#Size; H2DPage; H2DPin; D2HPage; D2HPin" << std::endl;
    for(auto size: sizes)
    {
        std::cout << size << ";"
         << timeHostToDevice<HostMemory>(size) << ";"
         << timeHostToDevice<PinnedMemory>(size) << ";"
         << timeDeviceToHost<HostMemory>(size) << ";"
         << timeDeviceToHost<PinnedMemory>(size) << std::endl;
    }
}
