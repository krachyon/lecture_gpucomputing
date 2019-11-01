#include <cstdint>
#include <chrono>
#include <iostream>
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;

// Quick and dirty RAII wrappers to avoid memleaks
struct Memory
{
    Memory() = delete;
    Memory(Memory const&) = delete;
    Memory(Memory &&) = delete;
    & operator=(Memory const&) = delete;
    & operator=(Memory &&) = delete;
    virtual ~Memory();


    void* _mem;
    size_t size;
};

struct DeviceMemory: public Memory
{
    template<typename T>
    DeviceMemory(size_t count)
    {
        size = count*sizeof(T);
        cudaMalloc(&_mem, size);
    }
    ~DeviceMemory(){cudaFree(_mem);}
};

struct HostMemory: public Memory
{
    template<typename T>
    DeviceMemory(size_t count)
    {
        size = count*sizeof(T);
        _mem = malloc(size);
    }
    ~DeviceMemory(){free(_mem);}
};

struct PinnedMemory: public Memory
{
    template<typename T>
    DeviceMemory(size_t count)
    {
        size = count*sizeof(T);
        cudaMallocHost(&_mem, size);
    }
    ~DeviceMemory(){cudaFreeHost(_mem);}
};

template <typename HostMem>
size_t timeDeviceToHost(size_t count)
{
    DeviceMemory<uint8_t>(count) dev;
    HostMem<uint8_t>(count) host;

    auto start = high_resolution_clock::now();
    cudaMemcpy(dev._mem, host._mem, host.size, cudaMemcpyDeviceToHost);
    auto end = high_resolution_clock::now();
    return std::chrono::duration_cast<nanoseconds>(end-start).count();
}

template <typename HostMem>
size_t timeHostToDevice(size_t count)
{
    DeviceMemory<uint8_t>(count) dev;
    HostMem<uint8_t>(count) host;

    auto start = high_resolution_clock::now();
    cudaMemcpy(host._mem, dev._mem, dev.size, cudaMemcpyHostToDevice);
    auto end = high_resolution_clock::now();
    return std::chrono::duration_cast<nanoseconds>(end-start).count();
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