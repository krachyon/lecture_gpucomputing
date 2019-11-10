#include <stdint.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cassert>
#include "memoryWrapper.h"

using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;



template<typename SRC, typename DST>
size_t timeMemcpy(size_t bytes, size_t nloop)
{
    SRC src(bytes);
    DST dst(bytes);

    auto start = high_resolution_clock::now();
    for(size_t _=0;_!=nloop;++_)
        cudaMemcpy(dst, src);
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

    const size_t nloop = 10;

    std::cout << "#Size; H2DPage; H2DPin; D2HPage; D2HPin; D2D" << std::endl;
    for(auto size: sizes)
    {
        std::cout << size << ";"
         << timeMemcpy<HostMemory,DeviceMemory>(size, nloop) << ";"
         << timeMemcpy<PinnedMemory,DeviceMemory>(size, nloop) << ";"
         << timeMemcpy<DeviceMemory,HostMemory>(size, nloop)  << ";"
         << timeMemcpy<DeviceMemory,PinnedMemory>(size, nloop)  << ";"
         << timeMemcpy<DeviceMemory,DeviceMemory>(size, nloop)  << std::endl;
    }
}
