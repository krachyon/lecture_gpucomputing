#include <vector>
#include <iostream>
#include <chrono>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::nanoseconds;

const size_t maxWait = 10000;
const size_t nIter = 1000;

__device__ clock_t diff;


__global__ void SleepKernel(clock_t ticks, bool writeDiff)
{
    clock_t start = clock();
    while (clock()-start < ticks){}
    if(writeDiff)
        diff = clock() - start;
}

void warmup()
{
    for(size_t _ = 0; _ != nIter*10; ++_)
    {
        SleepKernel<<<1,1>>>(100,false);
    }
}

int main()
{
    warmup();
    cout << "# waitticks ; elapsed time in nanoseconds" << std::endl;
    for(size_t waitticks = 0; waitticks!=maxWait; waitticks+=10) {
        auto start = high_resolution_clock::now();
        for(size_t _ = 0; _ != nIter; ++_)
        {
            SleepKernel<<<1,1>>>(waitticks,false);
        }
        auto end = high_resolution_clock::now();

        auto ns = std::chrono::duration_cast<nanoseconds>(end - start)/nIter;
        cout << waitticks << ";" << ns.count() << std::endl;
    }
}
