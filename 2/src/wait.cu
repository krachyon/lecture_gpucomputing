#include <vector>
#include <iostream>
#include <chrono>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

const size_t maxWait = 1000;
const size_t nIter = 10000;

__global__ void SleepKernel(clock_t ticks, bool writeDiff)
{
    clock_t start = clock();
    while (clock()-start < ticks){}
    if(writeDiff)
        __device__ diff = clock() - start;
}

int main()
{
    for(size_t i = 0; i!=maxWait; ++i) {
        auto start = high_resolution_clock::now();
        for(size_t _ = 0; _ != nIter; ++_)
        {
            SleepKernel<<<1,1>>>(i,false);
        }
        auto end = high_resolution_clock::now();

        microseconds us = (end - start)/nIter;
        cout << waitticks << ";" << us;
    }
}