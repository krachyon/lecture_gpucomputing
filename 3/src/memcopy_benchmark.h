# pragma once

#include "chTimer.hpp"
#include <tuple>
#include <cuda_runtime.h>

struct Timers
{
    ChTimer D2D;
    ChTimer D2H;
    ChTimer H2D;
};

Timers memcpy_benchmark(bool optUsePinnedMemory, size_t optMemorySize, size_t optMemCpyIterations);