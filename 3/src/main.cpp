/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 03
 *
 *                           Group : TBD
 *
 *                            File : main.cu
 *
 *                         Purpose : Memory Operations Benchmark
 *
 *************************************************************************************************/

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "chCommandLine.h"
#include "chTimer.hpp"
#include <cstdio>
#include <cassert>
#include "memcopy_benchmark.h"


const static int DEFAULT_MEM_SIZE = 10*1024*1024; // 10 MB
const static int DEFAULT_NUM_ITERATIONS = 1000;
const static int DEFAULT_BLOCK_DIM = 128;
const static int DEFAULT_GRID_DIM = 16;

//
// Function Prototypes
//
void printHelp(char*);

//
// Kernel Wrappers
//


extern void  globalMemCoalescedKernel_Wrapper(dim3 gridDim, dim3 blockDim, float* in, float* out, size_t n_elements);
extern void globalMemStrideKernel_Wrapper(dim3 gridDim, dim3 blockDim, float* in, float* out, size_t stride);
extern void globalMemOffsetKernel_Wrapper(dim3 gridDim, dim3 blockDim, float* in, float* out, size_t offset);

//
// Main
//

int main(int argc, char* argv[])
{
    // Show Help
    bool optShowHelp = chCommandLineGetBool("h", argc, argv);
    if (!optShowHelp)
        optShowHelp = chCommandLineGetBool("help", argc, argv);

    if (optShowHelp) {
        printHelp(argv[0]);
        exit(0);
    }

    ChTimer kernelTimer;

    //
    // Get kernel launch parameters and configuration
    //
    size_t optNumIterations = 0,
            optBlockSize = 0,
            optGridSize = 0;

    // Number of Iterations
    chCommandLineGet<size_t>(&optNumIterations, "i", argc, argv);
    chCommandLineGet<size_t>(&optNumIterations, "iterations", argc, argv);
    optNumIterations = (optNumIterations!=0) ? optNumIterations : DEFAULT_NUM_ITERATIONS;

    // Block Dimension / Threads per Block
    chCommandLineGet<size_t>(&optBlockSize, "t", argc, argv);
    chCommandLineGet<size_t>(&optBlockSize, "threads-per-block", argc, argv);
    optBlockSize = optBlockSize!=0 ? optBlockSize : DEFAULT_BLOCK_DIM;

    if (optBlockSize>1024) {
        std::cout << "***" << std::endl
                  << "*** Error - The number of threads per block is too big"
                  << std::endl
                  << "***" << std::endl;

        exit(-1);
    }

    // Grid Dimension
    chCommandLineGet<size_t>(&optGridSize, "g", argc, argv);
    chCommandLineGet<size_t>(&optGridSize, "grid-dim", argc, argv);
    optGridSize = optGridSize!=0 ? optGridSize : DEFAULT_GRID_DIM;

    // Sync after each Kernel Launch
    bool optSynchronizeKernelLaunch = chCommandLineGetBool("y", argc, argv);
    if (!optSynchronizeKernelLaunch)
        optSynchronizeKernelLaunch = chCommandLineGetBool("synchronize-kernel", argc, argv);

    dim3 grid_dim = dim3(optGridSize);
    dim3 block_dim = dim3(optBlockSize);

    int optStride = 1; //default stride for global-stride test
    chCommandLineGet<int>(&optStride, "stride", argc, argv);

    int optOffset = 0; //default offset for global-stride test
    chCommandLineGet<int>(&optOffset, "offset", argc, argv);

    // Allocate Memory
    // optStride resp. optOffset are NOT taken into account. Just make sure to allocate enough for these ops.
    size_t optMemorySize = 0;

    // determine memory size from parameters
    chCommandLineGet<size_t>(&optMemorySize, "s", argc, argv);
    if(optMemorySize == 0)
        chCommandLineGet<size_t>(&optMemorySize, "size", argc, argv);
    optMemorySize = optMemorySize!=0 ? optMemorySize : DEFAULT_MEM_SIZE;
    size_t n_elements = optMemorySize / sizeof(float);

    bool optUsePinnedMemory = chCommandLineGetBool("p", argc, argv);
    if (!optUsePinnedMemory)
        optUsePinnedMemory = chCommandLineGetBool("pinned-memory", argc, argv);

    int optMemCpyIterations = 0;
    chCommandLineGet<int>(&optMemCpyIterations, "im", argc, argv);
    chCommandLineGet<int>(&optMemCpyIterations, "memory-copy-iterations", argc, argv);
    optMemCpyIterations = optMemCpyIterations!=0 ? optMemCpyIterations : 1;


    // Parameter gathering done, let's do something useful

    //Memcopy test here
    Timers memcopy_timers{ChTimer(), ChTimer(), ChTimer()};
    bool optMemcopy = chCommandLineGetBool("memcpy", argc, argv);
    if (optMemcopy) {
        memcopy_timers = memcpy_benchmark(optUsePinnedMemory, optMemorySize, optMemCpyIterations);
    }

    //
    // Global Memory Tests
    //

    float* d_memoryA;
    float* d_memoryB;
    cudaMalloc(&d_memoryA, optMemorySize);
    cudaMalloc(&d_memoryB, optMemorySize);

    kernelTimer.start();
    for (size_t i = 0; i<optNumIterations; i++) {
        //
        // Launch Kernel
        //
        if (chCommandLineGetBool("global-coalesced", argc, argv)) {
            globalMemCoalescedKernel_Wrapper(grid_dim, block_dim, d_memoryA, d_memoryB, n_elements);
        }
        else if (chCommandLineGetBool("global-stride", argc, argv)) {
            globalMemStrideKernel_Wrapper(grid_dim, block_dim, d_memoryA, d_memoryB, optStride);
        }
        else if (chCommandLineGetBool("global-offset", argc, argv)) {
            globalMemOffsetKernel_Wrapper(grid_dim, block_dim, d_memoryA, d_memoryB, optOffset);
        }
        else {
            break;
        }

        if (optSynchronizeKernelLaunch) { // Synchronize after each kernel launch
            cudaDeviceSynchronize();

            //
            // Check for Errors
            //
            cudaError_t cudaError = cudaGetLastError();
            if (cudaError!=cudaSuccess) {
                std::cout << "***" << std::endl
                          << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
                          << std::endl
                          << "***" << std::endl;

                return -1;
            }
        }
    }

    // Mandatory synchronize after all kernel launches
    cudaDeviceSynchronize();
    kernelTimer.stop();

    //
    // Check for Errors
    //
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError!=cudaSuccess) {
        std::cout << "***" << std::endl
                  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
                  << std::endl
                  << "***" << std::endl;

        return -1;
    }

    // Print Measurement Results
    if(optMemcopy) {
        std::cout << "#size,pinned , H2D(μs),H2D(GBs), D2H(μs),D2H(GBs), D2D(), D2D(GBs)" << std::endl
                  << optMemorySize << ", "
                  << optUsePinnedMemory << ", "
                  << 1e6*memcopy_timers.H2D.getTime() << 1e-9*memcopy_timers.H2D.getBandwidth(optMemorySize, optMemCpyIterations) << ", "
                  << 1e6*memcopy_timers.D2H.getTime() << 1e-9*memcopy_timers.D2H.getBandwidth(optMemorySize, optMemCpyIterations) << ", "
                  << 1e6*memcopy_timers.D2D.getTime() << 1e-9*memcopy_timers.D2D.getBandwidth(optMemorySize, optMemCpyIterations) << ", "
                  << std::endl;
    }
    else {
        std::string type = "invalid";
        if(chCommandLineGetBool("global-coalesced", argc, argv)) type = "global-coalesced";
        if(chCommandLineGetBool("global-stride", argc, argv)) type = "global-stride";
        if(chCommandLineGetBool("global-offset", argc, argv)) type = "global-offset";
        
        size_t stride_offset = 0;
        if(optStride != 1) 
            stride_offset = optStride;
        else if(optOffset != 0)
            stride_offset = optOffset;

        std::cout << "#type, size, stride_offset, gDim, bDim, time(μs), bandwidth(GB/s)" << std::endl;        

        std::cout << type <<", "
        << optMemorySize  <<", "
        << stride_offset <<", "
        << grid_dim.x <<", "
        << block_dim.x <<", "
        << kernelTimer.getTime(optNumIterations) <<", "
        << optMemorySize/kernelTimer.getTime(optNumIterations)/(1E09)
        << std::endl;

    }
    return 0;
}

void
printHelp(char* programName)
{
    std::cout
            << "Usage: " << std::endl
            << "  " << programName << " [-p] [-s <memory_size>] [-i <num_iterations>]" << std::endl
            << "                [-t <threads_per_block>] [-g <blocks_per_grid]" << std::endl
            << "                [-m <memory-copy-iterations>] [-y] [-stride <stride>] [-offset <offset>]" << std::endl
            << "  --memcpy" << std::endl
            << "    Run memcopy benchmark" << std::endl
            << "  --global-{coalesced|stride|offset}" << std::endl
            << "    Run kernel analyzing global memory performance" << std::endl
            << "  -p|--pinned-memory" << std::endl
            << "    Use pinned Memory instead of pageable memory" << std::endl
            << "  -y|--synchronize-kernel" << std::endl
            << "    Synchronize device after each kernel launch" << std::endl
            << "  -s <memory_size>|--size <memory_size>" << std::endl
            << "    The amount of memory to allocate. Make sure that you make it divisible by 't' and to"
            << "    Allocate enough t*stride and t + offset for the respective operation" << std::endl
            << "  -t <threads_per_block>|--threads-per-block <threads_per_block>" << std::endl
            << "    The number of threads per block" << std::endl
            << "  -g <blocks_per_grid>|--grid-dim <blocks_per_grid>" << std::endl
            << "     The number of blocks per grid" << std::endl
            << "  -i <num_iterations>|--iterations <num_iterations>" << std::endl
            << "     The number of iterations to launch the kernel" << std::endl
            << "  --im <memory-copy-iterations>|--memory-iterations <memory-copy-iterations>" << std::endl
            << "     The number of times the memory will be copied. Use this to get more stable results." << std::endl
            << "  --stride <stride>" << std::endl
            << "     Stride parameter for global-stride test. Not that size parameter is ignored then." << std::endl
            << "  --offset <offset>" << std::endl
            << "     Offset parameter for global-offset test. Not that size parameter is ignored then." << std::endl
            << "" << std::endl;
}
