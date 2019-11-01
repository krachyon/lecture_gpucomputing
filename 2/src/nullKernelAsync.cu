/*
 *
 * nullKernelAsync.cu
 *
 * Microbenchmark for throughput of asynchronous kernel launch.
 *
 * Build with: nvcc -I ../chLib <options> nullKernelAsync.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 
 *
 * 1. Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright 
 *    notice, this list of conditions and the following disclaimer in 
 *    the documentation and/or other materials provided with the 
 *    distribution. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdio.h>
#include <vector>
#include <iostream>
using std::cout;

#include "chTimer.h"

__global__
void
NullKernel()
{
}

__global__ void NearlyNullKernel()
{
    //void* mem = cudaMalloc ( 1 * sizeof(float));
    //cudaFree(mem);
}

const size_t cIterations = 1000000;
double us(chTimerTimestamp start, chTimerTimestamp stop)
{
    double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
    return microseconds / (float) cIterations;
}

int
main()
{
    chTimerTimestamp start, stop;
    cout << "# block_count; thread_count; async; time/usec\n";

    std::vector<size_t> block_counts{1,2,4,8,16,32,64,128,512,1024};
    std::vector<size_t> thread_counts{1,2,4,8,16,32,64,128,512,1024};

    //measure Async
    for(auto block: block_counts)
    for(auto thread: thread_counts)
    {
        chTimerGetTime( &start );
        for ( size_t i = 0; i < cIterations; i++ ) 
	{
            NullKernel<<<block,thread>>>();
        }
        cudaThreadSynchronize();
        chTimerGetTime( &stop );
	
	cout << block << ";" << thread << ";" << us(start,stop) << std::endl;
    }

    //measure synchro
    for(auto block: block_counts)
    for(auto thread: thread_counts)
    {
        chTimerGetTime( &start );
        for ( size_t i = 0; i < cIterations; i++ ) 
	{
            NullKernel<<<block,thread>>>();
            cudaThreadSynchronize();
        }
        chTimerGetTime( &stop );
	
	cout << block << ";" << thread << ";" << us(start,stop) << std::endl;
    }

    return 0;
}
