/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <assert.h>
#include <vector>
#include <utility>
#include <iostream>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
      fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
      assert(result == cudaSuccess);
    }
#endif
    return result;
}

std::pair<float,float> profileCopies(float* h_a,
        float* h_b,
        float* d,
        unsigned int n)
{
    unsigned int bytes = n*sizeof(float);

    // events for timing
    cudaEvent_t startEvent, stopEvent;

    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
	
    //times are in miliseconds
    float timeh2d, timed2h;
    checkCuda(cudaEventElapsedTime(&timeh2d, startEvent, stopEvent));

    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));

    checkCuda(cudaEventElapsedTime(&timed2h, startEvent, stopEvent));

    for (int i = 0; i<n; ++i) {
        if (h_a[i]!=h_b[i]) {
            printf("*** transfers failed ***");
            break;
        }
    }

    // clean up events
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));
    return std::make_pair(timeh2d,timed2h);
}

int main()
{
    size_t kb = 1024;
    size_t GB = kb*kb*kb;
    std::vector<size_t> sizes;
    for (size_t current = kb; current<=GB; current *= 2)
        sizes.push_back(current/sizeof(float));

    std::cout << "#Size; H2DPage; H2DPin; D2HPage; D2HPin" << std::endl;

    for (unsigned int nElements: sizes) {
        const unsigned int bytes = nElements*sizeof(float);

        // host arrays
        float* h_aPageable, * h_bPageable;
        float* h_aPinned, * h_bPinned;

        // device array
        float* d_a;

        // allocate and initialize
        h_aPageable = (float*) malloc(bytes);                    // host pageable
        h_bPageable = (float*) malloc(bytes);                    // host pageable
        checkCuda(cudaMallocHost((void**) &h_aPinned, bytes)); // host pinned
        checkCuda(cudaMallocHost((void**) &h_bPinned, bytes)); // host pinned
        checkCuda(cudaMalloc((void**) &d_a, bytes));           // device

        for (int i = 0; i<nElements; ++i) h_aPageable[i] = i;
        memcpy(h_aPinned, h_aPageable, bytes);
        memset(h_bPageable, 0, bytes);
        memset(h_bPinned, 0, bytes);

        // output device info and transfer size
        cudaDeviceProp prop;
        checkCuda(cudaGetDeviceProperties(&prop, 0));


        // perform copies and report bandwidth
        auto tPage = profileCopies(h_aPageable, h_bPageable, d_a, nElements);
        auto tPinned = profileCopies(h_aPinned, h_bPinned, d_a, nElements);

        std::cout << bytes << ";"
                  << tPage.first << ";"
                  << tPinned.first << ";"
                  << tPage.second << ";"
                  << tPinned.second << std::endl;


        // cleanup
        cudaFree(d_a);
        cudaFreeHost(h_aPinned);
        cudaFreeHost(h_bPinned);
        free(h_aPageable);
        free(h_bPageable);
    }
    return 0;
}
