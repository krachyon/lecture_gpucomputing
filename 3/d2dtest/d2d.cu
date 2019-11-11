#include <stdio.h>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;
#define BLOCKSIZE   512

/***************/
/* COPY KERNEL */
/***************/
__global__ void copyKernel(const double * __restrict__ d_in, double * __restrict__ d_out, const int N) {

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= N) return;

    d_out[tid] = d_in[tid];

}

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

/********/
/* MAIN */
/********/
int main()
{

    const int N = 1000000;

    double* h_test = (double*) malloc(N*sizeof(double));

    for (int k = 0; k<N; k++) h_test[k] = 1.;

    double* d_in;
    checkCuda(cudaMalloc(&d_in, N*sizeof(double)));
    checkCuda(cudaMemcpy(d_in, h_test, N*sizeof(double), cudaMemcpyHostToDevice));

    double* d_out;
    checkCuda(cudaMalloc(&d_out, N*sizeof(double)));

    auto start = high_resolution_clock::now();
    checkCuda(cudaMemcpy(d_out, d_in, N*sizeof(double), cudaMemcpyDeviceToDevice));
    auto end = high_resolution_clock::now();
    printf("cudaMemcpy timing = %f [ns]\n", std::chrono::duration_cast<nanoseconds>((end-start)).count());

    start = high_resolution_clock::now();
    copyKernel <<<N/ BLOCKSIZE, BLOCKSIZE >>>(d_in, d_out, N);
    end = high_resolution_clock::now();
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());
    printf("Copy kernel timing = %f [ns]\n", std::chrono::duration_cast<nanoseconds>((end-start)).count());

    return 0;
}