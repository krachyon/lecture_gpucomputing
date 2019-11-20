#include <iostream>
#include "kernel.cuh"


__global__ void errorKernel(float* out)
{
    float volatile* volatile baz = nullptr;
    baz[0] = 23.f;
    *out = baz[0];
}
int reordering_repro()
{
    float* foo_d = nullptr;
    float* foo_h = new float;
    cudaMalloc(&foo_d, sizeof(float));
    errorKernel<<<1,1,20>>>(foo_d);
    quitOnCudaError();
    cudaDeviceSynchronize();

    return 324;
}

int main()
{
    std::cout << reordering_repro() << std::endl;
    std::cout <<reordering_repro() << std::endl;

}
