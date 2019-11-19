/******************************************************************************
 *
 *Computer Engineering Group, Heidelberg University - GPU Computing Exercise 04
 *
 *                  Group : TBD
 *
 *                   File : kernel.cu
 *
 *                Purpose : Memory Operations Benchmark
 *
 ******************************************************************************/

#include <stdio.h>
//
// Test Kernel
//

__global__ void 
globalMem2SharedMem
(float* device, float* outFloat, int device_size)
{

	extern __shared__ float shMem[];
	int incr = gridDim.x * blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	int index_size=device_size/sizeof(float);
	//printf("%f %d\n",index_size, incr);
	
	if(ind==0){
		*outFloat=shMem[0];
	}
	for(int i=ind;i<index_size;i+=incr){
		//printf("%d %d \n",ind, incr);
		shMem[i]=device[i];
	}
}

void globalMem2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* device, float* outFloat) {
	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>(device, outFloat, shmSize);
}

__global__ void 
SharedMem2globalMem
(float* device, float* outFloat, int device_size)
{

	extern __shared__ float shMem[];
	int incr = gridDim.x * blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	int index_size=device_size/sizeof(float);
	//printf("%f %d\n",index_size, incr);
	
	if(ind==0){
		*outFloat=shMem[0];
	}
	for(int i=ind;i<index_size;i+=incr){
		//printf("%d %d \n",ind, incr);
		device[i]=shMem[i];
	}
}

void SharedMem2globalMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* device, float* outFloat) {
	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>(device, outFloat, shmSize);
}

__global__ void 
SharedMem2Registers
(float* outFloat, int shared_size)
{
	extern __shared__ float shMem[];
	int incr = gridDim.x * blockDim.x;
	int ind = threadIdx.x +blockIdx.x * blockDim.x;
	int index_size=shared_size/sizeof(float);
	const int reg_size=256;	
	float reg[reg_size];

	reg[0]=1;

	int k=0;
	for(int i=ind;i<index_size && k<reg_size;i+=incr){
		reg[k]=shMem[i];
		//printf("i %d k %d threadID %d blockID %d\n",i,k,threadIdx.x,blockIdx.x);
		k++;
	}

	if(ind==0){
		*outFloat=reg[0];
	}
}
void SharedMem2Registers_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* outFloat) {
	SharedMem2Registers<<< gridSize, blockSize, shmSize >>>(outFloat, shmSize);
}

__global__ void 
Registers2SharedMem
( float* outFloat, int shared_size)
{
	extern __shared__ float shMem[];
	int incr = gridDim.x * blockDim.x;
	int ind = threadIdx.x +blockIdx.x * blockDim.x;
	int index_size=shared_size/sizeof(float);
	const int reg_size=256;	
	float reg[reg_size];
	
	reg[0]=1;

	int k=0;
	for(int i=ind;i<index_size && k<reg_size;i+=incr){
		shMem[i]=reg[k];
		//printf("i %d k %d threadID %d blockID %d\n",i,k,threadIdx.x,blockIdx.x);
		k++;
	}

	if(ind==0){
		*outFloat=reg[0];
	}
}
void Registers2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* outFloat) {
	Registers2SharedMem<<< gridSize, blockSize, shmSize >>>(outFloat,shmSize);
}

__global__ void 
bankConflictsRead
(float* outFloat, int shared_size, int stride, int rep, long* clock)
{
	extern __shared__ float shMem[];
	int ind = threadIdx.x * stride;
	int index_size=shared_size/sizeof(float);
	while(ind>=index_size) ind-=index_size;
	float reg;
	if(ind==0){
		reg=0;
		*outFloat=reg;
	}
    long init=clock64();
	for(int i=0;i<rep;i++){
		reg=shMem[ind];
	}
	__syncthreads();
    long final=clock64();
	if(ind==0){
        *clock=final-init;
	}
}

void bankConflictsRead_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* outFloat, int stride, int rep, long* clock) {
	bankConflictsRead<<< gridSize, blockSize, shmSize >>>(outFloat, shmSize, stride, rep, clock);
}

