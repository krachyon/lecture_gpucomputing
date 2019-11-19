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
	int incr = gridDim.x * blockDim.x;//increment of the shared/global memory index
	int ind = threadIdx.x + blockIdx.x * blockDim.x;//initial index of every thread in the shared/global memory
	int index_size=device_size/sizeof(float);//number of floats that fit into the shared memory
	
	if(ind==0){
		*outFloat=shMem[0];//assign shared memory value to output variable to prohibit compiler optimizations 
	}
	for(int i=ind;i<index_size;i+=incr){
		shMem[i]=device[i];//copy global -> shared
	}
}

void globalMem2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* device, float* outFloat) {
	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>(device, outFloat, shmSize);
}

__global__ void 
SharedMem2globalMem//analogous to global2shared
(float* device, float* outFloat, int device_size)
{

	extern __shared__ float shMem[];
	int incr = gridDim.x * blockDim.x;
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	int index_size=device_size/sizeof(float);
	
	if(ind==0){
		*outFloat=shMem[0];
	}
	for(int i=ind;i<index_size;i+=incr){
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
	int incr = gridDim.x * blockDim.x;//increment of the shared memory index
	int ind = threadIdx.x +blockIdx.x * blockDim.x;//initial index of every thread in the shared memory
	int index_size=shared_size/sizeof(float);//number of floats that fit into the shared memory
	const int reg_size=256;//constant register size to make sure it really is saved into a thread-local register 
	float reg[reg_size];

	reg[0]=1;

	int k=0;//index within the register
	for(int i=ind;i<index_size && k<reg_size;i+=incr){//make sure that both the register/shared mem index stay within bounds
		reg[k]=shMem[i];//copy shared->register
		k++;//increment register index
	}

	if(ind==0){
		*outFloat=reg[0];//assign register memory value to output variable to prohibit compiler optimizations 
	}
}
void SharedMem2Registers_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* outFloat) {
	SharedMem2Registers<<< gridSize, blockSize, shmSize >>>(outFloat, shmSize);
}

__global__ void 
Registers2SharedMem//analogous to shared2register
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
	int ind = threadIdx.x * stride;//determine index as thread index times stride
	int index_size=shared_size/sizeof(float);//number of floats in shared
	while(ind>=index_size) ind-=index_size;//make sure only valid addresses are probed
	float reg;//single float register
	if(ind==0){
		reg=0;
		*outFloat=reg;//assign register memory value to output variable to prohibit compiler optimizations 
	}
    long init=clock64();//take initial time
	for(int i=0;i<rep;i++){//repeat rep times for stability
		reg=shMem[ind];//load the same element into reg repeatedly
	}
	__syncthreads();//wait for all threads to finish reading
    long final=clock64();//take final time
	if(ind==0){
        *clock=final-init;//save time differnce to global memory
	}
}

void bankConflictsRead_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* outFloat, int stride, int rep, long* clock) {
	bankConflictsRead<<< gridSize, blockSize, shmSize >>>(outFloat, shmSize, stride, rep, clock);
}

