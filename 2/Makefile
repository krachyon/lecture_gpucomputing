CUDA_ROOT=/opt/cuda-7.0
INC=-I./inc -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=$(CUDA_ROOT)/bin/nvcc
NVCC_FLAGS=-O2 -arch=sm_20 -std=c++11 --compiler-options "-O2 -Wall -Wextra"

.PHONY: build
build: ./bin/nullKernelAsync
build: ./bin/wait
build: ./bin/memory

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/nullKernelAsync: ./src/nullKernelAsync.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

./bin/wait: ./src/wait.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

./bin/memory: ./src/memory.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)
