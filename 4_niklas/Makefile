CUDA_ROOT=/usr/lib/cuda
INC=-I./inc -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=/usr/bin/nvcc
NVCC_FLAGS=-O2 -arch=sm_50 --compiler-options "-O2 -Wall -Wextra"

.PHONY: build
build: ./bin/memCpy

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/memCpy: ./src/main.cpp ./src/kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)
