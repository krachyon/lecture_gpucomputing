cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ex3 LANGUAGES CXX CUDA)


#set(CMAKE_MODULE_PATH "/home/gpucomp08/software;${CMAKE_MODULE_PATH}")
#list(APPEND CMAKE_PREFIX_PATH "/home/gpucomp08/software")
#set(CMAKE_CUDA_COMPILER "/opt/cudaxx/")
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if( ${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL "8.0.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CUDA_STANDARD 14)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_20")
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CUDA_STANDARD 11)
endif()

# Cuda needs these to be happy it seems
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(wrapper
        src/memoryWrapper.h src/memoryWrapper.cpp)

add_executable(memory
        src/memory.cu
)

target_link_libraries(memory PUBLIC wrapper)

target_include_directories(wrapper PUBLIC inc /opt/cuda/include)
target_link_directories(wrapper PUBLIC /opt/cuda/lib64)

