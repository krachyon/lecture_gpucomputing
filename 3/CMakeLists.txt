cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ex3 LANGUAGES CXX CUDA)


#set(CMAKE_MODULE_PATH "/home/gpucomp08/software;${CMAKE_MODULE_PATH}")
#list(APPEND CMAKE_PREFIX_PATH "/home/gpucomp08/software")
#set(CMAKE_CUDA_COMPILER "/opt/cudaxx/")
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if( ${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL "8.0.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
    else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_20")
endif()

# Cuda needs these to be happy it seems
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(util STATIC
        inc/chCommandLine.h
        inc/chTimer.h
        inc/chTimer.hpp
        src/kernel.cu
        )

add_executable(main
        src/main.cpp
        src/memcopy_benchmark.cpp src/memcopy_benchmark.h)

target_include_directories(util PUBLIC inc /opt/cuda/include)
target_link_directories(util PUBLIC /opt/cuda/lib64)

target_compile_features(util PUBLIC cxx_std_11)
target_compile_features(main PUBLIC cxx_std_11)

# static linking only needed if you want to build binaries on a different machine (with different libc/libstdc++)
target_link_libraries(main PUBLIC util)# -static-libgcc -static-libstdc++