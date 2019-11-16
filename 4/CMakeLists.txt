cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ex4 LANGUAGES CXX CUDA)

find_package(Boost REQUIRED)
find_package(Eigen3)
find_package(GTest)
include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")# -fsanitize=address ")
if( ${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL "8.0.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_20")
endif()

# Cuda needs these to be happy it seems
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(inc)

add_library(matrix
        src/matrix/matrix.h
        src/matrix/matrix.cpp
)

add_executable(matrix_test
        src/matrix/matrix.h
        src/matrix/matrix_test.cpp
        )
add_executable(matrix_benchmark
        src/matrix/benchmark.cpp
        src/matrix/benchmark.cpp)


target_include_directories(matrix_test PUBLIC ${EIGEN_INCLUDE_DIR})
target_link_libraries(matrix_test PUBLIC gtest pthread)

target_link_libraries(matrix_benchmark)


add_library(util
        inc/chCommandLine.h
        inc/chTimer.h
        inc/chTimer.hpp
        inc/errorHandling.h
        src/kernel.cuh
        src/kernel.cu
        )

add_executable(main
        src/main.cpp)

add_executable(shared_mem_basic
        src/shared_mem_basic.cpp
        )

target_include_directories(util PUBLIC inc /opt/cuda/include)
target_link_directories(util PUBLIC /opt/cuda/lib64)

target_compile_features(util PUBLIC cxx_std_11)
target_compile_features(main PUBLIC cxx_std_11)

# static linking only needed if you want to build binaries on a different machine (with different libc/libstdc++)
target_link_libraries(main PUBLIC util)# -static-libgcc -static-libstdc++
target_link_libraries(shared_mem_basic PUBLIC util)