cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ex5 LANGUAGES CXX CUDA)

### imports

find_package(Boost REQUIRED)
find_package(Eigen3)
find_package(GTest)
include_directories(${Boost_INCLUDE_DIRS})

### define flags, compile options

set(CMAKE_CXX_STANDARD 14)
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

### Targets and how they link

# Putting
add_library(matrix SHARED
        matrix/matrix_cuda.cu
        matrix/matrix_cuda.cuh
        matrix/matrix.h
        matrix/matrix.cpp
        matrix/memoryWrapper.h
        matrix/memoryWrapper.cpp
        )

target_include_directories(matrix PUBLIC matrix)



add_executable(matrix_test
        test/matrix_test.cpp
        )
target_include_directories(matrix_test PUBLIC ${EIGEN_INCLUDE_DIR})
target_link_libraries(matrix_test PUBLIC matrix gtest)

add_executable(matrix_benchmark
        runner/benchmark.cpp)
target_link_libraries(matrix_benchmark PUBLIC matrix)


# todo adapt this for every machine if it needs to be specified by hand
target_include_directories(matrix PUBLIC inc /opt/cuda/include)
#target_link_directories(matrix PUBLIC /opt/cuda/lib64)
#${CMAKE_CUDA_COMPILER}









