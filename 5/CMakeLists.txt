cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ex5 LANGUAGES CXX CUDA)

### imports

find_package(Boost REQUIRED)
find_package(Eigen3)
find_package(GTest)
include_directories(${Boost_INCLUDE_DIRS})

### define flags, compile options

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra ")#-fsanitize=address ")
if (${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL "8.0.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
else ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_20")
endif ()

# Cuda needs these to be happy it seems
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

### Targets and how they link

# pure c++
add_library(matrix STATIC
        matrix/matrix.h
        matrix/matrix.cpp
        )
target_include_directories(matrix PUBLIC matrix)

# cuda contaminated part
add_library(matrix_cuda SHARED
        matrix/matrix_cuda.cu
        matrix/matrix_cuda.h
        matrix/matrix.h
        matrix/matrix.cpp
        matrix/memoryWrapper.cuh
        matrix/memoryWrapper.cu
        )
target_link_libraries(matrix_cuda
        PUBLIC matrix)


add_executable(matrix_test
        test/matrix_test.cpp
        )
target_include_directories(matrix_test PUBLIC ${EIGEN_INCLUDE_DIR})

# we need pthread because of the cluster gtest_main does not work and pthread is not added automatically with gtest
target_link_libraries(matrix_test PUBLIC matrix matrix_cuda gtest pthread)

add_executable(matrix_benchmark
        runner/benchmark.cpp)
target_link_libraries(matrix_benchmark PUBLIC matrix)


# todo adapt this for every machine if it needs to be specified by hand
target_include_directories(matrix PUBLIC inc /opt/cuda/include)
target_link_directories(matrix PUBLIC /opt/cuda/lib64)
target_compile_features(matrix PUBLIC cxx_std_11)
set_target_properties(matrix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#${CMAKE_CUDA_COMPILER}









