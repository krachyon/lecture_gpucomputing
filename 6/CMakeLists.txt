cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ex5 LANGUAGES CXX CUDA)

### imports

find_package(Boost REQUIRED)
find_package(Eigen3)
#find_package(GTest)
find_package(OpenMP)

include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.10.0
)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

include_directories(${Boost_INCLUDE_DIRS})

### define flags, compile options

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# This allows you to debug inside kernels with cuda-gdb
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -g -G")

# Weffc++ can be helpful but has a bunch of false positives. -fsanitize=<x> is great to find memory/threading issues
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wfloat-equal ")#-Weffc++")#-fsanitize=address ")
if (${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL "8.0.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30 --expt-relaxed-constexpr")
else ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_20")
endif ()

# Cuda needs these to be happy it seems
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

### Targets and how they link

# pure c++
add_library(tracing STATIC
        reduction/tracing.cpp
        reduction/include/tracing.h
        )

add_library(reduction STATIC
            reduction/include/reduction.h
            reduction/reduction.cpp
            reduction/reduction_cpp.h )
target_include_directories(reduction PUBLIC reduction/include)
target_link_libraries(reduction PUBLIC tracing)

# cuda contaminated part
add_library(reduction_cuda SHARED
        reduction/include/reduction.h
        reduction/errorHandling.cuh
        reduction/memoryWrapper.cuh
        reduction/memoryWrapper.cu
        reduction/reduction_naive.cu
        reduction/reduction_shared.cu
        reduction/reduction.cuh
        reduction/reduction_optimized.cu
        )
target_include_directories(reduction_cuda PUBLIC reduction/include)
target_include_directories(reduction_cuda PUBLIC include /opt/cuda/include)
target_link_directories(reduction_cuda PUBLIC /opt/cuda/lib64)
target_link_libraries(reduction_cuda PUBLIC tracing)


add_executable(reduction_test
        reduction_test/main.cpp
        reduction_test/test_cuda.cpp reduction_test/test_trace.cpp)
target_include_directories(reduction_test PUBLIC ${EIGEN_INCLUDE_DIR})
# we need pthread because of the cluster gtest_main does not work and pthread is not added automatically with gtest
target_link_libraries(reduction_test PUBLIC reduction reduction_cuda tracing gtest pthread)


add_executable(reduction_benchmark
        reduction_benchmark/benchmark.cpp)
target_link_libraries(reduction_benchmark PUBLIC reduction reduction_cuda )

add_executable(reduction_cpu
        CPU/GSR.cpp)

target_link_libraries(reduction_cpu PUBLIC)















