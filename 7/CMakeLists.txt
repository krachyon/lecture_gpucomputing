cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ex7 LANGUAGES CXX CUDA)

### imports

find_package(Boost REQUIRED)
find_package(Eigen3)
find_package(OpenMP)
find_package(benchmark)

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wfloat-equal -ffast-math -march=native")#-Weffc++")#-fsanitize=address ")

if (${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL "8.0.0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30 --expt-relaxed-constexpr")
else ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_20")
endif ()

# Cuda needs these to be happy it seems
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

### Targets and how they link



# cuda contaminated part
add_library(nbody_cuda SHARED
        src/nbody_naive.cu
        #src/nbody_thrust.cu
        src/nbody_shared.cu
        )

target_include_directories(nbody_cuda PUBLIC inc /opt/cuda/include)
target_link_directories(nbody_cuda PUBLIC /opt/cuda/lib64)

add_executable(nbody_benchmark
        src/benchmark.cpp)
target_link_libraries(nbody_benchmark PUBLIC nbody_cuda benchmark)

add_executable(testrunner
        src/test.cpp)
target_link_libraries(testrunner PUBLIC nbody_cuda)


# target_include_directories(reduction_test PUBLIC ${EIGEN_INCLUDE_DIR})
# we need pthread because of the cluster gtest_main does not work and pthread is not added automatically with gtest
# target_link_libraries(reduction_test PUBLIC reduction reduction_cuda tracing gtest pthread)

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 ")







