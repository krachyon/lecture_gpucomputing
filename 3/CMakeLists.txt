cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ex3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -arch=sm_35")
set(CMAKE_CUDA)

add_library(util STATIC
        inc/chCommandLine.h
        inc/chTimer.h
        inc/chTimer.hpp
        src/kernel.cu
        )


add_executable(main
        src/main.cpp
)



target_include_directories(util PUBLIC inc /opt/cuda/include)
target_link_directories(util PUBLIC /opt/cuda/lib)

target_compile_features(util PUBLIC cxx_std_14)
target_compile_features(main PUBLIC cxx_std_14)

target_link_libraries(main PUBLIC util)