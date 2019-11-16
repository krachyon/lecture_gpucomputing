#include "kernel.cuh"

#include <boost/lexical_cast.hpp>
#include <boost/range/irange.hpp>
#include <iostream>

size_t const kb = 1024;
size_t const n_iter = 100;

void measure_throughput()//(int argv, char** argc)
{
    auto sizes = boost::irange(kb, 49*kb, kb);
    auto n_threads = boost::irange(1,1025,1);

    std::cout <<"# bytes, threads, direction, time(ns)" << std::endl;
    for(auto size: sizes)
        for(auto n_thread: n_threads) {

            size_t n_elements = size/sizeof(float);
            if(n_elements%n_thread == 0) {
                std::cout << size << ", "
                          << n_thread << ", "
                          << "g2s" << ", "
                          << globalMem2SharedMem_Wrapper(1, n_thread, size, n_iter).count() << std::endl;
            }

            if(n_elements%n_thread == 0) {
                std::cout << size << ", "
                          << n_thread << ", "
                          << "s2g" << ", "
                          << sharedMem2globalMem_Wrapper(1, n_thread, size, n_iter).count() << std::endl;
            }
        }
}

void measure_block()
{
    size_t const size = 16*kb;
    auto n_blocks = boost::irange(1,101,1);
    
    std::cout <<"# blocks, direction, time(ns)" << std::endl;
    std::cout <<"# size: " << size;
    for(auto n_block: n_blocks) {
        std::cout << n_block << ", "
                  << "g2s" << ", "
                  << globalMem2SharedMem_Wrapper(n_block, 512, size, n_iter).count() << std::endl;
    }

    for(auto n_block: n_blocks) {
        std::cout << n_block << ", "
                  << "s2g" << ", "
                  << sharedMem2globalMem_Wrapper(n_block, 512, size, n_iter).count() << std::endl;
    }
}


int main(int argv, char** argc)
{
    measure_block();
    //measure_throughput();
}