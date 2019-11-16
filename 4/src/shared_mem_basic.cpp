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
            if(n_elements%n_thread == 0) { //only try computations where the addressing scheme works cleanly
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
    size_t const size = 16*kb; // arbitrarily chosen
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


void measure_register()
{
    // I'm assuming that copying to/from registers is completely thread local it doesn't really make sense
    // to launch a kernel with a bunch of threads unless we want to observe the effect of register spilling but
    // that seems where the question is headed?
    // TODO Probably should treat the shared memory as a working set and we could let one thread do everything or split
    // it up among threads

    auto n_elements = boost::irange(1,64,1);
    std::cout << "# bytes,, time(ns)";
    for (auto n: n_elements)
        sharedMem2Registers_Wrapper(1,1,n,n_iter);

}

int main(int argv, char** argc)
{
    //todo look at command line to see what tests to launch
    measure_block();
    //measure_throughput();
}