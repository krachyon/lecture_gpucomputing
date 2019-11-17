#include "kernel.cuh"

#include <boost/lexical_cast.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/join.hpp>

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

    auto bytes = boost::irange(kb,65*kb,kb);
    auto grids = boost::join(boost::irange(1,7,1), boost::irange(6,32,4));
    auto threads = std::vector<size_t>{1,2,3,4,5,6,7,8,16,32,64,128,256,512,1024};
    std::cout << "#direction, bytes, n_blocks, n_threads, time(ns)" << std::endl;
    for(auto grid: grids)
        for(auto thread: threads)
            for (auto byte: bytes) {
                if ((byte % thread) != 0 || (byte/thread) > n_registers)
                    continue;
                std::cout << "s2r, " << byte << ", " << grid << ", " << thread << ", "
                          << sharedMem2Registers_Wrapper(grid, thread, byte/thread, n_iter).count() << std::endl;
            }

    for(auto grid: grids)
        for(auto thread: threads)
            for (auto byte: bytes) {
                if ((byte % thread) != 0 || (byte/thread) >= n_registers)
                    continue;
                std::cout << "r2s, " << byte << ", " << grid << ", " << thread << ", "
                          << Registers2SharedMem_Wrapper(grid, thread, byte/thread, n_iter).count() << std::endl;
            }


}

int main(int argv, char** argc)
{
    if(argv != 2)
    {
        std::cout << "Shared Memory benchmark; Usage: ./<name> {b,r,t}" << std::endl
            << "(block, registers, throughput)" << std::endl;
    }
    else {
        auto arg = boost::lexical_cast<std::string>(argc[1]);
        if (arg == "b")
            measure_block();
        else if (arg == "t")
            measure_throughput();
        else if (arg == "r")
            measure_register();
    }
}