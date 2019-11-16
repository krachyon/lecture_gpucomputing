#include "kernel.cuh"

#include <boost/lexical_cast.hpp>
#include <boost/range/irange.hpp>
#include <iostream>

int main()//(int argv, char** argc)
{
    size_t kb = 1024;
    auto sizes = boost::irange(kb, 48*kb, kb);
    auto n_threads = boost::irange(1,1024,1);

    std::cout <<"# bytes, threads, time(ns)";
    for(auto size: sizes)
        for(auto n_thread: n_threads)
            std::cout << size << ", "
            << n_threads << ", "
            ", " << globalMem2SharedMem_Wrapper(1,n_thread,size).count() << std::endl;
}