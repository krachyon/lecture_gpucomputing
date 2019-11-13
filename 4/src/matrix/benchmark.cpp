#include "matrix.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <boost/range/irange.hpp>
#include <boost/lexical_cast.hpp>

int main (int argc, char** argv)
{

    size_t iters = 1;
    std::vector<size_t> N;
    if(argc == 1) {
        auto r = boost::irange(1, 4000, 1);
        std::copy(r.begin(), r.end(), std::back_inserter(N));
        //r = boost::irange(1000, 2000, 10);
        //std::copy(r.begin(), r.end(), std::back_inserter(N));
    }

    else if(argc == 3)
    {
        N = {boost::lexical_cast<size_t>(argv[1])};
        iters = {boost::lexical_cast<size_t>(argv[2])};
    }
    else
    {
        std::cout << "no arguments: sweep; else ./<name> matrixSize iterations" << std::endl;
        exit(0);
    }

    std::cout << "# iters, size, time(ns)" << std::endl;
    for(auto n: N) {
        auto a = make_ij_sum(n);
        auto b = make_ij_product(n);
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t _ = 0; _!=iters; ++_) {
            mmul(a, b);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << iters <<", "
        << n << ", "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << std::endl;

    }
}