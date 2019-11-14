#include "matrix.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <boost/range/irange.hpp>
#include <boost/lexical_cast.hpp>

void usage()
{
    std::cout << "***naive matrix multiplication benchmark***" << std::endl
              << "no arguments: predefined" << std::endl
              << "2 args: ./<name> Size iterations" << std::endl
              << "3 args: ./<name> size_from size_to step" << std::endl;
    exit(0);
}

int main(int argc, char** argv)
{

    size_t iters = 1;
    std::vector<size_t> N;
    if (argc==1) {
        auto r = boost::irange(1, 4000, 1);
        std::copy(r.begin(), r.end(), std::back_inserter(N));
        //r = boost::irange(1000, 2000, 10);
        //std::copy(r.begin(), r.end(), std::back_inserter(N));
    }

    else if (argc==3) {
        try {
            N = {boost::lexical_cast<size_t>(argv[1])};
            iters = {boost::lexical_cast<size_t>(argv[2])};
        }
        catch (...) {
            usage();
        }
    }
    else if (argc==4) {
        try {
            size_t start = {boost::lexical_cast<size_t>(argv[1])};
            size_t stop = {boost::lexical_cast<size_t>(argv[2])};
            size_t step = {boost::lexical_cast<size_t>(argv[3])};
            iters = 1;
            auto r = boost::irange(start, stop, step);
            std::copy(r.begin(), r.end(), std::back_inserter(N));
        }
        catch (...) {
            usage();
        }
    }
    else {
        usage();
    }

    std::cout << "# iters, size, time(ns)" << std::endl;
    for (auto n: N) {
        auto a = make_ij_sum(n);
        auto b = make_ij_product(n);
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t _ = 0; _!=iters; ++_) {
            mmul(a, b);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << iters << ", "
                  << n << ", "
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << std::endl;

    }
}