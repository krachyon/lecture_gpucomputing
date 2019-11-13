#include "matrix.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <boost/range/irange.hpp>

int main ()
{
    size_t iters = 1;
    std::vector<size_t> N;
    auto r = boost::irange(1, 100,1);
    std::copy(r.begin(),r.end(),std::back_inserter(N));
    r = boost::irange(100,1000,10);
    std::copy(r.begin(),r.end(),std::back_inserter(N));


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