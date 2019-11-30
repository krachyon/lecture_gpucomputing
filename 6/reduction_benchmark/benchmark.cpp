#include "reduction.h"
#include "tracing.h"
#include <boost/lexical_cast.hpp>
#include <boost/range/irange.hpp>
#include <iostream>

void usage()
{
    std::cout << "Usage: <exe> [method] [dtype_size=={16,32,64}] [size_from] [size_to]" << std::endl;
    exit(-1);
}

//template<typename T, typename F>
//void benchmark(size_t n, F func, std::string const& trace_name_prefix)
//{
//    std::vector<T>(n);
//    func();
//
//}
//

int main(int argc, char** argv)
{

//    std::string method="invalid";
//    size_t dtype_size=32;
//    size_t size_from=0;
//    size_t size_to=0;
//
//    if(argc == 5) {
//        try {
//            method = boost::lexical_cast<std::string>(argv[1]);
//            dtype_size = boost::lexical_cast<size_t>(argv[2]);
//            size_from = boost::lexical_cast<size_t>(argv[3]);
//            size_to = boost::lexical_cast<size_t>(argv[4]);
//        }
//        catch(...)
//        {
//            usage();
//        }
//    } else{
//        usage();
//    }
//    auto r = boost::irange(size_from, size_to);
//    for(auto n:r) {
//        if (dtype_size == 16 && method) {
//            benchmark<uint16_t>(n,)
//        }
//    }
//    }

auto pows = boost::irange(2,15);
std::vector<size_t> n_blocks = {1,2,4,8,16,32};
size_t n_iter = 2000;

std::cout << "N_elem,N_block,dtype,method,t_tot" << std::endl;
for(auto p: pows)
{
    for(auto n_block: n_blocks) {
        size_t n = (size_t)pow(2,p);
        if(n_block>=n||n/n_block > 1024)
            continue;
        std::vector<uint16_t> ini16(n);
        std::vector<uint32_t> ini32(n);
        std::vector<float> in32(n);
        std::vector<double> in64(n);

        reduce_cuda_naive(ini16,n_block, n_iter);
        std::cout << n << "," << n_block << "," << "uint16_t," << "cuda_naive,"
            << Trace::get("cuda_naive_start","cuda_naive_end") << "," << std::endl;
        reduce_cuda_shared(ini16,n_block,n_iter);
        std::cout << n << "," << n_block << "," << "uint16_t," << "cuda_shared,"
            << Trace::get("cuda_shared_start","cuda_shared_end") << "," << std::endl;

        reduce_cuda_naive(ini32,n_block,n_iter);
        std::cout << n << "," << n_block << "," << "uint32_t," << "cuda_naive,"
                  << Trace::get("cuda_naive_start","cuda_naive_end") << "," << std::endl;
        reduce_cuda_shared(ini32,n_block,n_iter);
        std::cout << n << "," << n_block << "," << "uint32_t," << "cuda_shared,"
                  << Trace::get("cuda_shared_start","cuda_shared_end") << "," << std::endl;

        reduce_cuda_naive(in32,n_block,n_iter);
        std::cout << n << "," << n_block << "," << "float," << "cuda_naive,"
                  << Trace::get("cuda_naive_start","cuda_naive_end") << "," << std::endl;
        reduce_cuda_shared(in32,n_block,n_iter);
        std::cout << n << "," << n_block << "," << "float," << "cuda_shared,"
                  << Trace::get("cuda_shared_start","cuda_shared_end") << "," << std::endl;

        reduce_cuda_naive(in64,n_block,n_iter);
        std::cout << n << "," << n_block << "," << "double," << "cuda_naive,"
                  << Trace::get("cuda_naive_start","cuda_naive_end") << "," << std::endl;
        reduce_cuda_shared(in64,n_block,n_iter);
        std::cout << n << "," << n_block << "," << "double," << "cuda_shared,"
                  << Trace::get("cuda_shared_start","cuda_shared_end") << "," << std::endl;

    }
}
}