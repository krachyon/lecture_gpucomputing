#include "reduction.h"
#include "tracing.h"
#include <boost/lexical_cast.hpp>
#include <boost/range/irange.hpp>
#include <iostream>
#include <numeric>


void usage()
{
    std::cout << "Usage: <exe> [method] [dtype_size=={16,32,64}] [size_from] [size_to]" << std::endl;
    exit(-1);
}

template<typename Value>
void csv_out(Value val)
{
    std::cout << val << std::endl;
}

template<typename Value, typename... Values>
void csv_out(Value val, Values... vals)
{
    std::cout << val << ",";
    csv_out(vals...);
}

template<typename... Values>
void csv_autotimings(Values... vals)
{

    csv_out(vals...,
            Trace::get(tracepoint::start,tracepoint::end),
            Trace::get(tracepoint::copy_start, tracepoint::copy_end),
            Trace::get(tracepoint::copy_end, tracepoint::backcopy_start),
            Trace::get(tracepoint::backcopy_start, tracepoint::backcopy_end)
            );
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

auto pows = boost::irange(10,22);
std::vector<size_t> n_blocks = {32,64,128,256,512,1024};
size_t n_iter = 2000;
std::cout << "#N_elem,N_iter,N_block,dtype,method,t_tot,t_copy,t_exec,t_backcopy" << std::endl;
for(auto p: pows)
{
    size_t n = 1 << p;
    std::vector<uint16_t> ini16(n);
    std::vector<uint32_t> ini32(n);
    std::vector<float> in32(n);
    std::vector<double> in64(n);

    for(auto n_block: n_blocks) {
        if(n_block>=n||n/n_block > 1024)
            continue;

        reduce_cuda_naive(ini16,n_block, n_iter);
        csv_autotimings(n,n_iter, n_block, "uint16_t", "cuda_naive");

        reduce_cuda_shared(ini16,n_block,n_iter);
        csv_autotimings(n,n_iter, n_block, "uint16_t", "cuda_shared");

        reduce_cuda_naive(ini32,n_block,n_iter);
        csv_autotimings(n,n_iter, n_block, "uint32_t", "cuda_naive");

        reduce_cuda_shared(ini32,n_block,n_iter);
        csv_autotimings(n,n_iter, n_block, "uint32_t", "cuda_shared");

        reduce_cuda_naive(in32,n_block,n_iter);
        csv_autotimings(n,n_iter, n_block, "float", "cuda_naive");
        reduce_cuda_shared(in32,n_block,n_iter);
        csv_autotimings(n,n_iter, n_block, "float", "cuda_shared");


        reduce_cuda_naive(in64,n_block,n_iter);
        csv_autotimings(n,n_iter, n_block, "double", "cuda_naive");
        reduce_cuda_shared(in64,n_block,n_iter);
        csv_autotimings(n,n_iter, n_block, "double", "cuda_shared");
    }


    thrust_reduce(in32);
    csv_autotimings(n, n_iter, 1, "float", "thrust");

    //Attention: Disregard memory timings for this method, they don't actually exist
    Trace::set(tracepoint::start);
    volatile float res;
    for(size_t i=0; i!=n_iter; ++i)
    {
        res = std::accumulate(in32.begin(), in32.end(), 0.f);
    }
    Trace::set(tracepoint::end);
    csv_autotimings(n,n_iter, 1, "float", "std::accumulate");

    for(size_t i=0; i!= n_iter; ++i)
    {
        res = reduce_cpu(in32);
    }
    csv_autotimings(n, n_iter, 1, "float", "cpu");
}
}