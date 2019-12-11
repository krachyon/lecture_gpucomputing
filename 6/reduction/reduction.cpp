#include "reduction.h"
#include "tracing.h"
#include <future>
#include <numeric>
#include <iterator>

float reduce_cpu(std::vector<float>const& in_orig)
{
    auto in = in_orig;



    for(size_t stride=2;stride <= in.size();stride*=2){
        for(size_t j=0;j+stride/2 < in.size();j+=stride){
            in[j]+=in[j+stride/2];
        }
    }
    return in[0];
}

float reduce_cpu_seq(std::vector<float>const& in)
{

    float res = 0;
    for(auto const elem: in)
        res+=elem;

    return res;
}

float reduce_cpu_async(std::vector<float>const& in)
{
    size_t n_threads = std::thread::hardware_concurrency();

    auto acc = [](decltype(in.begin()) beg, decltype(in.end()) end){
        return std::accumulate(beg, end, 0.0f);
    };

    using future = decltype(std::async(std::launch::async, acc, in.begin(), in.end()));

    std::vector<future> fut;

    size_t bs = in.size()/n_threads;

    for(size_t i=0; i != n_threads; ++i)
    {
        fut.push_back( std::async(std::launch::async,
                acc, in.begin()+i*bs, in.begin()+(i+1)*bs
                ));
    }
    fut.push_back( std::async(std::launch::async,
                              acc, in.begin()+ (n_threads*bs), in.end()
    ));

    auto ret = 0.0f;
    for(auto& res: fut)
    {
        ret += res.get();
    }

    return ret;

}