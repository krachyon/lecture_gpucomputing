#include "reduction.h"
#include "tracing.h"

float reduce_cpu(std::vector<float>const& in_orig)
{
    auto in = in_orig;

    unsigned long long int gauss = in.size()*(in.size()+1)/2;


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