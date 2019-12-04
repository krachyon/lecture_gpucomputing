#include "reduction.h"
#include "tracing.h"

float reduce_cpu(std::vector<float>const& in_orig)
{
    Trace::set(tracepoint::start);
    Trace::set(tracepoint::copy_start);
    auto in = in_orig;
    Trace::set(tracepoint::copy_end);

    unsigned long long int gauss = in.size()*(in.size()+1)/2;


    for(size_t stride=2;stride <= in.size();stride*=2){
        for(size_t j=0;j+stride/2 < in.size();j+=stride){
            in[j]+=in[j+stride/2];
        }
    }
    Trace::set(tracepoint::end);
    return in[0];
}