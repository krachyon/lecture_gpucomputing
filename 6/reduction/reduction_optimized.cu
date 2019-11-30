#include "reduction.h"
#include "tracing.h"
#include <thrust/device_vector.h>

float thrust_reduce(std::vector<float>const& in, size_t n_iter)
{
    Trace::set(tracepoint::start);

    Trace::set(tracepoint::copy_start);
    thrust::device_vector<float>(in.begin(),in.end());
    Trace::set(tracepoint::copy_end);
    float sum = 0.f;
    for(size_t i=0; i!= n_iter;++i)
        sum = thrust::reduce(in.begin(), in.end());

    Trace::set(tracepoint::end);
    return sum;
}

