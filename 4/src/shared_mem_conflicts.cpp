#include "kernel.cuh"
#include <iostream>
#include <boost/range/irange.hpp>
int main()
{
    auto strides = boost::irange(1,65,1);
    auto threads = boost::irange(1,10,1);
    auto blocks = boost::irange(1,10,1);

    for(auto stride: strides)
        for(auto thread: threads)
            for(auto block: blocks){
                size_t bytes = 12*1024*sizeof(float);
                if ((block*thread) > (bytes/sizeof(float)/64/stride) )
                    continue;
                auto result = bankConflictsRead_Wrapper(block, thread, stride, bytes);
                for (auto elem: result)
                    std::cout << elem << " ";
                std::cout << std::endl;
    }
}