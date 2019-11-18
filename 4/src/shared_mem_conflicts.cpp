#include "kernel.cuh"
#include <iostream>

int main()
{
    auto test = bankConflictsRead_Wrapper(1,1,1);
    for(auto elem: test)
        std::cout << elem << " ";
    std::cout << std::endl;
}