#include "nbody.h"
#include <iostream>

int main()
{
    std::cout << "time: "<< run_leapfrog_aos(10,32,10).count() << std::endl;
    std::cout << "time: "<< run_leapfrog_aos(100,32,10000).count() << std::endl;
    std::cout << "time: "<< run_leapfrog_aos(1000,32,10000).count() << std::endl;
    std::cout << "time: "<< run_leapfrog_aos(10000,32,10000).count() << std::endl;
    std::cout << "time: "<< run_leapfrog_aos(100000,32,1000000).count() << std::endl;

}
