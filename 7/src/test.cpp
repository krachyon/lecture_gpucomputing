#include "nbody.h"
#include <iostream>

int main()
{
    //std::cout << "time: "<< run_leapfrog_aos(100,32,100).first.count() << std::endl;
    //std::cout << "time: "<< run_leapfrog_aos(200,32,100).first.count() << std::endl;
    //std::cout << "time: "<< run_leapfrog_aos(400,32,100).first.count() << std::endl;
    //std::cout << "time: "<< run_leapfrog_aos(1000,32,100).first.count() << std::endl;

    std::cout << "time: " << run_leapfrog_soa(100,32,100).first.count() << std::endl;
    std::cout << "time: " << run_leapfrog_soa(500,512,100).first.count() << std::endl;


    //std::cout << "vec: " << res.second. << std::endl;
}

//#include "utility.cuh"
//int main()
//
//{
//    float3 a{0,0,0};
//    float3 b{1,1,1};
//
//    a+=b*2;
//    std::cout << a.x << a.y << a.z;
//}