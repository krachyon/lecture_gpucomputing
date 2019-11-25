#include "matrix.h"
#include "matrix_cuda.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <boost/range/irange.hpp>
#include <boost/lexical_cast.hpp>
#include <eigen3/Eigen/Core>
#include <omp.h>

void usage()
{
    std::cout << "***naive matrix multiplication benchmark***" << std::endl
              << "need 5 args: ./<name> 'method' size_from size_to size_step iters thread_size" << std::endl
              << "method:= {'cpu', 'eigen', 'cuda_naive', 'cuda_shared'}" << std::endl;

    exit(0);
}

template<typename T>
Matrix<T> mmul_eigen(Matrix<T> const& lhs, Matrix<T> const& rhs)
{
    using eigen_T = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    eigen_T lhs_eigen(lhs.M, lhs.N);
    for (size_t i = 0; i!=lhs.M; ++i)
        for(size_t j = 0;j!=lhs.N; ++j)
            lhs_eigen(i,j)=lhs(i,j);

    eigen_T rhs_eigen(rhs.M, rhs.N);
    for (size_t i = 0; i!=rhs.M; ++i)
        for(size_t j = 0;j!=rhs.N; ++j)
            rhs_eigen(i,j)=rhs(i,j);

    eigen_T ret_eigen = lhs_eigen * rhs_eigen;

    Matrix<T> ret(ret_eigen.rows(), ret_eigen.cols());
    for (size_t i = 0; i!=ret.M; ++i)
        for(size_t j = 0;j!=ret.N; ++j)
            ret(i,j)=ret_eigen(i,j);

    return ret;
}

int main(int argc, char** argv)
{
    Eigen::initParallel();

    size_t iters = 1;
    uint32_t thread_size = 8;
    std::vector<size_t> N;
    std::string method;

    if (argc==7) {
        try {
            method = boost::lexical_cast<std::string>(argv[1]);
            size_t start = boost::lexical_cast<size_t>(argv[2]);
            size_t stop = boost::lexical_cast<size_t>(argv[3]);
            size_t step = boost::lexical_cast<size_t>(argv[4]);
            iters = boost::lexical_cast<size_t>(argv[5]);
            thread_size = boost::lexical_cast<size_t>(argv[6]);
            auto r = boost::irange(start, stop, step);
            std::copy(r.begin(), r.end(), std::back_inserter(N));

            }
            catch (...) {
                usage();
            }
        }
    else {
        usage();
    }

    std::cout << "#method, iters, size, thread_size, memtime(ns), time(ns)" << std::endl;
    std::cout << "#Threads for eigen: " << Eigen::nbThreads( ) << std::endl;
    for (auto n: N) {
        auto a = make_ij_sum(n);
        auto b = make_ij_product(n);
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t _ = 0; _!=iters; ++_) {
            if (method == "cpu")
                mmul(a,b);
            else if(method == "eigen")
                mmul_eigen(a,b);
            else if(method == "cuda_naive")
                mmul_cuda_naive(a,b,thread_size);
            else if(method == "cuda_shared")
                mmul_cuda_shared(a,b,thread_size);
            else
                usage();
        }

        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << method << ", "
                  << iters << ", "
                  << n << ", "
                  << thread_size << ", "
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(lastMemoryOp).count() << ", "
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count() << std::endl;

    }
}
