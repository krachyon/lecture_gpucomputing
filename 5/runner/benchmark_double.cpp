#include "matrix.h"
#include "matrix_cuda.h"
#include <eigen3/Eigen/Core>
#include <boost/range/irange.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>

using std::chrono::high_resolution_clock;

int main(int argc, char** argv)
{
    size_t start = boost::lexical_cast<size_t>(argv[1]);
    size_t stop = boost::lexical_cast<size_t>(argv[2]);
    size_t step = boost::lexical_cast<size_t>(argv[3]);
    auto N = boost::irange(start, stop, step);

    std::cout << "# double precission" << std::endl;
    std::cout << "#N,cuda,eigen" << std::endl;
    for (auto n: N) {
        Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
        Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, n);

        Matrix<double> AA(n,n);
        Matrix<double> BB(n,n);

        auto start = high_resolution_clock::now();
        mmul_cuda_shared(AA,BB);
        auto stop = high_resolution_clock::now();
        std::chrono::nanoseconds cuda_time = stop-start;

        start = high_resolution_clock::now();
        Eigen::MatrixXd C = A*B;
        stop = high_resolution_clock::now();
        std::chrono::nanoseconds eigen_time = stop-start;

        std::cout<<n<<","<< cuda_time.count() << "," << eigen_time.count() << std::endl;
    }

}