
#include <sstream>
#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>

#include "matrix.h"
#include "matrix_cuda.h"


TEST(mmul,identity_square)
{
    Matrix<float> a(2,2);

    a(0,0) = 1.f;
    a(0,1) = 0.f;
    a(1,0) = 0.f;
    a(1,1) = 1.f;

    auto b = Matrix<float>::zeros(2,2);
    b(0,0) = 1.f;
    b(0,1) = 2.f;
    b(1,0) = 3.f;
    b(1,1) = 4.f;

    auto c = mmul(a,b);
    EXPECT_EQ(c(0,0),1.f);
    EXPECT_EQ(c(0,1), 2.f);
    EXPECT_EQ(c(1,0),3.f);
    EXPECT_EQ(c(1,1), 4.f);
}



TEST(mmul, square_compare_with_Eigen)
{
    //create a random 10*10 matrix in eigen and corresponding own matrix
    size_t m=10, n=10;
    Eigen::MatrixXd mat_eigen = Eigen::MatrixXd::Random(m,n);
    Matrix<double> mat(m,n);
    for (size_t i = 0; i!=m;++i)
        for(size_t j = 0;j!=n;++j)
            mat(i,j)=mat_eigen(i,j);

    //multiply both and check if they're the same
    auto m2 = mmul(mat,mat);
    auto m2_eigen = mat_eigen*mat_eigen;
    for (size_t i = 0; i!=m;++i)
        for(size_t j = 0;j!=n;++j)
            EXPECT_FLOAT_EQ(m2(i,j), m2_eigen(i,j));
}

TEST(mmul, non_square)
{
    Matrix<int> a(10,20);
    Matrix<int> b(20,5);

    auto c = mmul(a,b);
    EXPECT_EQ(c.M,a.M);
    EXPECT_EQ(c.N,b.N);
}

TEST(mmul, generators)
{
    auto A = make_ij_sum(5);
    auto B = make_ij_product(5);

    //verified with python:
    //x = np.arange(0,5)
    //A = np.outer(x,x)
    //B = np.add.outer(x,x)
    //C = A@B
    std::cout <<std::endl << A << std::endl;
    std::cout <<std::endl << A << std::endl;
    std::cout <<std::endl << mmul(A,B) << std::endl;
}

TEST(mmul_cuda, smoke)
{
    auto A = make_ij_sum(5);
    auto B = make_ij_product(5);

    auto C = mmul_cuda_naive_float(A,B);
    auto C_ref = mmul(A,B);

    EXPECT_FLOAT_EQ(C(1,1),C_ref(1,1));
    EXPECT_FLOAT_EQ(C(3,3),C_ref(3,3));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}