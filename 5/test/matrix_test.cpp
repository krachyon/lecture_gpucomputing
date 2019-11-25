
#include <sstream>
#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>

#include "matrix.h"
#include "matrix_cuda.h"

TEST(matrix, zeros)
{
    Matrix<double> A = Matrix<double>::zeros(20,20);
    for(auto it = A.begin(); it!=A.end();++it)
        EXPECT_EQ(0,*it);

    Matrix<double> C = Matrix<double>::zeros(15,15);
    for(auto it = C.begin(); it!=C.end();++it)
        EXPECT_EQ(0,*it);

    Matrix<float> B = Matrix<float>::zeros(3,12);
    for(auto it = B.begin(); it!=B.end();++it)
        EXPECT_EQ(0,*it);
}

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

//TODO all of these should be parametrized both by type and multiply operation

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

TEST(mmul, types)
{
    Matrix<double> A64 = Matrix<double>::zeros(20,20);
    Matrix<double> B64 = Matrix<double>::zeros(20,20);

    auto C64 = mmul(A64,B64);
    for(auto it = C64.begin(); it!=C64.end();++it)
    {
        EXPECT_DOUBLE_EQ(0,*it);
    }

    Matrix<int16_t> A16 = Matrix<int16_t>::zeros(20,20);
    Matrix<int16_t> B16 = Matrix<int16_t>::zeros(20,20);

    auto C16 = mmul(A16,B16);
    for(auto it = C16.begin(); it!=C16.end();++it)
    {
    EXPECT_EQ(0,*it);
    }
}


TEST(mmul_cuda, simple_equality)
{
    auto A = make_ij_sum(5);
    auto B = make_ij_product(5);

    auto C = mmul_cuda_naive(A,B);
    auto C_ref = mmul(A,B);

    for(auto it = C.begin(), rit = C_ref.begin(); it != C.end(); ++it,++rit)
    {
        EXPECT_FLOAT_EQ(*it,*rit);
    }
}

TEST(mmul_cuda, large_equality)
{
    auto A = make_ij_sum(50);
    auto B = make_ij_product(50);

    auto C = mmul_cuda_naive(A,B);
    auto C_ref = mmul(A,B);

    for(auto it = C.begin(), rit = C_ref.begin(); it != C.end(); ++it,++rit)
    {
        EXPECT_FLOAT_EQ(*it,*rit);
    }
}

TEST(mmul_cuda, square_compare_with_Eigen)
{
    //create a random 10*10 matrix in eigen and corresponding own matrix
    size_t m=40, n=40;
    Eigen::MatrixXd mat_eigen = Eigen::MatrixXd::Random(m,n);
    Matrix<double> mat(m,n);
    for (size_t i = 0; i!=m;++i)
        for(size_t j = 0;j!=n;++j)
            mat(i,j)=mat_eigen(i,j);

    //multiply both and check if they're the same
    auto m2 = mmul_cuda_naive(mat,mat);
    auto m2_eigen = mat_eigen*mat_eigen;
    for (size_t i = 0; i!=m;++i)
        for(size_t j = 0;j!=n;++j)
            EXPECT_FLOAT_EQ(m2(i,j), m2_eigen(i,j));
}

TEST(mmul_cuda, non_square)
{
    size_t m=20, n=40, o=30;
    Eigen::MatrixXd mat_eigen_left = Eigen::MatrixXd::Random(m,n);
    Eigen::MatrixXd mat_eigen_right = Eigen::MatrixXd::Random(n,o);

    Matrix<double> mat_left(m,n);
    Matrix<double> mat_right(n,o);

    for (size_t i = 0; i!=m;++i)
        for(size_t j = 0;j!= n;++j)
            mat_left(i,j)=mat_eigen_left(i,j);
    for (size_t i = 0; i!=n;++i)
        for(size_t j = 0;j!= o;++j)
            mat_right(i,j)=mat_eigen_right(i,j);

    //multiply both and check if they're the same
    auto m2 = mmul_cuda_naive(mat_left,mat_right);
    auto m2_eigen = mat_eigen_left*mat_eigen_right;

    for (size_t i = 0; i!= m2.M ;++i)
        for(size_t j = 0; j!=m2.N; ++j)
            EXPECT_FLOAT_EQ(m2(i,j), m2_eigen(i,j));
}
TEST(mmul_cuda, tiny)
{
}

//template <class T>
//class MatrixSuite: public ::testing::Test
//{};
//
//using testing::Types;
//using MatrixTypes = Types<float,double,int16_t> ;
//
//TYPED_TEST_SUITE(PrimeTableTest, MatrixTypes);


TEST(mmul_cuda, double_exact_tiling)
{
    Matrix<double> A64 = Matrix<double>::zeros(16,16);
    Matrix<double> B64 = Matrix<double>::zeros(16,16);

    auto C64 = mmul_cuda_naive(A64,B64);
    for(double* it = C64.begin(); it!=C64.end();++it)
    {
        EXPECT_DOUBLE_EQ(0,*it);
    }
}

TEST(mmul_cuda, double_crooked_tiling)
{
    Matrix<double> A64 = Matrix<double>::zeros(15,15);
    Matrix<double> B64 = Matrix<double>::zeros(15,15);

    auto C64 = mmul_cuda_naive(A64,B64);
    for(double* it = C64.begin(); it!=C64.end();++it)
    {
        EXPECT_DOUBLE_EQ(0,*it);
    }
}

TEST(mmul_cuda, double_single_tile)
{
    Matrix<double> A64 = Matrix<double>::zeros(8,8);
    Matrix<double> B64 = Matrix<double>::zeros(8,8);

    auto C64 = mmul_cuda_naive(A64,B64);
    for(double* it = C64.begin(); it!=C64.end();++it)
    {
        EXPECT_DOUBLE_EQ(0,*it);
    }
}

TEST(mmul_cuda, double_less_than_single_tile)
{
    Matrix<double> A64 = Matrix<double>::zeros(6,6);
    Matrix<double> B64 = Matrix<double>::zeros(6,6);

    auto C64 = mmul_cuda_naive(A64,B64);
    for(double* it = C64.begin(); it!=C64.end();++it)
    {
        EXPECT_DOUBLE_EQ(0,*it);
    }
}

TEST(mmul_cuda, float_large)
{
    Matrix<float> A64 = Matrix<float>::zeros(50,50);
    Matrix<float> B64 = Matrix<float>::zeros(50,50);

    auto C64 = mmul_cuda_naive(A64,B64);
    for(auto it = C64.begin(); it!=C64.end();++it)
    {
        EXPECT_DOUBLE_EQ(0,*it);
    }
}

TEST(mmul_cuda, float_single_tile)
{
    Matrix<float> A64 = Matrix<float>::zeros(8,8);
    Matrix<float> B64 = Matrix<float>::zeros(8,8);

    auto C64 = mmul_cuda_naive(A64,B64);
    for(auto it = C64.begin(); it!=C64.end();++it)
    {
        EXPECT_DOUBLE_EQ(0,*it);
    }
}

TEST(mmul_cuda, float_two_tiles)
{
    Matrix<float> A64 = Matrix<float>::zeros(16,16);
    Matrix<float> B64 = Matrix<float>::zeros(16,16);

    auto C64 = mmul_cuda_naive(A64,B64);
    for(auto it = C64.begin(); it!=C64.end();++it)
    {
        EXPECT_DOUBLE_EQ(0,*it);
    }
}


TEST(mmul_cuda, int16_t)
{
    Matrix<int16_t> A16 = Matrix<int16_t>::zeros(20,20);
    Matrix<int16_t> B16 = Matrix<int16_t>::zeros(20,20);

    auto C16 = mmul_cuda_naive(A16,B16);
    for(int16_t* it = C16.begin(); it!=C16.end();++it)
    {
        EXPECT_EQ(0,*it);
    }
}
//
//TEST(mmul_cuda_shared, smoke)
//{
//    Matrix<float> A(5,5);
//    Matrix<float> B(5,5);
//    mmul_cuda_shared(A,B);
//}



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}