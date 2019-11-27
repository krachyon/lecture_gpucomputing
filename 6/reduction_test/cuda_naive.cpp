#include <gtest/gtest.h>
#include "reduction.h"
#include <numeric>


TEST(cuda_naive, smoke)
{
    std::vector<float> in(1024);
    std::iota(in.begin(), in.end(), 0);

    reduce_cuda_naive(in, 2);
}

TEST(cuda_naive, resize)
{
    std::vector<float> in(513);
    reduce_cuda_naive(in, 2);
    EXPECT_EQ(1024, in.size());
}


TEST(cuda_naive, zeros)
{
    std::vector<float> in(1024);
    std::fill(in.begin(), in.end(), 0);

    EXPECT_FLOAT_EQ(0, reduce_cuda_naive(in, 2));
}

TEST(cuda_naive, ones)
{
    std::vector<float> in(16);
    std::fill(in.begin(), in.end(), 1);

    //EXPECT_FLOAT_EQ(1024, reduce_cuda_naive(in, 1));
    EXPECT_FLOAT_EQ(16, reduce_cuda_naive(in, 2));
    //EXPECT_FLOAT_EQ(1024, reduce_cuda_naive(in, 4));
    //EXPECT_FLOAT_EQ(1024, reduce_cuda_naive(in, 8));
    //EXPECT_FLOAT_EQ(1024, reduce_cuda_naive(in, 16));
}