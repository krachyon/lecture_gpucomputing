#include <gtest/gtest.h>
#include "reduction.h"
#include <numeric>

TEST(cpu, ones)
{
    std::vector<float> in0(1,1);
    EXPECT_FLOAT_EQ(1, reduce_cpu(in0));
    std::vector<float> in(1000,1);
    EXPECT_FLOAT_EQ(1000, reduce_cpu(in));
    std::vector<float> in2(2000,1);
    EXPECT_FLOAT_EQ(2000, reduce_cpu(in2));
}

TEST(cpu, debug)
{
    std::vector<float> in0(17,1);
    EXPECT_FLOAT_EQ(16, reduce_cpu(in0));
}

TEST(cpu, zero)
{
    std::vector<float> in0(1,0);
    EXPECT_FLOAT_EQ(0, reduce_cpu(in0));
    std::vector<float> in(1000,0);
    EXPECT_FLOAT_EQ(0, reduce_cpu(in));
    std::vector<float> in2(2000,0);
    EXPECT_FLOAT_EQ(0, reduce_cpu(in2));
}

TEST(cpu, iota)
{
    std::vector<float> in(1000);
    std::iota(in.begin(),in.end(),1);
    EXPECT_FLOAT_EQ(std::accumulate(in.begin(),in.end(),0.f), reduce_cpu(in));
}

