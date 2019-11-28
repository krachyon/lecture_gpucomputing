#include "gtest/gtest.h"
#include "gtest/gtest-typed-test.h"
#include "reduction.h"
#include <numeric>




template <typename T>
class Cuda_Naive : public testing::Test
{
public:
    Cuda_Naive(){}
    ~Cuda_Naive() override{}

    std::vector<size_t> sizes = {1,2,4,8,16,32,512,1024};
    std::vector<size_t> block_sizes = {1,2,4,8,16};

    // maybe generalize this to all methods. But might be a bit too weird...
    // std::function<T (std::vector<T>&, size_t)> reduce_func;
};

template<typename T>
class Cuda_Naive_Float: public Cuda_Naive<T>{};

template<typename T>
class Cuda_Naive_Integral: public Cuda_Naive<T>{};

typedef testing::Types<float,double,int32_t,uint32_t,int16_t,uint16_t> allTypes;
typedef testing::Types<float,double> floats;
typedef testing::Types<uint32_t,int16_t,uint16_t> integral;

TYPED_TEST_SUITE(Cuda_Naive, allTypes);
TYPED_TEST_SUITE(Cuda_Naive_Float, floats);
TYPED_TEST_SUITE(Cuda_Naive_Integral, integral);

TYPED_TEST(Cuda_Naive, smoke)
{
    for(auto s: this->sizes)
        for(auto bs: this->block_sizes) {
            std::vector<TypeParam> in(s);
            std::iota(in.begin(), in.end(), 0);

            reduce_cuda_naive(in, bs);
        }
}

TYPED_TEST(Cuda_Naive, resize)
{
    std::vector<TypeParam> in(513);
    reduce_cuda_naive(in, 2);
    EXPECT_EQ(1024, in.size());
}


TYPED_TEST(Cuda_Naive_Float, zeros)
{
    for(auto s: this->sizes)
        for(auto bs: this->block_sizes) {
            if(bs >= s) {
                continue;
            }
            std::vector<TypeParam> in(s);
            std::fill(in.begin(), in.end(), 0);

            EXPECT_FLOAT_EQ(0, reduce_cuda_naive(in, bs)) << "size: " << s << " block size: " << bs;
        }
}

TYPED_TEST(Cuda_Naive_Integral, zeros)
{
    for(auto s: this->sizes)
        for(auto bs: this->block_sizes) {
            if(bs >= s) {
                continue;
            }
            std::vector<TypeParam> in(s);
            std::fill(in.begin(), in.end(), 0);

            EXPECT_EQ(0, reduce_cuda_naive(in, bs)) << "size: " << s << " block size: " << bs;
        }
}


TYPED_TEST(Cuda_Naive_Float, ones)
{
    for(auto s: this->sizes)
        for(auto bs: this->block_sizes) {
            if (bs >= s) {
                continue;
            }
            std::vector<TypeParam> in(s);
            std::fill(in.begin(), in.end(), 1);

            EXPECT_FLOAT_EQ(s, reduce_cuda_naive(in, bs)) << "size: " << s << " block size: " << bs;;
        }
}

TYPED_TEST(Cuda_Naive_Integral, ones)
{
    for(auto s: this->sizes)
        for(auto bs: this->block_sizes) {
            if (bs >= s) {
                continue;
            }
            std::vector<TypeParam> in(s);
            std::fill(in.begin(), in.end(), 1);

            EXPECT_EQ(s, reduce_cuda_naive(in, bs)) << "size: " << s << " block size: " << bs;;
        }
}

TEST(thrust,simple)
{
    std::vector<float> in(1024);
    std::iota(in.begin(),in.end(),0.f);
    EXPECT_EQ(thrust_reduce(in),std::accumulate(in.begin(),in.end(), 0.f));
}