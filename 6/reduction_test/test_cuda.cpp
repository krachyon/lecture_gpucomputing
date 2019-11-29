#include "gtest/gtest.h"
#include "gtest/gtest-typed-test.h"
#include "reduction.h"
#include <numeric>

//TODO this works to re-use test cases for all declared combinations of data type and function to test
// but i guess it's now a bit Eldritch, needs a bunch of helpers and is still verbose in the declaration
// Maybe the type-lists could be generated easier with boost.hana or boost.mpl but at that point maybe gtest just
// isn't supposed to be (ab)used like this
//TODO also gtest_filter doesn't really work anymore

//The thing a test case get's instantiated with
template<typename _DataType, template<typename> class _ReduceFunc>
struct testCombination
{
    using DataType = _DataType;
    using ReduceFunc = _ReduceFunc<DataType>;
};

template<typename T>
struct cuda_naive_func
{
    T operator()(std::vector<T>& in,size_t blockSize){return reduce_cuda_naive(in,blockSize);}
};

template<typename T>
struct cuda_shared_func
{
    T operator()(std::vector<T>& in,size_t blockSize){return reduce_cuda_shared(in,blockSize);}
};

template <typename T>
class CudaFixture : public testing::Test
{
public:
    CudaFixture(){}
    ~CudaFixture() override{}

    std::vector<size_t> sizes = {1,2,4,8,16,32,512,1024};
    std::vector<size_t> block_sizes = {1,2,4,8,16};
};

template<typename T>
class CudaFixture_Float: public CudaFixture<T>{};

template<typename T>
class CudaFixture_Integral: public CudaFixture<T>{};


using allTypes = testing::Types<testCombination<float,cuda_naive_func>,
                                testCombination<double,cuda_naive_func>,
                                testCombination<uint16_t,cuda_naive_func>,
                                testCombination<int16_t,cuda_naive_func>,
                                testCombination<uint32_t,cuda_naive_func>,
                                testCombination<int32_t,cuda_naive_func>,
                                testCombination<float,cuda_shared_func>,
                                testCombination<double,cuda_shared_func>,
                                testCombination<uint16_t,cuda_shared_func>,
                                testCombination<int16_t,cuda_shared_func>,
                                testCombination<uint32_t,cuda_shared_func>,
                                testCombination<int32_t,cuda_shared_func>
                >;

using floats = testing::Types<testCombination<float,cuda_naive_func>,
                              testCombination<double,cuda_naive_func>,
                              testCombination<float,cuda_shared_func>,
                              testCombination<double,cuda_shared_func>
        >;
using integral = testing::Types<
        testCombination<uint16_t,cuda_naive_func>,
        testCombination<int16_t,cuda_naive_func>,
        testCombination<uint32_t,cuda_naive_func>,
        testCombination<int32_t,cuda_naive_func>,
        testCombination<uint16_t,cuda_shared_func>,
        testCombination<int16_t,cuda_shared_func>,
        testCombination<uint32_t,cuda_shared_func>,
        testCombination<int32_t,cuda_shared_func>
        >;

TYPED_TEST_SUITE(CudaFixture, allTypes);
TYPED_TEST_SUITE(CudaFixture_Float, floats);
TYPED_TEST_SUITE(CudaFixture_Integral, integral);

TYPED_TEST(CudaFixture, smoke)
{
    for(auto s: this->sizes)
        for(auto bs: this->block_sizes) {
            std::vector<typename TypeParam::DataType> in(s);
            std::iota(in.begin(), in.end(), 0);

            typename TypeParam::ReduceFunc()(in, bs);
        }
}

TYPED_TEST(CudaFixture, resize)
{
    std::vector<typename TypeParam::DataType> in(513);
    reduce_cuda_naive(in, 2);
    EXPECT_EQ(1024, in.size());
}


TYPED_TEST(CudaFixture_Float, zeros)
{
    for(auto s: this->sizes)
        for(auto bs: this->block_sizes) {
            if(bs >= s) {
                continue;
            }
            std::vector<typename TypeParam::DataType> in(s);
            std::fill(in.begin(), in.end(), 0);

            EXPECT_FLOAT_EQ(0, reduce_cuda_naive(in, bs)) << "size: " << s << " block size: " << bs;
        }
}

TYPED_TEST(CudaFixture_Integral, zeros)
{
    for(auto s: this->sizes)
        for(auto bs: this->block_sizes) {
            if(bs >= s) {
                continue;
            }
            std::vector<typename TypeParam::DataType> in(s);
            std::fill(in.begin(), in.end(), 0);

            EXPECT_EQ(0, reduce_cuda_naive(in, bs)) << "size: " << s << " block size: " << bs;
        }
}


TYPED_TEST(CudaFixture_Float, ones)
{
    for(auto s: this->sizes)
        for(auto bs: this->block_sizes) {
            if (bs >= s) {
                continue;
            }
            std::vector<typename TypeParam::DataType> in(s);
            std::fill(in.begin(), in.end(), 1);

            EXPECT_FLOAT_EQ(s, reduce_cuda_naive(in, bs)) << "size: " << s << " block size: " << bs;;
        }
}

TYPED_TEST(CudaFixture_Integral, ones)
{
    for(auto s: this->sizes)
        for(auto bs: this->block_sizes) {
            if (bs >= s) {
                continue;
            }
            std::vector<typename TypeParam::DataType> in(s);
            std::fill(in.begin(), in.end(), 1);

            EXPECT_EQ(s, reduce_cuda_naive(in, bs)) << "size: " << s << " block size: " << bs;;
        }
}

TEST(debug,shared)
{
    std::vector<float> in(64);
    std::fill(in.begin(), in.end(), 1);
    EXPECT_EQ(64, reduce_cuda_shared(in, 2));
}

TEST(thrust,simple)
{
    std::vector<float> in(1024);
    std::iota(in.begin(),in.end(),0.f);
    EXPECT_EQ(thrust_reduce(in),std::accumulate(in.begin(),in.end(), 0.f));
}