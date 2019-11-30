#include <gtest/gtest.h>
#include "tracing.h"
#include <iostream>
#include <thread>

const size_t accuracy = 800;
//TODO looks like the accuracy is shit
TEST(DISABLED_tracing, instant)
{
    //warmup
    Trace::set("dummy");
    Trace::set("dummy2");


    for(auto i=0;i!=1000;++i) {
        Trace::set("a");
        Trace::set("b");
        EXPECT_GT(accuracy, Trace::get("a", "b"));
    }

}


TEST(DISABLED_tracing, particle_instant)
{
    for(auto i=0;i!=1000;++i) {
        {
            TraceParticle("a");
        }
        EXPECT_GT(accuracy, Trace::get("a_start", "a_stop"));
    }
}


TEST(DISABLED_tracing, sleep)
{
    Trace::set("dummy");
    Trace::set("dummy2");


    for(auto i=0;i!=1000;++i) {
        Trace::set("a");
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        Trace::set("b");
        EXPECT_GT(100*1000+accuracy, Trace::get("a", "b"));
        EXPECT_LT(100*1000-accuracy, Trace::get("a", "b"));
    }

}
