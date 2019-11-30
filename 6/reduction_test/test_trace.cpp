#include <gtest/gtest.h>
#include "tracing.h"
#include <iostream>
#include <thread>


TEST(tracing, instant)
{
    const size_t accuracy = 100;

    //warmup
    Trace::set(tracepoint::start);
    Trace::set(tracepoint::end);


    for(auto i=0;i!=1000;++i) {
        Trace::set(tracepoint::start);
        Trace::set(tracepoint::end);
        auto result = Trace::get(tracepoint::start, tracepoint::end);
        EXPECT_GT(accuracy, result);
    }

}


//TEST(tracing, particle_instant)
//{
//    for(auto i=0;i!=1000;++i) {
//        {
//            TraceParticle("a");
//        }
//        EXPECT_GT(accuracy, Trace::get("a_start", "a_stop"));
//    }
//}

// So neither usleep() std::this_thread::sleep_for() nor this really work reliably as the thread is sometimes
// just put to sleep. But distribution looks mostly accurate with a bunch of sharp outliers.
// Could rewrite tests to remove outliers but why bother...
void spinsleep(std::chrono::nanoseconds t)
{
    auto start = std::chrono::high_resolution_clock::now();
    volatile bool dummy = true;
    while(dummy)
    {
        if((std::chrono::high_resolution_clock::now()-start) >= t)
            return;
    }
}

TEST(DISABLED_tracing,spinsleep)
{
    size_t const accuracy = 1000;
    size_t const n_ns = 5000;

    for(auto i=0;i!=1000;++i) {
        auto start = std::chrono::high_resolution_clock::now();
        spinsleep(std::chrono::nanoseconds(n_ns));
        auto stop = std::chrono::high_resolution_clock::now();

        auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        std::cout << diff << std::endl;

        EXPECT_GT(n_ns+accuracy, diff);
        EXPECT_LT(n_ns-accuracy, diff);
    }
}


TEST(DISABLED_, sleep)
{
    Trace::set(tracepoint::start);
    Trace::set(tracepoint::end);

    size_t const accuracy = 1000;
    size_t const n_ns = 5000;

    for(auto i=0;i!=1000;++i) {
        Trace::set(tracepoint::start);
        spinsleep(std::chrono::nanoseconds(n_ns));
        Trace::set(tracepoint::end);
        EXPECT_GT(n_ns+accuracy, Trace::get(tracepoint::start, tracepoint::end));
        EXPECT_LT(n_ns-accuracy, Trace::get(tracepoint::start, tracepoint::end));
    }

}
