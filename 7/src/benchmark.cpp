#include <benchmark/benchmark.h>
#include "nbody.h"


static void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int n = 64; n <= 2048; n*=2)
        for (int thread : {32})//{1,2,4,8,32,40,50,80,128,256,300,512,1024})
            if(n>=thread)
                b->Args({n, thread});
}


static void BM_nbody_naive(benchmark::State& state)
{
    size_t N = state.range(0);
    size_t threads_per_block = state.range(1);

    for (auto _ : state) {

        auto time = run_leapfrog_aos(N, threads_per_block, iters);

        state.SetIterationTime(time.count());
    }
}

BENCHMARK(BM_nbody_naive)->Apply(CustomArguments)->UseManualTime();
BENCHMARK_MAIN();