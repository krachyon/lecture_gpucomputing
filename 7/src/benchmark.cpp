#include <benchmark/benchmark.h>
#include "nbody.h"


static void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int n = 128; n <= 1024; n+=128)
        for (int thread : {1,2,8,16,32,50,128,200,512,700})
            b->Args({n, thread});
}


static void BM_nbody_naive(benchmark::State& state)
{
    size_t N = state.range(0);
    size_t threads_per_block = state.range(1);

    for (auto _ : state) {

        auto time = run_leapfrog_aos(N, threads_per_block, iters).first;

        state.SetIterationTime(time.count());
    }
    state.counters["N"] = N;
    state.counters["threads_per_block"] = threads_per_block;
}
static void BM_nbody_naive_unaligned(benchmark::State& state)
{
    size_t N = state.range(0);
    size_t threads_per_block = state.range(1);

    for (auto _ : state) {

        auto time = run_leapfrog_aos_unaligned(N, threads_per_block, iters).first;

        state.SetIterationTime(time.count());
    }
    state.counters["N"] = N;
    state.counters["threads_per_block"] = threads_per_block;
}

static void BM_nbody_shared(benchmark::State& state)
{
    size_t N = state.range(0);
    size_t threads_per_block = state.range(1);

    for (auto _ : state) {

    auto time = run_leapfrog_soa(N, threads_per_block, iters).first;

    state.SetIterationTime(time.count());
    }
    state.counters["N"] = N;
    state.counters["threads_per_block"] = threads_per_block;
}

BENCHMARK(BM_nbody_naive)->Apply(CustomArguments)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_nbody_naive_unaligned)->Apply(CustomArguments)->UseManualTime()->Unit(benchmark::kMillisecond);;
BENCHMARK(BM_nbody_shared)->Apply(CustomArguments)->UseManualTime()->Unit(benchmark::kMillisecond);;

BENCHMARK_MAIN();