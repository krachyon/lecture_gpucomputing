#include <benchmark/benchmark.h>
#include "nbody.h"


static void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int n = 64; n <= 1024; n*=2)
        for (int thread : {32, 1024})
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
}
static void BM_nbody_naive_unaligned(benchmark::State& state)
{
    size_t N = state.range(0);
    size_t threads_per_block = state.range(1);

    for (auto _ : state) {

        auto time = run_leapfrog_aos_unaligned(N, threads_per_block, iters).first;

        state.SetIterationTime(time.count());
    }}

BENCHMARK(BM_nbody_naive)->Apply(CustomArguments)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_nbody_naive_unaligned)->Apply(CustomArguments)->UseManualTime()->Unit(benchmark::kMillisecond);;

BENCHMARK_MAIN();