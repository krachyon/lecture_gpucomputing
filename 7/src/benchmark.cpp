#include <benchmark/benchmark.h>
#include "nbody.h"


static void CustomArguments(benchmark::internal::Benchmark* b) {
    int n = 1024;
        for (int thread = 1; thread <= 700; thread += 1) {
            b->Args({n, thread});
        }

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
static void BM_nbody_unaligned(benchmark::State& state)
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
BENCHMARK(BM_nbody_unaligned)->Apply(CustomArguments)->UseManualTime()->Unit(benchmark::kMillisecond);;
BENCHMARK(BM_nbody_shared)->Apply(CustomArguments)->UseManualTime()->Unit(benchmark::kMillisecond);;

BENCHMARK_MAIN();