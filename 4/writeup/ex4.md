\author{Sebastian Meßlinger, Niklas Euler}
\date{\today}
\title{GPU computing, Exercise 3}
\maketitle

## Paper review

### Debunking the 100X GPU vs. CPU Myth, Lee et. al. 

The researchers, all employees of Intel, benchmark a selection of commonly used algorithms on both a state of the art CPU and a GPU. In their analysis they go into detail how a given problem can or cannot make use of all the resources of both platforms. They lay out the algorithm's bottlenecks and offer some pointers as to how one can tune their code to maximum performance. 

This paper highlights that benchmarking and performance evaluation offer many pitfalls, among them badly chosen baselines that can skew results quite dramatically. The ratio of floating point operations the devices can perform as well as the ratio of their memory bandwidth and memory/cache sizes place an envelope on the achievable performance difference that was found to hold well, unless further hardware implemented functionality could be leveraged. Therefore any reported speedup beyond an order of magnitude appears implausible 

Especially when commercial interests are at play one should take a lot of care to ensure reproducibility and to clearly outline the area in which the results are supposed to be valid. It's a little baffling that previous research neglected to compare their results against established and vetted code that is readily available for problems like linear algebra but it also feels like it would have helped transparency along quite a bit to release the source-code underlying the results. If the performance figures rely heavily on hand tuning assembly for a single CPU model, it seems like a stretch to draw conclusions about CPUs in general. When solving a scientific or commercial problem, be it large and very computationally demanding, few can afford for engineers with an intimate architectural familiarity with the target platform to tease out the last iota of performance.

### NVIDIA Tesla: A Unified Graphics and Computing Architecture, Lindholm et. al.

The paper "NVIDIA Tesla: A Unified Graphics and Computing Architecture" by Erik Lindholm, John Nickolls, Stuart Oberman and John Montrym" from 2006 introduces the novelties of the GPU architecture Tesla. The primary design objective was to introduce a unified architecture, which combines the formerly separate vertex and pixel processors, enabling the parallel computing capabilities of the GPU. Furthermore, the then new programming language CUDA was developed along the new hardware design.

On the hardware side, the architecture is ordered into texture/processor clusters, which each contain streaming multiprocessors, the new unified vertex/geometry/pixel processing units. In addition to this, there is the new single-instruction, multiple-thread operation mode, in which a single instruction is fetched and broadcasted to a group of 32 threads called a warp and consequently computed independently by every thread. These threads are bundled into thread blocks and distributed among the streaming multiprocessors; the architecture was conceived to be flexible in the number of SMs in order to remove any hurdles of parallelism.

Many of the concepts introduced in this paper still have heavy impact on GPU computing today. The unified approach to processing units is hugely successful and has been further implemented in all following architectures of NVIDIA. The C/C++ extension CUDA also is still in use in current day GPU computing, with no change in popularity in the near future to be expected.

## Shared memory, basic

These results where produced by the CMake target `shared_mem`, which has been run on a NVIDIA Geforce GTX 970 with appropriate compute capability 5.0 . An alternative implementation which has been tried to run on the cluster is available as `shared_mem_basic` and `shared_mem_conflict` but this produced unrealistic results on the cluster. It's likely that some errors introduced by avoiding specific compiler optimizations have something to do with this.

|||
|:----:|:----:|
|![shared2global](./s2g_1b.svg){width=50%}|![shared2global](./s2g.svg){width=50%}|

The total available bandwidth between shared and global memory seems to be around 10GB/s. When using a single thread block, one can see that the maximum transfer speed is only reached when using multiple threads (but saturates at around 500) and a transfer size of at least a few kilobytes. The overall bandwidth seems quite low compared as the card internal memory speed should be on the order of 100 GB/s, but as it is divided among the streaming multiprocessors and we're only utilizing one it's not unrealistic.

When using more blocks and therefore more SMs, we see that multiple thread blocks see a simultaneous bandwidth of around 10 GB/s when the number of threads is chosen high enough. As the blocks are not scheduled to run fully simultaneously when reaching higher block count, there are diminishing returns for the total bandwidth not captured by this graph.

|||
|:----:|:----:|
|![shared Memory Bandwidth](./r2s1000.svg  ){width=50%}|![shared Memory Bandwidth](./s2r1000.svg ){width=50%}|
|![shared Memory Bandwidth](./r2s8000.svg  ){width=50%}|![shared Memory Bandwidth](./s2r8000.svg ){width=50%}|
|![shared Memory Bandwidth](./r2s16000.svg ){width=50%}|![shared Memory Bandwidth](./s2r16000.svg){width=50%}|
|![shared Memory Bandwidth](./r2s24000.svg ){width=50%}|![shared Memory Bandwidth](./s2r24000.svg){width=50%}|
|![shared Memory Bandwidth](./r2s32000.svg ){width=50%}|![shared Memory Bandwidth](./s2r32000.svg){width=50%}|
|![shared Memory Bandwidth](./r2s40000.svg ){width=50%}|![shared Memory Bandwidth](./s2r40000.svg){width=50%}|
|![shared Memory Bandwidth](./r2s48000.svg ){width=50%}|![shared Memory Bandwidth](./s2r48000.svg){width=50%}|

When copying between registers and shared memory, the same ceiling of ~10Gb/s per thread block is reached. For increasing size of the data block more and more threads are required to be able to efficiently transfer the memory. While for small data sizes one can increase the overall throughput by adding more threadblocks, for larger sizes the returns diminish and the bandwidth per block even decreases as more and more blocks content for a limited amount of physical memory.

## Shared memory, conflicts

In this experiment a set of threads access the same value repeatedly. The shared memory access has been wrapped so that access that would be out of bounds starts hitting the start of the memory block again. Subsequent 4 Byte Float values map to subsequent memory banks. As long as there are more banks than Float values, an optimal bank usage is possible. With the introduction of a Stride, this optimal solution should degrade since the indices map begin to map to the same banks. Up until 32 threads, no such behaviour is visible. Several possible explanations are available, including compiler optimizations (mapping to free banks anyway), or advanced multicast/broadcast techniques are employed. The latter are only possible within warps of size 32. For higher thread counts, the computation time varies with the stride. For example, there are frequent decreases in computation time to the prior level for 64 threads for stride values of multiples of 4 (excluding 4 itself). This is explained by the index loopin around, begining at a stride of 8. The threads then access the same values in one bank, causing a cheaper multicast.


![shared Memory conflicts](./s2r_c.svg ){width=100%}

## Matrix multiplication on CPU

The code producing the graphs is available as CMake target `matrix_benchmark`, the correct function of the `matrix` library is verified by a few test cases in `matrix_test`

![Flops depending on problem size](./Ryzen 3600_matrix_flops.svg){width=100%}

As can clearly be seen the performance tends to drop with increasing problem size and shows large dips especially around $N~2^x$, e.g. $N=256$ and $N=1024$.
The noise in the regime $N <~ 200$ can probably be explained by a warmup taken for the branch predictor and differences in scheduling between the runs.

The dips in performance can be attributed to cache misses, running the binary under `perf stat` shows `60,36% backend cycles idle` and `4,39% L1-dcache misses` for a problem size of 300 but `87,26% backend cycles idle` and `50,16% L1-dcache misses` for 512. This implies that the CPU performs less than one instruction (0,79) per cycle, which is terrible considering the chip should be super-scalar.

For $N=5000$ the cache can be assumed to be fully ineffective, as the problem size exceeds the L3 cache and the sustained Flops/s are $0.54$ Gflop/s. As the measurements beyond $N=1500$ are increasingly noisy, most likely due to the fact that the L3 cache is shared between cores (and fully reserving a multi-core CPU with a single threaded workload seems unrealistic), this should only be seen as a rough estimate, but is a far cry from the $~5$ Gflops/s achieved at peak.

$C_5=A_5B_5$:

|||||
|:----:|:----:|:----:|:---:|
|0|30|60|90|120|
|0|40|80|120|160|
|0|50|100|150|200|
|0|60|120|180|240|
|0|70|140|210|280|
Matrix used for benchmark, see CMake target `matrix_test`

![Flops depending on problem size](./Xeon E5-1620_matrix_flops.svg "abc"){width=100%}

On the cluster a strange effect was seen, where in contiguous runs with increasing matrix size the performance would decay in an exponential looking way. This could be due to thermal issues where the CPU would step down the clock until a steady-state temperature can be maintained. Whatever the cause, the data is noisy enough, that it's dubious if the effect of any small optimization could even be measured accurately.
