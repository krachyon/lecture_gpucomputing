\author{Sebastian Meßlinger, Niklas Euler}
\date{\today}
\title{GPU computing, Exercise 3}
\maketitle

## Paper review

### Debunking the 100X GPU vs. CPU Myth, Lee et. al. 

The researchers, all employees of Intel, benchmark a selection of commonly used algorithms on both a state of the art CPU and a GPU. In their analysis they go into detail how a given problem can or cannot make use of all the resources of both platforms. They lay out the algorithm's bottlenecks and offer some pointers as to how one can tune their code to maximum performance. 

This paper highlights that benchmarking and performance evaluation offer many pitfalls, among them badly chosen baselines that can skew results quite dramatically. The ratio of floating point operations the devices can perform as well as the ratio of their memory bandwidth and memory/cache sizes place an envelope on the achievable performance difference that was found to hold well, unless further hardware implemented functionality could be leveraged. Therefore any reported speedup beyond an order of magnitude appears implausible 

Especially when commercial interests are at play one should take a lot of care to ensure reproducibility and to clearly outline the area in which the results are supposed to be valid. It's a little baffling that previous research neglected to compare their results against established and vetted code that is readily available for problems like linear algebra but it also feels like it would have helped transparency along quite a bit to release the source-code underlying the results. If the performance figures rely heavily on hand tuning assembly for a single CPU model, it seems like a stretch to draw conclusions about CPUs in general. When solving a scientific or commercial problem, be it large and very computationally demanding, few can afford for engineers with an intimate architectural familiarity with the target platform to tease out the last iota of performance.


## Matrix multiplication on CPU

![Flops depending on problem size](./Ryzen 3600_matrix_flops.svg "abc"){width=100%}

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



