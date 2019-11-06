
\author{Sebastian Meßlinger}
\date{\today}
\title{Exercise2}
\maketitle

## 2.1

||||
|:----:|:----:|:----:|
| ![async 1-16](./plots/0-5_async_True.svg "abc"){width=50%}| ![async 32-1024](./plots/5-10_async_True.svg "abc"){width=50%} |
| ![sync 1-16](./plots/0-5_async_False.svg "abc"){width=50%} |  ![sync 32-1024](./plots/5-10_async_False.svg "abc"){width=50%}|

There's an interesting effect with the first few kernel startups. Even though every kernel startup was repeated 50000 times and the dummy kernel was invoked that amount of times before the actual measurement, there seems to be a lengthy warmup phase, maybe relating to the branch predictor figuring out the code paths in the CUDA-driver.

As the iteration order was Async->blocks->threads the warmup hit the `async=True, blocks=1` run
Overall for there seems to be a penalty of around $7.5 \,\rm{μs}$ for synchronous startup of kernels.

## 2.2

![kernel execution time](./plots/wait.svg){width=80%}

A simple `clock64() < start+waitcycles` condition with an empty loop was used to busy wait a set amount of cycles.

The minimum kernel lifetime is on the order of 2μs or 1000 clock cycles, so it would only make sense to launch another kernel if the original kernel where to sleep longer than 4000 clock ticks or 4.1μs as to be able to actually do anything useful. The baseline was computed from the first 100 samples at low clock ticks.
Another interesting feature of the clocks to runtime dependency seems to be that the simple call to `clock64()` followed by a comparison seems to take on the order of 100 clock cycles, as only after increasing the target-cycles by roughly that amount we observer an increase in runtime.

## 2.3

![memory throughput with flawed measurement](./plots/memory_error.svg){width=110%}

Initially a measurement of allocating 
two blocks of memory on the host/device, zeroing them, performing a single `cudaMemcpy()` and measuring with `std::chrono::high_resolution_clock` was attempted, however for the paged memory this yielded unrealistic speeds where the copy time from/to paged host memory was not only faster than pinned memory, but also went faster than the theoretical maximum throughput of PCIe 2.0 (8GB/s).

![memory throughput](./plots/memory.svg){width=110%}

A second attempt was made to modify [this example by nvidia](https://raw.githubusercontent.com/NVIDIA-developer-blog/code-samples/master/series/cuda-cpp/optimize-data-transfers/bandwidthtest.cu) which allocates two blocks of host memory, writes ascending number into the first, copies it to the device, copies back from device to the second block and compares the host blocks. This seems to get around whatever optimization caused the copy with paged memory to be deferred/omitted. The result is along expectations, with copy operations with pinned memory being about 5 times quicker than paged memory operations after a constant overhead was no longer relevant at about 0.5 MB
