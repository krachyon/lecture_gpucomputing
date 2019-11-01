# Exercise 2
## 2.1
||||
|:----:|:----:|:----:|
| ![](./plots/0-5_async_True.svg "abc")| ![](./plots/5-10_async_True.svg "abc") |
| ![](./plots/0-5_async_False.svg "abc") |  ![](./plots/5-10_async_False.svg "abc") |

There's an interesting effect with the first few kernel startups. Even though every kernel startup was repeated 50000 times and the dummy kernel was invoked that amount of times before the actual measurement, there seems to be a lengthy warmup phase, maybe relating to the branch predictor figuring out the code paths in the CUDA-driver.

As the iteration order was Async->blocks->threads the warmup hit the `async=True, blocks=1` run
Overall for there seems to be a penalty of around $7.5 \,\rm{μs}$ for synchronous startup of kernels.
