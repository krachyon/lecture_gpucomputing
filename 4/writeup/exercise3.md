
\author{Sebastian Meßlinger, Niklas Euler}
\date{\today}
\title{GPU computing, Exercise 3}
\maketitle

## 3.1 Memcpy

![Memcpy Device2Device](./mem.svg "abc"){width=80%}

Again there seems to be an unreasonably fast copy time for memory. The issues in the previous exercise where due to switching the order of arguments of cudaMemcpy() and ignoring the error. In this case though, both pointers are of the same type and there is no error reported by CUDA. Also initialization of the memory does not change the behaviour.
It would be feasible to implement some sort of copy-on-write optimization to avoid unnecessary memory movements, yet some research turned up no reference to such behaviour.

## 3.2 global coalesced

![memory access coalesced](./coalesced.svg "abc"){width=80%}

Both the thread per block and block count have been varied between 1 32 and 1024 respectively.
There seems to be slightly diminishing returns for high block sizes but the result is that more threads and more blocks increase throughput, as expected. It is therefore paramount to bundle memory access through thread blocks and the block grid.

## 3.3 and 3.4 global stride and offset  
![memory access coalesced](./global-stride.svg "abc"){width=70%}

![memory access coalesced](./global-offset.svg "abc"){width=70%}

The plot contains the aggregated results of varying blocks and threads in the same interval als in 3.2. As a single thread only copies a single element, there seems to be no penalty to either access with offset or stride. This is unexpected behaviour, as stride/offset would increase the number of cache lines that have to be fetched, but no such phenomenon is visible in the data. One possible explaination could be the additional overhead caused by each thread copying a single element.

