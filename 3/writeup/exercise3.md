
\author{Sebastian Meßlinger}
\date{\today}
\title{Exercise 3}
\maketitle

## 3.1

Again there seems to be an unreasonably fast copy time for memory. The issues in the previous exercise where due to switching the order of arguments of cudaMemcpy() and ignoring the error. In this case though, both pointers are of the same type and there is no error reported by CUDA. Also initialization of the memory does not change the behaviour.
It would be feasible to implement some sort of copy-on-write optimization to avoid unnecessary memory movements, yet some research turned up no reference to such behaviour.

## 2.2

## 2.3
