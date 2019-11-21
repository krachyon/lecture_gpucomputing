#include "memoryWrapper.cuh"
#include <cassert>


void cudaMemcpy(Memory const& dest, Memory const& src) {
    assert(dest.size >= src.size);

    cudaMemcpyKind kind = cudaMemcpyDefault;

    if (dest.kind == memKind::device) {
        if (src.kind == memKind::device)
            kind = cudaMemcpyDeviceToDevice;
        else if (src.kind == memKind::pinned || src.kind == memKind::host)
            kind = cudaMemcpyHostToDevice;
        else
            throw (std::logic_error{"unknown memory type encountered"});
    } else if (dest.kind == memKind::pinned || dest.kind == memKind::host) {
        if (src.kind == memKind::device)
            kind = cudaMemcpyDeviceToHost;
        else if (src.kind == memKind::pinned || src.kind == memKind::host)
            kind = cudaMemcpyHostToHost;
        else
            throw (std::logic_error{"unknown memory type encountered"});
    }
    checkCuda(cudaMemcpy(dest._mem, src._mem, dest.size, kind));
}

