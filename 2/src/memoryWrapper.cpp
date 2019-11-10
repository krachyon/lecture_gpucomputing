#include "memoryWrapper.h"
#include <cassert>
#include <stdexcept>
#include <sstream>

void cudaMemcpy(Memory const& dest, Memory const& src)
{
    assert(dest.size>=src.size);

    cudaMemcpyKind kind = cudaMemcpyDefault;

    if (dest.kind==memKind::device) {
        if (src.kind==memKind::device)
            kind = cudaMemcpyDeviceToDevice;
        else if (src.kind==memKind::pinned || src.kind==memKind::host)
            kind = cudaMemcpyHostToDevice;
        else
            throw (std::logic_error{"unknown memory type encountered"});
    }
    else if (dest.kind==memKind::pinned || dest.kind==memKind::host) {
        if (src.kind==memKind::device)
            kind = cudaMemcpyDeviceToHost;
        else if (src.kind==memKind::pinned || src.kind==memKind::host)
            kind = cudaMemcpyHostToHost;
        else
            throw (std::logic_error{"unknown memory type encountered"});
    }

    cudaError_t err = cudaMemcpy(dest._mem, src._mem, dest.size, kind);

    if (err!=cudaSuccess) {
        std::stringstream message{""};
        message << "cuda Error: " << err << ": " << cudaGetErrorString(err);
        throw (std::runtime_error(message.str()));
    }
}
