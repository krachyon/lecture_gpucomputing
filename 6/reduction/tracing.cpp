#include "include/tracing.h"

Trace* Trace::_instance = nullptr;

Trace& Trace::instance() {
    if (!_instance)
        _instance = new Trace;
    return *_instance;
}

Trace::TimePoint Trace::get(tracepoint name)
{
    return instance()._traces[size_t(name)];
}

// returns time delta between traces in nanoseconds
uint64_t Trace::get(tracepoint name_start, tracepoint name_stop)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>
            (instance()._traces[size_t(name_stop)]-instance()._traces[size_t(name_start)]).count();
}

void Trace::set(tracepoint name)
{
    instance()._traces[size_t(name)] = std::chrono::high_resolution_clock::now();
}