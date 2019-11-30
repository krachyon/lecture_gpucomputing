#include "include/tracing.h"

Trace* Trace::_instance = nullptr;

Trace& Trace::instance() {
    if (!_instance)
        _instance = new Trace;
    return *_instance;
}

Trace::TimePoint Trace::get(std::string const& name)
{
    if(instance()._traces.count(name) != 0)
        return instance()._traces[name];
    else
        throw(std::runtime_error("no trace point "+name));
}

uint64_t Trace::get(std::string const& name_start, std::string const& name_stop)
{
    if(instance()._traces.count(name_start) == 0)
        throw(std::runtime_error("no trace point "+name_start));
    if(instance()._traces.count(name_stop) == 0)
        throw(std::runtime_error("no trace point "+name_stop));

    return std::chrono::duration_cast<std::chrono::nanoseconds>
            (instance()._traces[name_stop]-instance()._traces[name_start]).count();
}


void Trace::set(std::string const& name)
{
    instance()._traces[name] = std::chrono::high_resolution_clock::now();
}