#include "tracing.h"

Trace* Trace::_instance = nullptr;

Trace& Trace::instance() {
    if (!_instance)
        _instance = new Trace;
    return *_instance;
}

Trace::TimePoint Trace::get(std::string const& name)
{
    if(instance()._traces.count("name") != 0)
        return instance()._traces[name];
    else
        throw(std::runtime_error("no trace point "+name));
}
void Trace::set(std::string const& name)
{
    instance()._traces[name] = std::chrono::high_resolution_clock::now();
}