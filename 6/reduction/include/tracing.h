#pragma once
#include <chrono>
#include <string>
#include <map>
#include <unordered_map>
#include <stdexcept>


class Trace
{
public:
    using TimePoint = std::chrono::high_resolution_clock::time_point;

    static Trace& instance();

    static TimePoint get(std::string const& name);
    static uint64_t get(std::string const& name_start, std::string const& name_stop);
    static void set(std::string const& name);

private:
    Trace()=default;
    static Trace* _instance;
    std::unordered_map<std::string, TimePoint> _traces;
};

//RAII tracer, i.e create a TraceParticle and there will be traces when it enters and leaves scope
class TraceParticle
{
public:
    TraceParticle(std::string const& name)
        : _name(name)
    {
        Trace::set(name+"_start");
    }
    ~TraceParticle()
    {
        Trace::set(_name+"_stop");
    }
    //The only thing this thing is supposed to do is to be born and die again when going out of scope
    TraceParticle()=delete;
    TraceParticle(TraceParticle const&) = delete;
    TraceParticle(TraceParticle&&) = delete;
    TraceParticle& operator =(TraceParticle const&) = delete;
    TraceParticle operator=(TraceParticle&&) = delete;
private:
    std::string _name;
};