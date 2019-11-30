#pragma once
#include <chrono>
#include <string>
//#include <map>
//#include <unordered_map>
#include <stdexcept>
#include <array>

enum class tracepoint: size_t
{
    start,
    end,
    copy_start,
    copy_end,
    backcopy_start,
    backcopy_end,
    _SIZE
};


class Trace
{
public:
    using TimePoint = std::chrono::high_resolution_clock::time_point;

    static Trace& instance();

    static TimePoint get(tracepoint name);
    static uint64_t get(tracepoint name_start, tracepoint name_stop);
    static void set(tracepoint name);

private:
    Trace()=default;
    static Trace* _instance;
    std::array<TimePoint,static_cast<size_t>(tracepoint::_SIZE)> _traces;
};

//TODO this doesn't make that much sense without being able to specify strings for tracepoints
// sadly using map/unordered_map for this purpose is too slow

////RAII tracer, i.e create a TraceParticle and there will be traces when it enters and leaves scope
//class TraceParticle
//{
//public:
//    TraceParticle(std::string const& name)
//        : _name(name)
//    {
//        Trace::set(name+"_start");
//    }
//    ~TraceParticle()
//    {
//        Trace::set(_name+"_stop");
//    }
//    //The only thing this thing is supposed to do is to be born and die again when going out of scope
//    TraceParticle()=delete;
//    TraceParticle(TraceParticle const&) = delete;
//    TraceParticle(TraceParticle&&) = delete;
//    TraceParticle& operator =(TraceParticle const&) = delete;
//    TraceParticle operator=(TraceParticle&&) = delete;
//private:
//    std::string _name;
//};