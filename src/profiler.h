#pragma once
#include <chrono>
#include <string>
#include <map>
#include <vector>
#include "logger.h"

namespace coacd {
namespace profiler {

struct TimingEntry {
    std::string name;
    double duration_ms;
    int count;
};

class ScopedTimer {
public:
    ScopedTimer(const std::string& name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        record(name_, duration / 1000.0);
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
    static void record(const std::string& name, double duration_ms);
};

void enable();
void disable();
bool is_enabled();
void reset();
void print_summary();
std::vector<TimingEntry> get_timings();

} // namespace profiler
} // namespace coacd
