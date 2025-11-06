#include "profiler.h"
#include <mutex>
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace coacd {
namespace profiler {

static bool enabled = false;
static std::mutex mutex;
static std::map<std::string, std::pair<double, int>> timings; // name -> (total_ms, count)

void enable() {
    std::lock_guard<std::mutex> lock(mutex);
    enabled = true;
}

void disable() {
    std::lock_guard<std::mutex> lock(mutex);
    enabled = false;
}

bool is_enabled() {
    std::lock_guard<std::mutex> lock(mutex);
    return enabled;
}

void reset() {
    std::lock_guard<std::mutex> lock(mutex);
    timings.clear();
}

void ScopedTimer::record(const std::string& name, double duration_ms) {
    if (!is_enabled()) return;
    
    std::lock_guard<std::mutex> lock(mutex);
    auto& entry = timings[name];
    entry.first += duration_ms;
    entry.second += 1;
}

std::vector<TimingEntry> get_timings() {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<TimingEntry> result;
    for (const auto& kv : timings) {
        result.push_back({kv.first, kv.second.first, kv.second.second});
    }
    return result;
}

void print_summary() {
    auto entries = get_timings();
    if (entries.empty()) {
        logger::info("No profiling data collected.");
        return;
    }
    
    // Sort by total time descending
    std::sort(entries.begin(), entries.end(), 
              [](const TimingEntry& a, const TimingEntry& b) { return a.duration_ms > b.duration_ms; });
    
    double total_ms = 0;
    for (const auto& e : entries) {
        total_ms += e.duration_ms;
    }
    
    logger::info("========================================");
    logger::info("PROFILING SUMMARY");
    logger::info("========================================");
    logger::info("{:<40} {:>12} {:>8} {:>12} {:>8}", 
                 "Function", "Total (ms)", "Calls", "Avg (ms)", "% Total");
    logger::info("----------------------------------------");
    
    for (const auto& e : entries) {
        double avg_ms = e.count > 0 ? e.duration_ms / e.count : 0;
        double percent = total_ms > 0 ? (e.duration_ms / total_ms * 100.0) : 0;
        
        std::ostringstream oss;
        oss << std::left << std::setw(40) << e.name
            << std::right << std::setw(12) << std::fixed << std::setprecision(2) << e.duration_ms
            << std::setw(8) << e.count
            << std::setw(12) << std::fixed << std::setprecision(2) << avg_ms
            << std::setw(8) << std::fixed << std::setprecision(1) << percent << "%";
        logger::info(oss.str());
    }
    
    logger::info("----------------------------------------");
    logger::info("Total profiled time: {:.2f} ms ({:.2f} s)", total_ms, total_ms / 1000.0);
    logger::info("========================================");
}

} // namespace profiler
} // namespace coacd
