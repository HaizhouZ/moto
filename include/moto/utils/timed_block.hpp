#ifndef MOTO_UTILS_TIMED_BLOCK_HPP
#define MOTO_UTILS_TIMED_BLOCK_HPP

#include <algorithm>
#include <fmt/core.h>
#include <x86intrin.h> // For __rdtsc() on GCC/Clang and MSVC

namespace moto {
namespace utils {

/**
 * @brief string literal wrapper, used to store string literals as template arguments
 * @tparam N
 */
template <size_t N>
struct string_literals {
    constexpr string_literals(const char (&str)[N]) {
        std::copy_n(str, N, value);
    }
    char value[N];
};
// Function to read the Time Stamp Counter (TSC) with serialization
inline unsigned long long rdtsc() {
    return __rdtsc();
}

// Function to get the TSC frequency in cycles per second (Hz)
unsigned long long get_tsc_frequency();
/**
 * @brief perf_timer class to store timing information of labeled code blocks
 * @note the overhead of timing (calling @ref rdtscp) is 25-75 cycles
 * @tparam label
 */
template <string_literals label>
class perf_timer {
    double durations = 0.0;
    double elapsed_cycles;
    size_t count_ = 0;
    unsigned long long st_, ed_;

  public:
    /**
     * @brief get the timing storage object for manipulation
     *
     * @return auto& the storage
     */
    static auto &get() {
        static perf_timer<label> storage;
        return storage;
    }
    static size_t &count() { return get().count_; }
    static void start(size_t n = 1) {
#ifdef ENABLE_TIMED_BLOCK
        count() += n;
        get().st_ = rdtsc();
#endif // ENABLE_TIMED_BLOCK
    }
    static void end() {
#ifdef ENABLE_TIMED_BLOCK
        auto &t = perf_timer<label>::get();
        t.ed_ = rdtsc();
        t.elapsed_cycles += std::max((int)t.ed_ - (int)t.st_, 0);
#endif // ENABLE_TIMED_BLOCK
    }
    /**nt the average time
     *
     */
    ~perf_timer() {
#ifdef ENABLE_TIMED_BLOCK
        count_ = count_ == 0 ? 1 : count_;
        durations = elapsed_cycles / static_cast<double>(get_tsc_frequency()) * 1e6; // convert to microseconds
        auto avg = durations / count_;
        fmt::print("{}: {} us, count {} cycles {}\n", label.value, avg, count_, elapsed_cycles);
#endif // ENABLE_TIMED_BLOCK
    }
};

/// @def ENABLE_TIMED_BLOCK
/// define this to enable the timed_block, timed_block({code}) or timed_block_labeled(label, {code})

/// @brief start the timer, timed_block_start(label, n)
#define timed_block_start(label, ...) moto::utils::perf_timer<label>::start(__VA_ARGS__);
#define timed_block_end(label, ...) moto::utils::perf_timer<label>::end();
/// @brief start the timer, timed_block_labeled(label, n, {code})
#define timed_block_labeled_n(label, n, ...) \
    {                                        \
        timed_block_start(label);            \
        __VA_ARGS__                          \
        timed_block_end(label);              \
    }
/// @brief start the timer, timed_block_labeled(label, {code}), increment the count by 1
#define timed_block_labeled(label, ...) timed_block_labeled_n(label, 1, __VA_ARGS__)

} // namespace utils
} // namespace moto

#endif // MOTO_UTILS_TIMED_BLOCK_HPP