#ifndef MOTO_UTILS_TIMED_BLOCK_HPP
#define MOTO_UTILS_TIMED_BLOCK_HPP

#include <fmt/core.h>
#include <algorithm>
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
inline unsigned long long rdtscp() {
    unsigned int aux;
    return __rdtscp(&aux);
}

// Function to get the TSC frequency in cycles per second (Hz)
unsigned long long get_tsc_frequency();
/**
 * @brief timing_storage class to store timing information of labeled code blocks
 *
 * @tparam label
 */
template <string_literals label>
struct timing_storage {
    double durations = 0.0;
    unsigned long long elapsed_cycles;
    size_t count_ = 0;
    /**
     * @brief get the timing storage object for manipulation
     *
     * @return auto& the storage
     */
    static auto &get() {
        static timing_storage<label> storage;
        return storage;
    }
    static size_t& count() { return get().count_; }
    /**
     * @brief Destroy the timing storage object, meanwhile print the average time
     *
     */
    ~timing_storage() {
        count_ = count_ == 0 ? 1 : count_;
        durations = elapsed_cycles / static_cast<double>(get_tsc_frequency()) * 1e6; // convert to microseconds
        auto avg = durations / count_;
        fmt::print("{}: {} us, count {}\n", label.value, avg, count_);
    }
};

/// @def timed_block
/// [code blocks], will use the code block as labels

/// @def timed_block_labeled
/// [label, code blocks]

/// @def ENABLE_TIMED_BLOCK
/// define this to enable the timed_block, timed_block({code}) or timed_block_labeled(label, {code})

#ifdef ENABLE_TIMED_BLOCK
/// @def ENABLE_TIMED_BLOCK
/// define this to enable the timed_block, timed_block({code}) or timed_block_labeled(label, {code})
#define timed_block_impl(label, ...)                                     \
    {                                                                    \
        static auto &timing = moto::utils::timing_storage<label>::get(); \
        timing.count_++;                                                  \
        auto start = moto::utils::rdtscp();                              \
        __VA_ARGS__;                                                     \
        auto end = moto::utils::rdtscp();                                \
        timing.elapsed_cycles += end - start;                            \
    }
#define timed_block(...) timed_block_impl(#__VA_ARGS__, __VA_ARGS__)
#define timed_block_labeled(label, ...) timed_block_impl(label, __VA_ARGS__)
#else
#define timed_block(...) __VA_ARGS__
#define timed_block_labeled(label, ...) __VA_ARGS__
#endif

} // namespace utils
} // namespace moto

#endif // MOTO_UTILS_TIMED_BLOCK_HPP