#ifndef ATRI_UTILS_TIMED_BLOCK_HPP
#define ATRI_UTILS_TIMED_BLOCK_HPP

#include <chrono>
#include <fmt/core.h>

namespace atri {
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

/**
 * @brief timing_storage class to store timing information of labeled code blocks
 *
 * @tparam label
 */
template <string_literals label>
struct timing_storage {
    using duration_t = std::chrono::high_resolution_clock::duration;
    duration_t durations{0};
    size_t count = 0;
    /**
     * @brief get the timing storage object for manipulation
     *
     * @return auto& the storage
     */
    static auto &get() {
        static timing_storage<label> storage;
        return storage;
    }
    /**
     * @brief Destroy the timing storage object, meanwhile print the average time
     *
     */
    ~timing_storage() {
        auto avg = durations / count;
        auto per = std::chrono::duration_cast<std::chrono::microseconds>(avg).count();
        fmt::print("{}: {} us\n", label.value, per);
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
        static std::chrono::high_resolution_clock::time_point start;     \
        static std::chrono::high_resolution_clock::time_point end;       \
        static auto &timing = atri::utils::timing_storage<label>::get(); \
                                                                         \
        start = std::chrono::high_resolution_clock::now();               \
        __VA_ARGS__;                                                     \
        end = std::chrono::high_resolution_clock::now();                 \
                                                                         \
        timing.durations += end - start;                                 \
        timing.count++;                                                  \
    }
#define timed_block(...) timed_block_impl(#__VA_ARGS__, __VA_ARGS__)
#define timed_block_labeled(label, ...) timed_block_impl(label, __VA_ARGS__)
#else
#define timed_block(...) __VA_ARGS__
#define timed_block_labeled(label, ...) __VA_ARGS__
#endif

} // namespace utils
} // namespace atri

#endif // ATRI_UTILS_TIMED_BLOCK_HPP