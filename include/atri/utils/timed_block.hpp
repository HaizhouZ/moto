#ifndef ATRI_UTILS_TIMED_BLOCK_HPP
#define ATRI_UTILS_TIMED_BLOCK_HPP

#include <chrono>
#include <fmt/core.h>

namespace atri {
namespace utils {

template <size_t N>
struct string_literals {
    constexpr string_literals(const char (&str)[N]) {
        std::copy_n(str, N, value);
    }
    char value[N];
};

template <string_literals label>
struct timing_storage {
    using duration_t = std::chrono::high_resolution_clock::duration;
    duration_t durations{0};
    size_t count = 0;
    static auto &get() {
        static timing_storage<label> storage;
        return storage;
    }
    ~timing_storage() {
        auto avg = durations / count;
        auto per = std::chrono::duration_cast<std::chrono::microseconds>(avg).count();
        fmt::print("{}: {} us\n", label.value, per);
    }
};

#ifdef ENABLE_TIMED_BLOCK
#define timed_block_impl(label, ...)                                 \
    {                                                                \
        static std::chrono::high_resolution_clock::time_point start; \
        static std::chrono::high_resolution_clock::time_point end;   \
        static auto &timing = timing_storage<label>::get();          \
                                                                     \
        start = std::chrono::high_resolution_clock::now();           \
        __VA_ARGS__;                                                 \
        end = std::chrono::high_resolution_clock::now();             \
                                                                     \
        timing.durations += end - start;                             \
        timing.count++;                                              \
    }
// macro : [code blocks], will use the code block as labels
#define timed_block(...) timed_block_impl(#__VA_ARGS__, __VA_ARGS__)
// macro : [label, code blocks]
#define timed_block_labeled(label, ...) timed_block_impl(label, __VA_ARGS__)
#else
#define timed_block(...) __VA_ARGS__
#define timed_block_labeled(label, ...) __VA_ARGS__
#endif

} // namespace utils
} // namespace atri

#endif // ATRI_UTILS_TIMED_BLOCK_HPP