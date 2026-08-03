#ifndef MOTO_CORE_PARALLEL_JOB_HPP
#define MOTO_CORE_PARALLEL_JOB_HPP

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <fmt/core.h>
#include <stdexcept>
#include <string_view>

#ifdef MOTO_USE_OMP
#include <omp.h>
#define MAX_THREADS omp_get_max_threads()
#else
#define MAX_THREADS 1
#endif

namespace moto {
enum class parallel_block_order {
    reverse,
    forward,
};

inline parallel_block_order get_parallel_block_order() {
    static const parallel_block_order order = [] {
        const char *env = std::getenv("MOTO_PARALLEL_BLOCK_ORDER");
        if (env == nullptr) {
            // Default to chunking in the same order as the provided view.
            // Backward recursions pass an already-reversed traversal view, so
            // "forward" chunking still preserves traversal-locality.
            return parallel_block_order::forward;
        }
        const std::string_view value(env);
        if (value == "forward") {
            return parallel_block_order::forward;
        }
        return parallel_block_order::reverse;
    }();
    return order;
}

inline size_t normalize_parallel_jobs(size_t n_jobs) {
    const size_t max_threads = std::max<size_t>(1, static_cast<size_t>(MAX_THREADS));
    return std::clamp(n_jobs, size_t(1), max_threads);
}
/**
 * @brief parallel job for a range [start, stop)
 *
 * @tparam callback_t callback function must be invocable with size_t
 * @param start start idx
 * @param stop stop idx
 * @param callback
 */
template <typename callback_t>
inline void parallel_for(size_t start, size_t stop, callback_t &&callback,
                         size_t n_jobs = MAX_THREADS, bool no_except = false) {

    const size_t count = stop - start;
    if (count == 0) {
        return;
    }
    size_t n_threads = std::min(normalize_parallel_jobs(n_jobs), count);
    auto call = [&](size_t tid, size_t i) {
        if constexpr (std::invocable<callback_t, size_t>)
            callback(i);
        else if constexpr (std::invocable<callback_t, size_t, size_t>)
            callback(tid, i);
        else
            static_assert(false, "callback must be invocable with size_t or [size_t, size_t]");
    };
    if (n_threads == 1) {
        if (!no_except) {
            for (size_t i = start; i < stop; ++i) {
                call(0, i);
            }
            return;
        }
        try {
            for (size_t i = start; i < stop; ++i) {
                call(0, i);
            }
        } catch (const std::exception &e) {
            fmt::print("Exception in parallel_for inner loop: {}\n", e.what());
            throw std::runtime_error("Exception caught in parallel_for");
        }
        return;
    }
    const size_t chunk_size = (count + n_threads - 1) / n_threads;
    const bool reverse_order = get_parallel_block_order() == parallel_block_order::reverse;
    auto run_chunks = [&](auto &&body) {
#ifdef MOTO_USE_OMP
#pragma omp parallel for schedule(static, 1) num_threads(n_threads)
#endif
        for (size_t j = 0; j < n_threads; j++) {
            const size_t begin = reverse_order ? start + (n_threads - j - 1) * chunk_size
                                               : start + j * chunk_size;
            const size_t end = std::min(begin + chunk_size, stop);
            for (size_t i = begin; i < end; ++i) {
                body(j, i);
            }
        }
    };
    if (!no_except) {
        run_chunks(call);
        return;
    }

    std::atomic_bool except_caught = false;
    run_chunks([&](size_t tid, size_t i) {
        try {
            call(tid, i);
        } catch (const std::exception &e) {
            fmt::print("Exception in parallel_for inner loop: {}\n", e.what());
            except_caught.store(true, std::memory_order_relaxed);
        }
    });
    if (except_caught.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Exception caught in parallel_for");
    }
}

} // namespace moto

#endif // MOTO_CORE_PARALLEL_JOB_HPP
