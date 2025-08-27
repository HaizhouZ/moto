#ifndef MOTO_CORE_PARALLEL_JOB_HPP
#define MOTO_CORE_PARALLEL_JOB_HPP

#include <cstddef>

#ifdef MOTO_USE_OMP
#include <omp.h>
#define MAX_THREADS omp_get_max_threads()
#else
#define MAX_THREADS 1
#endif

namespace moto {
template <class... T>
constexpr bool always_false = false;
/**
 * @brief Get the number of threads available for parallel execution
 *
 * @return size_t number of threads
 */
inline size_t get_num_threads() {
    return MAX_THREADS;
}
/**
 * @brief parallel job for a range [start, stop)
 *
 * @tparam callback_t callback function must be invocable with size_t
 * @param start start idx
 * @param stop stop idx
 * @param callback
 */
template <typename callback_t, bool reverse_block = true>
inline void parallel_for(size_t start, size_t stop, callback_t &&callback, size_t n_jobs = MAX_THREADS) {

    size_t n_threads = n_jobs;
    size_t chunk_size = (stop - start + n_threads - 1) / n_threads;
    // try {
#ifdef MOTO_USE_OMP
    omp_set_num_threads(n_threads);
#pragma omp parallel for schedule(static, 1)
#endif
    for (size_t j = 0; j < n_threads; j++) {
        size_t begin;
        if constexpr (reverse_block)
            begin = (n_threads - j - 1) * chunk_size;
        else
            begin = j * chunk_size;
        size_t end = std::min(begin + chunk_size, stop); // Ensure bounds are within _nodes size
        for (size_t i = begin; i < end; ++i) {
            // for (size_t i = end-1; i >= begin && i < end; i--) {
            if constexpr (std::invocable<callback_t, size_t>)
                callback(i);
            else if constexpr (std::invocable<callback_t, size_t, size_t>)
                callback(j, i); // pass thread id as second argument
            else
                static_assert(always_false<callback_t>,
                              "callback must be invocable with size_t or [size_t, size_t]");
        }
    }
    // } catch (...) {
    //     throw;
    // }
}

template <typename callback_t>
inline void sequential_for(size_t start, size_t stop, callback_t &&callback, size_t n_jobs = MAX_THREADS) {

    size_t n_threads = n_jobs;
    size_t chunk_size = (stop - start + n_threads - 1) / n_threads;
    // try {
#ifdef MOTO_USE_OMP
    omp_set_num_threads(n_threads);
#pragma omp parallel for ordered schedule(static, 1)
#endif
    for (size_t j = 0; j < n_threads; j++) {
        size_t begin = j * chunk_size;
        size_t end = std::min(begin + chunk_size, stop); // Ensure bounds are within _nodes size
#ifdef MOTO_USE_OMP
#pragma omp ordered
#endif
        for (size_t i = begin; i < end; ++i) {
            if constexpr (std::invocable<callback_t, size_t>)
                callback(i);
            else if constexpr (std::invocable<callback_t, size_t, size_t>)
                callback(j, i); // pass thread id as second argument
            else
                static_assert(always_false<callback_t>,
                              "callback must be invocable with size_t or [size_t, size_t]");
        }
    }
    // } catch (...) {
    //     throw;
    // }
}

} // namespace moto

#endif // MOTO_CORE_PARALLEL_JOB_HPP