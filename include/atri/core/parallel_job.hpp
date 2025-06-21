#ifndef ATRI_CORE_PARALLEL_JOB_HPP
#define ATRI_CORE_PARALLEL_JOB_HPP

#include <cstddef>

#ifdef ATRI_USE_OMP
#include <omp.h>
#define MAX_THREADS omp_get_max_threads()
#else
#define MAX_THREADS 1
#endif

namespace atri {

/**
 * @brief parallel job for a range [start, stop)
 *
 * @tparam callback_t callback function must be invocable with size_t
 * @param start start idx
 * @param stop stop idx
 * @param callback
 */
template <typename callback_t, bool reverse_block = true>
    requires std::invocable<callback_t, size_t>
inline void parallel_for(size_t start, size_t stop, callback_t &&callback) {

    size_t n_threads = MAX_THREADS;
    size_t chunk_size = (stop - start + n_threads - 1) / n_threads;
#ifdef ATRI_USE_OMP
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
            callback(i);
        }
    }
}

template <typename callback_t>
    requires std::invocable<callback_t, size_t>
inline void sequential_for(size_t start, size_t stop, callback_t &&callback) {

    size_t n_threads = MAX_THREADS;
    size_t chunk_size = (stop - start + n_threads - 1) / n_threads;
#ifdef ATRI_USE_OMP
#pragma omp parallel for ordered schedule(static, 1)
#endif
    for (size_t j = 0; j < n_threads; j++) {
        size_t begin = j * chunk_size;
        size_t end = std::min(begin + chunk_size, stop); // Ensure bounds are within _nodes size
#ifdef ATRI_USE_OMP
#pragma omp ordered
#endif
        for (size_t i = begin; i < end; ++i) {
            callback(i);
        }
    }
}

} // namespace atri

#endif // ATRI_CORE_PARALLEL_JOB_HPP