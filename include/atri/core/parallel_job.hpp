#ifndef ATRI_CORE_PARALLEL_JOB_HPP
#define ATRI_CORE_PARALLEL_JOB_HPP

#include <cstddef>
#include <omp.h>

namespace atri {

/**
 * @brief parallel job for a range [start, stop)
 *
 * @tparam callback_t callback function must be invocable with size_t
 * @param start start idx
 * @param stop stop idx
 * @param callback
 */
template <typename callback_t>
    requires std::invocable<callback_t, size_t>
inline void parallel_for(size_t start, size_t stop, callback_t &&callback) {
    size_t n_threads = omp_get_max_threads();
    size_t chunk_size = (stop - start + n_threads - 1) / n_threads;
#pragma omp parallel for schedule(static, 1)
    for (size_t j = 0; j < n_threads; j++) {
        size_t begin = j * chunk_size;
        size_t end = std::min(begin + chunk_size, stop); // Ensure bounds are within _nodes size
        for (size_t i = begin; i < end; ++i) {
            callback(i);
        }
    }
}

} // namespace atri

#endif // ATRI_CORE_PARALLEL_JOB_HPP