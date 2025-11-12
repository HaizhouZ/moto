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
inline void parallel_for(size_t start, size_t stop, callback_t &&callback,
                         size_t n_jobs = MAX_THREADS, bool no_except = false) {

    size_t n_threads = n_jobs;
    size_t chunk_size = (stop - start + n_threads - 1) / n_threads;
    bool except_caught = false;
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
            auto call = [&]() {
                if constexpr (std::invocable<callback_t, size_t>)
                    callback(i);
                else if constexpr (std::invocable<callback_t, size_t, size_t>)
                    callback(j, i); // pass thread id as second argument
                else
                    static_assert(false, "callback must be invocable with size_t or [size_t, size_t]");
            };
            if (!no_except)
                call();
            else
                try {
                    call();
                } catch (const std::exception &e) {
                    fmt::print("Exception in parallel_for inner loop: {}\n", e.what());
                    except_caught = true;
                }
        }
    }
    if (except_caught) { /// @warning not complete
        throw std::runtime_error("Exception caught in parallel_for");
    }
    // } catch (...) {
    //     throw;
    // }
}

template <typename callback_t>
inline void sequential_for(size_t start, size_t stop, callback_t &&callback,
                           size_t n_jobs = MAX_THREADS, bool no_except = false) {

    size_t n_threads = n_jobs;
    size_t chunk_size = (stop - start + n_threads - 1) / n_threads;
    bool except_caught = false;
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
            if (except_caught)
                break;
            auto call = [&]() {
                if constexpr (std::invocable<callback_t, size_t>)
                    callback(i);
                else if constexpr (std::invocable<callback_t, size_t, size_t>)
                    callback(j, i); // pass thread id as second argument
                else
                    static_assert(false, "callback must be invocable with size_t or [size_t, size_t]");
            };
            if (!no_except)
                call();
            else
                try {
                    call();
                } catch (const std::exception &e) {
                    fmt::print("Exception in sequential_for inner loop: {}\n", e.what());
                    except_caught = true;
                }
        }
    }
    if (except_caught) { /// @warning not complete
        throw std::runtime_error("Exception caught in sequential_for");
    }
}

} // namespace moto

#endif // MOTO_CORE_PARALLEL_JOB_HPP